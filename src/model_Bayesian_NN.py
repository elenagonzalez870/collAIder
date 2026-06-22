"""Bayesian-NN backend: the deterministic NN backend (model_NN.py) with a
Bayesian first layer added for uncertainty.

The first Linear layer of each network is replaced by a torchbnn.BayesLinear,
whose posterior MEANS are frozen to collAIder's trained NN weights and whose
later layers are copied verbatim. Two consequences:

  * Deterministic mode (mean weights, sampling noise zeroed) reproduces the
    deterministic NN backend EXACTLY, so the point prediction is unchanged
    (verified bit-for-bit against best_NN_class_model.pt / best_NN_reg_model.pt,
    see verify_against_nn). This is the "augmented model matches the original in
    deterministic mode" check.
  * Sampling mode (many stochastic forward passes) gives a predictive spread:
    a per-component standard deviation for the regressor, and the standard
    total / aleatoric / epistemic decomposition for the classifier.

Only the first-layer widths carry uncertainty. They keep their PRIOR scale here;
to get calibrated widths, freeze-train them on these same checkpoints with
collAIder-training (freeze_means mode, init_checkpoint = the NN checkpoint) and
point CLASS_WIDTHS / REG_WIDTHS at the resulting files. With prior widths the
deterministic prediction is still exact, but the sampled spread is not
calibrated.

Same public API (process_encounters) and same physics front-end as model_NN.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchbnn as bnn
from numpy.core.multiarray import _reconstruct
torch.serialization.add_safe_globals([_reconstruct])

from encounter_physics import EncounterRegimeClassifier

# Stochastic forward passes used for the uncertainty estimate.
DEFAULT_N_SAMPLES = 100
# BayesLinear prior. The means are frozen to the trained NN weights; only the
# widths are Bayesian, initialized at this prior scale.
PRIOR_MU = 0.0
PRIOR_SIGMA = 0.1


def _bayes_first_layer():
    return bnn.BayesLinear(prior_mu=PRIOR_MU, prior_sigma=PRIOR_SIGMA,
                           in_features=5, out_features=512)


class RegressionBayesNet(nn.Module):
    """RegressionNeuralNetwork from model_NN, with a Bayesian first layer."""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            _bayes_first_layer(),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return F.softmax(logits, dim=-1)  # mass fractions, as in model_NN


class ClassificationBayesNet(nn.Module):
    """ClassificationNeuralNetwork from model_NN, with a Bayesian first layer."""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            _bayes_first_layer(),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)  # raw logits, as in model_NN


def load_frozen_means(model, det_state, widths_state=None):
    """Load a deterministic model_NN state_dict as the frozen posterior means of
    the Bayesian first layer (weight -> weight_mu, bias -> bias_mu) and the
    verbatim weights of every later layer. Optionally load trained first-layer
    widths (the *_log_sigma tensors) from widths_state; otherwise the widths
    keep their prior scale."""
    bayes_state = model.state_dict()
    for key, value in det_state.items():
        if key == 'linear_relu_stack.0.weight':
            bayes_state['linear_relu_stack.0.weight_mu'] = value
        elif key == 'linear_relu_stack.0.bias':
            bayes_state['linear_relu_stack.0.bias_mu'] = value
        elif key in bayes_state:
            bayes_state[key] = value
        else:
            raise KeyError(f"unexpected key in deterministic checkpoint: {key}")
    if widths_state is not None:
        for key in ('linear_relu_stack.0.weight_log_sigma',
                    'linear_relu_stack.0.bias_log_sigma'):
            if key in widths_state:
                bayes_state[key] = widths_state[key]
    model.load_state_dict(bayes_state)


def set_means_only(model, enabled):
    """Toggle deterministic (mean-weight) forward passes on the Bayesian first
    layer. enabled=True zeros the sampling noise so the layer uses weight_mu /
    bias_mu exactly (reproducing model_NN); enabled=False samples again."""
    first_layer = model.linear_relu_stack[0]
    if not isinstance(first_layer, bnn.BayesLinear):
        return
    if enabled:
        first_layer.weight_eps = torch.zeros_like(first_layer.weight_log_sigma)
        first_layer.bias_eps = torch.zeros_like(first_layer.bias_log_sigma)
    else:
        first_layer.weight_eps = None
        first_layer.bias_eps = None


class PerformCollision(nn.Module):

    _models_loaded = False
    _classification_model = None
    _regression_model = None
    _input_mean = None
    _input_std = None

    # Frozen means come from collAIder's deployed NN checkpoints.
    CLASS_CHECKPOINT = "../models/best_NN_class_model.pt"
    REG_CHECKPOINT = "../models/best_NN_reg_model.pt"
    # Optional calibrated widths: freeze-means checkpoints trained on the two
    # checkpoints above. Leave None to use the prior width scale.
    CLASS_WIDTHS = None
    REG_WIDTHS = None

    def __init__(self, age, pericenter, velocity_inf, mass1, mass2):
        super(PerformCollision, self).__init__()

        if not PerformCollision._models_loaded:
            PerformCollision._classification_model = ClassificationBayesNet()
            PerformCollision._regression_model = RegressionBayesNet()

            class_ckpt = torch.load(self.CLASS_CHECKPOINT, map_location="cpu", weights_only=False)
            reg_ckpt = torch.load(self.REG_CHECKPOINT, map_location="cpu", weights_only=False)

            class_widths = None
            reg_widths = None
            if self.CLASS_WIDTHS is not None:
                class_widths = torch.load(self.CLASS_WIDTHS, map_location="cpu", weights_only=False)["model_state_dict"]
            if self.REG_WIDTHS is not None:
                reg_widths = torch.load(self.REG_WIDTHS, map_location="cpu", weights_only=False)["model_state_dict"]

            load_frozen_means(PerformCollision._classification_model, class_ckpt["model_state_dict"], class_widths)
            load_frozen_means(PerformCollision._regression_model, reg_ckpt["model_state_dict"], reg_widths)

            # Same normalization statistics as model_NN (stored in the checkpoint).
            PerformCollision._input_mean = class_ckpt["train_mean"]
            PerformCollision._input_std = class_ckpt["train_std"]

            PerformCollision._classification_model.eval()
            PerformCollision._regression_model.eval()
            PerformCollision._models_loaded = True

        self.classification_model = PerformCollision._classification_model
        self.regression_model = PerformCollision._regression_model
        self.input_mean = PerformCollision._input_mean
        self.input_std = PerformCollision._input_std

        X = self.Transform(age, pericenter, velocity_inf, mass1, mass2)
        self.X_norm = self.Standard_Scale(X, self.input_mean, self.input_std)
        self.m_ini_tot = mass1 + mass2

    def Transform(self, age, pericenter, velocity_inf, mass1, mass2):
        # Identical features to model_NN.
        X = np.array([
            np.log10(age + 0.001),
            np.log10(pericenter + 0.1),
            np.log10(velocity_inf + 10.),
            np.log(mass1),
            np.log(mass2)], dtype=np.float32)
        return X

    def Standard_Scale(self, X, mean, std):
        X_norm = (X - mean) / std
        return torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0)

    def PerformClassification(self, n_samples=DEFAULT_N_SAMPLES):
        """Deterministic class + softmax probabilities (identical to model_NN),
        plus the sampled total / aleatoric / epistemic uncertainty (nats)."""
        eps = 1e-12
        set_means_only(self.classification_model, True)
        with torch.no_grad():
            det_logits = self.classification_model(self.X_norm)
            class_probs = F.softmax(det_logits, dim=1).squeeze(0).tolist()
        predicted_class = int(np.argmax(class_probs))

        set_means_only(self.classification_model, False)
        with torch.no_grad():
            probs = np.stack([
                F.softmax(self.classification_model(self.X_norm), dim=1).squeeze(0).numpy()
                for _ in range(n_samples)], axis=0)  # (n_samples, 4)
        set_means_only(self.classification_model, True)  # leave it deterministic

        mean_p = probs.mean(axis=0)
        total = float(-(mean_p * np.log(mean_p + eps)).sum())
        aleatoric = float((-(probs * np.log(probs + eps)).sum(axis=1)).mean())
        uncertainty = {
            "total": total,
            "aleatoric": aleatoric,
            "epistemic": total - aleatoric,  # mutual information (BALD)
        }
        return predicted_class, class_probs, uncertainty

    def PerformRegression(self, n_samples=DEFAULT_N_SAMPLES):
        """Deterministic mass components in Msun (identical to model_NN) plus the
        per-sample mass components (Msun) for the predictive spread."""
        set_means_only(self.regression_model, True)
        with torch.no_grad():
            det = self.regression_model(self.X_norm).squeeze(0).numpy()
        predicted_values = [float(v * self.m_ini_tot) for v in det]

        set_means_only(self.regression_model, False)
        with torch.no_grad():
            samples = np.stack([
                self.regression_model(self.X_norm).squeeze(0).numpy()
                for _ in range(n_samples)], axis=0)  # (n_samples, 3) fractions
        set_means_only(self.regression_model, True)  # leave it deterministic
        sample_masses = samples * self.m_ini_tot
        return predicted_values, sample_masses


def process_collisions(pred_class, pred_reg):
    """Enforce consistency between the class and the regressed masses, identical
    to model_NN.process_collisions."""
    if pred_class == 0:  # both stars destroyed
        pred_reg[2] += pred_reg[0] + pred_reg[1]
        pred_reg[0], pred_reg[1] = 0.0, 0.0
    if pred_class == 1 or pred_class == 3:  # one star survives (merger or stripped)
        pred_reg[2] += pred_reg[1]
        pred_reg[1] = 0.0
    return pred_class, pred_reg


def process_encounters(ages, masses1, masses2, pericenters, velocities_inf, n_samples=DEFAULT_N_SAMPLES):
    """Process multiple stellar encounters with the Bayesian-NN backend.

    Same inputs and outputs as model_NN.process_encounters, with two extra keys:
      'predicted_values_std' : 1-sigma spread of each mass component (Msun),
      'uncertainty'          : {'total','aleatoric','epistemic'} for the class
                               prediction (nats; epistemic is mutual information).
    Tidal captures and flybys are set by the physics front-end, so their spread
    and uncertainty are zero.
    """
    ages = np.atleast_1d(ages)
    masses1 = np.atleast_1d(masses1)
    masses2 = np.atleast_1d(masses2)
    pericenters = np.atleast_1d(pericenters)
    velocities_inf = np.atleast_1d(velocities_inf)

    n = len(ages)
    if not all(len(arr) == n for arr in [masses1, masses2, pericenters, velocities_inf]):
        raise ValueError("All input arrays must have the same length")

    classifier = EncounterRegimeClassifier()
    results = []

    for i in range(n):
        age = ages[i]
        mass1 = masses1[i]
        mass2 = masses2[i]
        pericenter = pericenters[i]
        velocity_inf = velocities_inf[i]

        flag = False
        if mass1 < mass2:
            mass1, mass2 = mass2, mass1
            flag = True

        regime = classifier.classify_encounter(
            age=age, mass1=mass1, mass2=mass2,
            pericenter=pericenter, velocity_inf=velocity_inf)

        if regime == 'collision':
            regime_flag = -1
            collision = PerformCollision(age, pericenter, velocity_inf, mass1, mass2)
            predicted_class, class_probs, uncertainty = collision.PerformClassification(n_samples)
            det_values, sample_masses = collision.PerformRegression(n_samples)

            # Deterministic prediction goes through the consistency rule exactly
            # like model_NN; the same rule is applied per sample so the reported
            # spread is the spread of the consistency-corrected masses.
            predicted_class, predicted_values = process_collisions(predicted_class, det_values)
            corrected = np.array([
                process_collisions(predicted_class, list(s))[1] for s in sample_masses])
            predicted_values_std = [float(s) for s in corrected.std(axis=0)]

        elif regime == 'tidal_capture':
            regime_flag = -2
            predicted_class = 1
            predicted_values = [1. * (mass1 + mass2), 0., 0.]
            predicted_values_std = [0., 0., 0.]
            class_probs = [0., 1., 0., 0.]
            uncertainty = {"total": 0.0, "aleatoric": 0.0, "epistemic": 0.0}

        else:  # flyby
            regime_flag = -3
            predicted_class = 2
            predicted_values = [mass1, mass2, 0]
            predicted_values_std = [0., 0., 0.]
            class_probs = [0., 0., 1., 0.]
            uncertainty = {"total": 0.0, "aleatoric": 0.0, "epistemic": 0.0}

        if flag == True and int(predicted_class) != 1:
            predicted_values[0], predicted_values[1] = predicted_values[1], predicted_values[0]
            predicted_values_std[0], predicted_values_std[1] = predicted_values_std[1], predicted_values_std[0]

        results.append({
            'regime_flag': regime_flag,
            'predicted_class': int(predicted_class),
            'predicted_values': [float(v) for v in predicted_values],
            'predicted_values_std': [float(v) for v in predicted_values_std],
            'class_probs': [float(p) for p in class_probs],
            'uncertainty': {k: float(v) for k, v in uncertainty.items()}})

    return results


def verify_against_nn(n=3000, seed=0):
    """Check that deterministic mode reproduces the model_NN backend exactly.
    Builds both networks from the same checkpoints and compares forward passes
    on random normalized inputs. Runs in collAIder's environment (imports
    model_NN, hence the physics front-end)."""
    import model_NN
    torch.manual_seed(seed)
    X = torch.randn(n, 5)

    det_cls = model_NN.ClassificationNeuralNetwork()
    det_cls.load_state_dict(torch.load(PerformCollision.CLASS_CHECKPOINT, map_location="cpu", weights_only=False)["model_state_dict"])
    det_cls.eval()
    bay_cls = ClassificationBayesNet()
    load_frozen_means(bay_cls, torch.load(PerformCollision.CLASS_CHECKPOINT, map_location="cpu", weights_only=False)["model_state_dict"])
    bay_cls.eval()
    set_means_only(bay_cls, True)

    det_reg = model_NN.RegressionNeuralNetwork()
    det_reg.load_state_dict(torch.load(PerformCollision.REG_CHECKPOINT, map_location="cpu", weights_only=False)["model_state_dict"])
    det_reg.eval()
    bay_reg = RegressionBayesNet()
    load_frozen_means(bay_reg, torch.load(PerformCollision.REG_CHECKPOINT, map_location="cpu", weights_only=False)["model_state_dict"])
    bay_reg.eval()
    set_means_only(bay_reg, True)

    with torch.no_grad():
        cls_delta = (det_cls(X) - bay_cls(X)).abs().max().item()
        reg_delta = (det_reg(X) - bay_reg(X)).abs().max().item()
    print(f"classifier max|delta logit|    = {cls_delta:.3e}")
    print(f"regressor  max|delta fraction| = {reg_delta:.3e}")
    print("deterministic mode reproduces model_NN" if max(cls_delta, reg_delta) == 0.0
          else "WARNING: deterministic mode does not match model_NN")


if __name__ == "__main__":
    verify_against_nn()
