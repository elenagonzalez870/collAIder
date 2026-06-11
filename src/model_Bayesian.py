import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchbnn as bnn
from numpy.core.multiarray import _reconstruct
torch.serialization.add_safe_globals([_reconstruct])

from encounter_physics import EncounterRegimeClassifier

# Offset used inside log10 for the velocity feature, matching the training data
LOG_OFFSET = 0.01


class MultiTaskNet(nn.Module):
    """
    Multi-task network with a Bayesian first layer (torchbnn.BayesLinear).

    The first shared layer holds a distribution over each weight instead of a
    point value, so every forward pass samples a slightly different network.
    Running the model many times and aggregating gives an uncertainty estimate
    for each prediction.

    Architecture: shared trunk -> classification head (4 logits) + three
    regression experts (one per surviving-star outcome), each predicting
    (mtot, q) where mtot = (M1_f + M2_f)/(M1_i + M2_i) and
    q = (M2_f/M1_f)/(M2_i/M1_i).
    """
    def __init__(self, input_dim, shared_hidden_sizes, class_hidden_sizes, reg_hidden_sizes, activation='relu'):
        super(MultiTaskNet, self).__init__()
        activation_fn = nn.ReLU if activation == 'relu' else nn.GELU

        # Shared layers - only the first one is Bayesian
        shared_layers = []
        prev_dim = input_dim
        for i, h in enumerate(shared_hidden_sizes):
            if i == 0:
                shared_layers.append(bnn.BayesLinear(in_features=prev_dim, out_features=h, prior_mu=0.0, prior_sigma=0.1))
            else:
                shared_layers.append(nn.Linear(prev_dim, h))
            shared_layers.append(activation_fn())
            prev_dim = h
        self.shared = nn.Sequential(*shared_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, class_hidden_sizes[0]),
            activation_fn(),
            nn.Linear(class_hidden_sizes[0], 4))

        # Regression experts: 1-star/merged (class 1), 2-star (class 2),
        # stripped survivor (class 3). Class 0 leaves no stars to regress.
        self.reg_expert1 = nn.Sequential(
            nn.Linear(prev_dim, reg_hidden_sizes[0]),
            activation_fn(),
            nn.Linear(reg_hidden_sizes[0], 2))
        self.reg_expert2 = nn.Sequential(
            nn.Linear(prev_dim, reg_hidden_sizes[0]),
            activation_fn(),
            nn.Linear(reg_hidden_sizes[0], 2))
        self.reg_expert3 = nn.Sequential(
            nn.Linear(prev_dim, reg_hidden_sizes[0]),
            activation_fn(),
            nn.Linear(reg_hidden_sizes[0], 2))

    def forward(self, x):
        shared_out = self.shared(x)
        class_logits = self.classifier(shared_out)

        probs = F.softmax(class_logits, dim=1)
        gating = probs[:, 1:4]

        reg1 = self.reg_expert1(shared_out)
        reg2 = self.reg_expert2(shared_out)
        reg3 = self.reg_expert3(shared_out)

        return class_logits, gating, reg1, reg2, reg3


class PerformCollision(nn.Module):

    _models_loaded = False
    _model = None
    _input_mean = None
    _input_std = None

    def __init__(self, age, pericenter, velocity_inf, mass1, mass2, radius1, radius2):
        super(PerformCollision, self).__init__()

        if not PerformCollision._models_loaded:
            checkpoint = torch.load(
                "../models/jalombar_bayesian_take2.pt",
                map_location=torch.device("cpu"),
                weights_only=False)

            # Rebuild the architecture from the hyperparameters stored in the
            # checkpoint: [*shared_sizes, class_size, reg_size, activation, lr]
            best_params = checkpoint['best_params']
            n_params = len(best_params)
            PerformCollision._model = MultiTaskNet(
                input_dim=6,
                shared_hidden_sizes=best_params[:n_params - 4],
                class_hidden_sizes=[best_params[n_params - 4]],
                reg_hidden_sizes=[best_params[n_params - 3]],
                activation=best_params[n_params - 2])

            # Load normalization statistics (first 5 features only; the
            # equal-mass flag is passed through unscaled)
            PerformCollision._input_mean = checkpoint["X_mean"]
            PerformCollision._input_std = checkpoint["X_std"]

            # Load model weights
            PerformCollision._model.load_state_dict(checkpoint["model"])

            # Set model to evaluation mode (the Bayesian layer still samples
            # new weights on every forward pass)
            PerformCollision._model.eval()

            PerformCollision._models_loaded = True

        self.model = PerformCollision._model
        self.input_mean = PerformCollision._input_mean
        self.input_std = PerformCollision._input_std

        # Transform and normalize the input data
        X = self.Transform(age, pericenter, velocity_inf, mass1, mass2, radius1, radius2)
        self.X_norm = self.Standard_Scale(X, self.input_mean, self.input_std)

        # Save initial masses in Msun
        self.mass1 = mass1
        self.mass2 = mass2
        self.m_ini_tot = mass1 + mass2

    def Transform(self, age, pericenter, velocity_inf, mass1, mass2, radius1, radius2):
        # This model was trained on different features than the NN/MoE
        # backends: raw age and masses, the pericenter normalized by the sum
        # of stellar radii, log-velocity, plus an equal-mass flag.
        X = np.array([
            age,
            mass1,
            mass2,
            pericenter / (radius1 + radius2),
            np.log10(velocity_inf + LOG_OFFSET),
            float(mass1 == mass2)], dtype=np.float32)
        return X

    def Standard_Scale(self, X, mean, std):
        X_norm = X.copy()
        X_norm[:5] = (X[:5] - mean) / std
        X_norm = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0)
        return X_norm

    def PerformClassification_and_Regression(self, n_samples=100):
        """
        Run n_samples stochastic forward passes and aggregate.

        Each pass samples new first-layer weights, classifies the collision,
        and hard-gates the regression expert that matches the sampled class.
        The per-sample outputs are converted to masses and then aggregated,
        so the reported std reflects both classification and regression
        spread.
        """
        sample_probs = np.empty((n_samples, 4), dtype=np.float64)
        sample_class = np.empty(n_samples, dtype=np.int64)
        sample_mtot = np.empty(n_samples, dtype=np.float64)
        sample_q = np.empty(n_samples, dtype=np.float64)

        with torch.no_grad():
            for s in range(n_samples):
                class_logits, _, reg1, reg2, reg3 = self.model(self.X_norm)
                pred_class = int(class_logits.argmax(dim=1))

                # Hard-gate: use the expert matching the sampled class.
                # Class 0 leaves no stars; classes 1 and 3 leave a single
                # star, so the final mass ratio q is 0 by definition.
                if pred_class == 1:
                    mtot = float(reg1[0, 0])
                    q = 0.0
                elif pred_class == 2:
                    mtot = float(reg2[0, 0])
                    q = float(reg2[0, 1])
                elif pred_class == 3:
                    mtot = float(reg3[0, 0])
                    q = 0.0
                else:
                    mtot = 0.0
                    q = 0.0

                sample_probs[s] = F.softmax(class_logits, dim=1).squeeze(0).numpy()
                sample_class[s] = pred_class
                sample_mtot[s] = min(mtot, 1.0)  # total final mass cannot exceed total initial mass
                sample_q[s] = max(q, 0.0)        # mass ratio cannot be negative

        # Convert each sample from (mtot, q) to [M1_f, M2_f, M_unbound] in Msun.
        # mtot is a fraction of the total initial mass; q is normalized by the
        # initial mass ratio.
        m_tot_f = sample_mtot * self.m_ini_tot
        q_f = sample_q * (self.mass2 / self.mass1)
        m1_f = m_tot_f / (1.0 + q_f)
        m2_f = m_tot_f - m1_f
        m_unbound = self.m_ini_tot - m_tot_f
        sample_values = np.stack([m1_f, m2_f, m_unbound], axis=1)

        # Aggregate: majority vote for the class, mean/std for the rest
        predicted_class = int(np.bincount(sample_class, minlength=4).argmax())
        class_probs = [float(p) for p in sample_probs.mean(axis=0)]
        predicted_values = [float(v) for v in sample_values.mean(axis=0)]
        predicted_values_std = [float(v) for v in sample_values.std(axis=0)]

        return predicted_class, predicted_values, predicted_values_std, class_probs


def process_encounters(ages, masses1, masses2, pericenters, velocities_inf, n_samples=100):
    """
    Process multiple stellar encounters with uncertainty estimates.

    Parameters:
    -----------
    ages : list or array
        Stellar ages in Gyr
    masses1, masses2 : list or array
        Stellar masses in solar masses
    pericenters : list or array
        Pericenter distances in Rsun
    velocities_inf : list or array
        Velocities at infinity in km/s
    n_samples : int
        Number of stochastic forward passes used to estimate uncertainties

    Returns:
    --------
    results : list of dicts
        Each dict contains 'regime_flag', 'predicted_class',
        'predicted_values', 'predicted_values_std' and 'class_probs'.
        'predicted_values_std' is the standard deviation of each mass
        component across the sampled networks; it is zero for tidal captures
        and flybys, whose outcomes are set by the physics-based classifier.
    """

    # Convert to arrays
    ages = np.atleast_1d(ages)
    masses1 = np.atleast_1d(masses1)
    masses2 = np.atleast_1d(masses2)
    pericenters = np.atleast_1d(pericenters)
    velocities_inf = np.atleast_1d(velocities_inf)

    # Check all inputs have same length
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

        # Classify encounter
        regime = classifier.classify_encounter(
            age=age, mass1=mass1, mass2=mass2,
            pericenter=pericenter, velocity_inf=velocity_inf)

        if regime == 'collision':

            regime_flag = -1
            radius1 = float(classifier.StellarRadiusEstimator(age, mass1))
            radius2 = float(classifier.StellarRadiusEstimator(age, mass2))
            collision = PerformCollision(age, pericenter, velocity_inf, mass1, mass2, radius1, radius2)
            predicted_class, predicted_values, predicted_values_std, class_probs = \
                collision.PerformClassification_and_Regression(n_samples=n_samples)

        elif regime == 'tidal_capture':
            regime_flag = -2
            # Assume a merger with no mass loss
            predicted_class = 1
            predicted_values = [1.*(mass1 + mass2), 0., 0.]
            predicted_values_std = [0., 0., 0.]
            class_probs = [0., 1., 0., 0.]

        else: #flyby
            regime_flag = -3
            # Stars fly by each other with no mass loss
            predicted_class = 2
            predicted_values = [mass1, mass2, 0]
            predicted_values_std = [0., 0., 0.]
            class_probs = [0., 0., 1., 0.]

        if flag == True and int(predicted_class) != 1:
            predicted_values[0], predicted_values[1] = predicted_values[1], predicted_values[0]
            predicted_values_std[0], predicted_values_std[1] = predicted_values_std[1], predicted_values_std[0]

        results.append({
        'regime_flag': regime_flag,
        'predicted_class': int(predicted_class),
        'predicted_values': [float(v) for v in predicted_values],
        'predicted_values_std': [float(v) for v in predicted_values_std],
        'class_probs': [float(p) for p in class_probs]})

    return results


if __name__ == "__main__":
    # Run example
    # Prompt user for inputs
    age = float(input("Enter stellar age in Gyr: "))
    mass1 = float(input("Enter mass of star 1 (Msun): "))
    mass2 = float(input("Enter mass of star 2 (Msun): "))
    pericenter = float(input("Enter pericenter distance (Rsun): "))
    velocity_inf = float(input("Enter velocity at infinity (km/s): "))

    # Process single encounter
    results = process_encounters([age], [mass1], [mass2], [pericenter], [velocity_inf])
    result = results[0]

    print(f"Regime flag: {result['regime_flag']} (-1=collision, -2=tidal_capture, -3=flyby)")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Predicted values: {result['predicted_values']}")
    print(f"Predicted values std: {result['predicted_values_std']}")
