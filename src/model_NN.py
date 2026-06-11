import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from numpy.core.multiarray import _reconstruct
torch.serialization.add_safe_globals([_reconstruct])

from encounter_physics import EncounterRegimeClassifier

class PerformCollision(nn.Module):

    _models_loaded = False 
    _classification_model = None
    _regression_model = None
    _input_mean = None
    _input_std = None

    def __init__(self, age, pericenter, velocity_inf, mass1, mass2):
        super(PerformCollision, self).__init__() 

        if not PerformCollision._models_loaded:
            PerformCollision._classification_model = ClassificationNeuralNetwork()
            PerformCollision._regression_model = RegressionNeuralNetwork()

            # Load the saved checkpoint for the classification model
            classification_checkpoint = torch.load(
                "../models/best_NN_class_model.pt",
                map_location=torch.device("cpu"),
                weights_only=False)

            regression_checkpoint = torch.load(
                "../models/best_NN_reg_model.pt",
                map_location=torch.device("cpu"), 
                weights_only=False)

            # Load normalization statistics
            PerformCollision._input_mean  = classification_checkpoint["train_mean"]
            PerformCollision._input_std  = classification_checkpoint["train_std"]

            # Load model weights
            PerformCollision._classification_model.load_state_dict(classification_checkpoint["model_state_dict"])
            PerformCollision._regression_model.load_state_dict(regression_checkpoint["model_state_dict"])

            # Set models to evaluation mode
            PerformCollision._classification_model.eval()
            PerformCollision._regression_model.eval()
            
            PerformCollision._models_loaded = True

        self.classification_model = PerformCollision._classification_model
        self.regression_model = PerformCollision._regression_model
        self.input_mean = PerformCollision._input_mean
        self.input_std = PerformCollision._input_std

        # Transform and normalize the input data
        X = self.Transform(age, pericenter, velocity_inf, mass1, mass2)
        self.X_norm = self.Standard_Scale(X, self.input_mean, self.input_std)

        # Save total initial mass in Msun 
        self.m_ini_tot = mass1 + mass2

    def Transform(self, age, pericenter, velocity_inf, mass1, mass2):
        X = np.array([
            np.log10(age + 0.001),
            np.log10(pericenter + 0.1),
            np.log10(velocity_inf + 10.),
            np.log(mass1),
            np.log(mass2)], dtype=np.float32)
        return X

    def Standard_Scale(self, X, mean, std):
        X_norm = (X - mean) / std
        X_norm = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0)
        return X_norm

    def PerformClassification(self):
        with torch.no_grad():
            classification_pred = self.classification_model(self.X_norm)
            predicted_class = torch.argmax(classification_pred, dim=1).item()
        return predicted_class

    def PerformRegression(self):
        with torch.no_grad():
            regression_pred  = self.regression_model(self.X_norm)
            predicted_values = regression_pred.squeeze(0).tolist()  # Converts tensor to a list
            predicted_values = [val * self.m_ini_tot for val in predicted_values] 
            predicted_values = [float(val) for val in predicted_values]
        return predicted_values

def process_collisions(pred_class, pred_reg): 
    """
    Process NN predictions for consistency between classification and regression
    """

    if pred_class == 0:  #if both stars are destroyed 
        pred_reg[2] += pred_reg[0] + pred_reg[1] #re-assign masses for mass conservation 
        pred_reg[0], pred_reg[1] = 0.0, 0.0
    if pred_class == 1 or pred_class == 3:  #if only one star survives (merger or stripped)
        pred_reg[2] += pred_reg[1] #re-assign masses for mass conservation
        pred_reg[1] = 0.0
    return pred_class, pred_reg

def process_encounters(ages, masses1, masses2, pericenters, velocities_inf):
    """
    Process multiple stellar encounters.

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
        
    Returns:
    --------
    results : list of dicts
        Each dict contains 'regime_flag', 'predicted_class' and 'predicted_values'
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
            collision = PerformCollision(age, pericenter, velocity_inf, mass1, mass2) 
            predicted_class = collision.PerformClassification()
            predicted_values = collision.PerformRegression() 

            # Pre-process the outputs before returning to user to enforce consistency between classification and regression predictions
            predicted_class, predicted_values = process_collisions(predicted_class, predicted_values)

        elif regime == 'tidal_capture':
            regime_flag = -2
            # Assume a merger with no mass loss 
            predicted_class = 1 
            predicted_values = [1.*(mass1 + mass2), 0., 0.]

        else: #flyby
            regime_flag = -3
            # Stars fly by each other with no mass loss
            predicted_class = 2
            predicted_values = [mass1, mass2, 0]

        if flag == True and int(predicted_class) != 1:
            predicted_values[0], predicted_values[1] = predicted_values[1], predicted_values[0]

        results.append({
        'regime_flag': regime_flag,
        'predicted_class': int(predicted_class),
        'predicted_values': [float(v) for v in predicted_values]})
    
    return results



class RegressionNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 512),
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
        fractions = F.softmax(logits, dim=-1)  # Convert to probabilities
        return fractions

class ClassificationNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


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