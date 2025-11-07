import torch
from torch import nn
import numpy as np
import h5py
import torch.nn.functional as F
from numpy.core.multiarray import _reconstruct
torch.serialization.add_safe_globals([_reconstruct])

class EncounterRegimeClassifier:
    """
    Classifies stellar encounters into collision, tidal capture, or flyby regimes.
    """
    def __init__(self):
        self.G = 1.90682 * 10**(5)  # km^2 Rsun MSun^-1 s^-2
        
    def StellarRadiusEstimator(self, target_age, target_mass):
    
        """
        Function to interpolate stellar radii from Posydin v2 MESA tracks 
        
        Input: np.array, in the format [[age1, mass1], [age2, mass2], ...] where age is in Gyr and mass in solar masses 
        Output: list, in the format [radius1, radius 2] in units of solar radii
        """
        
        filename = "/home/egp8636/b1095/POSYDON_data_v2_grids_0.01Zsun.tar.gz/POSYDON_data/single_HMS/1e-02_Zsun.h5"
        
        with h5py.File(filename, "r") as f:
            grid = f['grid']
            masses = grid['initial_values']['S1_star_mass'][()]
                
            #Convert ages from Gyr to yr 
            target_age *= 10**9
            
            # Find indices of the two nearest mass tracks
            if target_mass <= masses.min():
                matches = np.argmin(masses), np.argmin(masses)
            elif target_mass >= masses.max():
                matches = np.argmax(masses), np.argmax(masses)
            else:
                matches = np.argsort(np.abs(masses - target_mass))[:2]

            interp_radii = []
            interp_masses = []
            central_h1s = []
            for i in matches:
                run = grid[f"run{int(i)}"]

                ages = run['history1']['star_age'][:]
                logR = run['history1']['log_R'][:]
                
                # First: check if the star is past the TAMS 
                age_match = np.argsort(np.abs(ages - target_age))[0]
                central_h1s.append(run['history1']['center_h1'][age_match]) # Central hydrogen fraction at given time, if below 10^(-5) then star is past the TAMS

                radii = 10**logR  # convert log(R/Rsun) → Rsun

                # Interpolate in time within the track
                r_interp = np.interp(target_age, ages, radii)
                interp_radii.append(r_interp)
                interp_masses.append(masses[i])
            
            # Now interpolate in masses:
            # Sort by increasing mass to avoid np.interp confusion
            if interp_masses[1] < interp_masses[0]:
                interp_masses[0], interp_masses[1] = interp_masses[1], interp_masses[0]
                interp_radii[0] , interp_radii[1]  = interp_radii[1] , interp_radii[0]

            radii_predictions = np.interp(target_mass, interp_masses, interp_radii)

            if central_h1s[0] < 10**(-5)  and central_h1s[1] < 10**(-5):
                raise ValueError(f" Star of mass {target_mass} and age {target_age/10**(9)} Gyr is past the TAMS with a central_h1 fraction of (~{np.mean(central_h1s)}), we can't compute the collision.")

        return radii_predictions

    def classify_encounter(self, age, mass1, mass2, pericenter, velocity_inf):
        """
        Classify encounter type based on physical parameters.
        
        Parameters:
        -----------
        age: float
            Stellar age in Gyr 
        mass1, mass2 : float or array
            Stellar masses in solar masses
        pericenter : float or array
            Pericenter distance in Rsun
        velocity_inf : float or array
            Velocity at infinity in km/s

        Returns:
        --------
        regime : array of strings
            'collision', 'tidal_capture', or 'flyby' for each encounter
        """

        r_peri = np.asarray(pericenter)
        v_inf = np.asarray(velocity_inf)
        
        radius1 = self.StellarRadiusEstimator(age, mass1)
        radius2 = self.StellarRadiusEstimator(age, mass2)

        R1 = np.asarray(radius1)
        R2 = np.asarray(radius2)
        
        # Physical collision: pericenter < sum of radii
        collision_criterion = r_peri < (R1 + R2)
        
        # Tidal capture criterion
        E_tidal = self.tidal_energy_loss(mass1, mass2, R1, R2, r_peri, v_inf) # Msun * km^2/ s^2
        E_orb = 0.5 * (mass1 * mass2 / (mass1 + mass2)) * velocity_inf**2 # Msun (km/s)^2
        E_final = E_orb - E_tidal

        tidal_capture_criterion = E_final <= 0
        
        # Classification
        regime = np.where(collision_criterion, 'collision',
                         np.where(tidal_capture_criterion, 'tidal_capture', 'flyby'))
        
        return regime
    
    def tidal_energy_loss(self, M1, M2, R1, R2, rp, vinf):

        G = 1.90682 * 10**(5)  # km^2 Rsun MSun^-1 s^-2

        # Reduced mass
        mu = M1 * M2 / (M1 + M2)  # Msun

        # Gravitational potential energy constant
        k = G * M1 * M2  # (km/s)^2 Rsun Msun

        # Compute periapsis velocity v_p using energy conservation:
        # 0.5 * mu * vinf^2 = -k / r_p + 0.5 * mu * v_p^2
        # => v_p^2 = vinf^2 + 2 * k / (mu * r_p)
        v_p_sq = vinf**2. + 2. * k / (mu * rp)
        v_p = np.sqrt(v_p_sq) # km/s

        # Orbital energy: E = 0.5 * mu * vinf^2
        E = 0.5 * mu * vinf**2 # Msun (km/s)^2

        # Angular momentum: l = mu * r_p * v_p
        l = mu * rp * v_p # Msun Rsun km/s

        # Calculate the eccentricity 
        e = np.sqrt(1 + (2. * E * l**2.) / (mu * k**2.))

        # Calculate eta parameters using Eq 2 in Portegies Zwart et al. 1993
        eta1 = (M1 / (M1 + M2))**(1./2.) * (rp/R1)**(3./2.)
        eta2 = (M2 / (M1 + M2))**(1./2.) * (rp/R2)**(3./2.)

        # Calculate alpha from eq A5 in Mardling et al. 2001
        alpha1 = 1. + 0.5 * np.abs((eta1 - 2.) / 2.)**(1.5)
        alpha2 = 1. + 0.5 * np.abs((eta2 - 2.) / 2.)**(1.5)

        # Calculate zeta from eq A4 in Mardling et al. 2001
        zeta1 = eta1 * (2./(1 + e))**(alpha1 / 2.)
        zeta2 = eta2 * (2./(1 + e))**(alpha2 / 2.)

        # Determine which polytrope index to use
        # For M < 0.8 Msun stars, we'll use n = 1.5
        # For M > 0.8 Msun stars, we'll use n = 3

        n1 = 1.5 if M1 < 0.8 else 3.0
        n2 = 1.5 if M2 < 0.8 else 3.0
    
        # Calculate the energy dissipated from tides by each star using Eq 4 in Portegies Zwart et al. 1993
        E_1 =(G * M2**2. / R1) * ((R1/rp)**6. * self.T2(zeta1, n1) + (R1/rp)**8. * self.T3(zeta1, n1))
        E_2 =(G * M1**2. / R2) * ((R2/rp)**6. * self.T2(zeta2, n2) + (R2/rp)**8. * self.T3(zeta2, n2))

        # In the case of a star with 0.4 Msun < M < 0.8 Msun, we will also compute n=3 tidal dissipation, since 
        # for those masses the star has both a radiative core and a convective envelope. 
        if M1 >= 0.4 and M1 <= 0.8: 
            # E1 here is the n = 1.5 approx
            n1 = 3 
            E_1_n3 =(G * M2**2. / R1) * ((R1/rp)**6. * self.T2(zeta1, n1) + (R1/rp)**8. * self.T3(zeta1, n1))
            E_1_interp = (E_1_n3 * (M1 - 0.4) + E_1 * (0.8 - M1)) / 0.4
            E_1 = E_1_interp
            
        if M2 >= 0.4 and M2 <= 0.8:
            # E2 here is the n = 1.5 approx 
            n2 = 3 
            E_2_n3 =(G * M1**2. / R2) * ((R2/rp)**6. * self.T2(zeta2, n2) + (R2/rp)**8. * self.T3(zeta2, n2))
            E_2_interp = (E_2_n3 * (M2 - 0.4) + E_2 * (0.8 - M2)) / 0.4
            E_2 = E_2_interp

        E_tidal = E_1 + E_2 # Msun * km^2/ s^2

        return E_tidal

    def T2(self, zeta, n):
        x = np.log10(zeta)
        if n == 3.: 
            # n=3 polytrope coefficients for T2 and T3 (PZM fits)
            A2 =  -1.124; B2 = 0.877; C2 = -13.37; D2 = 21.55; E2 = -16.48; F2 = 4.124
        
        if n == 1.5:
            # n=1.5 polytrope coefficients for T2 and T3 (PZM fits)
            A2 =  -0.397; B2 = 1.678; C2 = 1.277; D2 = -12.42; E2 = 9.446; F2 = -5.550

        T2 = max(10**(A2 + B2*x + C2*x**2 + D2*x**3 + E2*x**4 + F2*x**5), 1e-5)

        return T2

    def T3(self, zeta, n):
        x = np.log10(zeta)
        if n == 3.: 
            # n=3 polytrope coefficients for T2 and T3 (PZM fits)
            A3 =  -1.703; B3 = 2.653; C3 = -14.34; D3 = 12.85; E3 = -0.492; F3 = -3.600
        
        if n == 1.5:
            # n=1.5 polytrope coefficients for T2 and T3 (PZM fits)
            A3 =  -0.909; B3 = 1.574; C3 = 12.37; D3 = -57.40; E3 = 80.10; F3 = -46.43

        T3 = max(10**(A3 + B3*x + C3*x**2 + D3*x**3 + E3*x**4 + F3*x**5), 1e-5)

        return T3

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
                "examples/sample_classmodel.pt",  # replace with your file path
                map_location=torch.device("cpu"), # load on CPU for demonstration
                weights_only=False)  

            regression_checkpoint = torch.load(
                "examples/sample_regmodel.pt",
                map_location=torch.device("cpu"), 
                weights_only=False)

            # Load normalization statistics
            PerformCollision._input_mean  = classification_checkpoint["train_mean"]
            PerformCollision._input_std  = classification_checkpoint["train_std"]

            # Load model weights**
            PerformCollision._classification_model.load_state_dict(classification_checkpoint["model_state_dict"])
            PerformCollision._regression_model.load_state_dict(regression_checkpoint["model_state_dict"])

            # Set models to evaluation mode**
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
    
        # Load model weights during initialization
        self.classification_model.load_state_dict(classification_checkpoint["model_state_dict"])
        self.regression_model.load_state_dict(regression_checkpoint["model_state_dict"])

        # Save total initial mass in Msun 
        self.m_ini_tot = mass1 + mass2

    def Transform(self, age, pericenter, velocity_inf, mass1, mass2):
        X = np.array([
            np.log10(age + 0.001),
            np.log10(pericenter + 1.),
            np.log10(velocity_inf),
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
        return predicted_values

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
        Each dict contains 'regime_flag', 'predicted_class', 'predicted_values'
    """

    # Convert to arrays
    ages = np.atleast_1d(ages)
    masses1 = np.atleast_1d(masses1)
    masses2 = np.atleast_1d(masses2)
    pericenters = np.atleast_1d(pericenters)
    velocities_inf = np.atleast_1d(velocities_inf)
    
    # Check all inputs have same length**
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
        
        # Classify encounter
        # Classify encounter
        regime = classifier.classify_encounter(
            age=age, mass1=mass1, mass2=mass2, 
            pericenter=pericenter, velocity_inf=velocity_inf)

        if regime == 'collision':
            regime_flag = -1
            collision = PerformCollision(age, pericenter, velocity_inf, mass1, mass2) 
            predicted_class = collision.PerformClassification()
            predicted_values = collision.PerformRegression() 


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

        results.append({
        'regime_flag': regime_flag,
        'predicted_class': predicted_class,
        'predicted_values': predicted_values})
    
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