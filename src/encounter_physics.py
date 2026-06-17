import numpy as np
import h5py

class EncounterRegimeClassifier:
    """
    Classifies stellar encounters into collision, tidal capture, or flyby regimes.
    """
    def __init__(self):
        self.G = 1.90682 * 10**(5)  # km^2 Rsun MSun^-1 s^-2

    def StellarRadiusEstimator(self, target_age, target_mass):

        """
        Interpolate a stellar radius from the POSYDON v2 MESA tracks.

        Input: target_age in Gyr, target_mass in solar masses
        Output: radius in solar radii

        Raises ValueError if the star is past the TAMS (central hydrogen
        fraction below 1e-5) by more than a 10% relative age tolerance.
        """

        filename = "../data/POSYDON_data_v2_grids_0.01Zsun.tar.gz/POSYDON_data/single_HMS/1e-02_Zsun.h5"

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
            tams_ages = []
            for i in matches:
                run = grid[f"run{int(i)}"]

                ages = run['history1']['star_age'][:]
                logR = run['history1']['log_R'][:]
                h1 = run['history1']['center_h1'][:]

                # First: check if the star is past the TAMS
                age_match = np.argsort(np.abs(ages - target_age))[0]
                central_h1s.append(run['history1']['center_h1'][age_match]) # Central hydrogen fraction at given time, if below 10^(-5) then star is past the TAMS

                radii = 10**logR  # convert log(R/Rsun) → Rsun

                # Interpolate in time within the track
                r_interp = np.interp(target_age, ages, radii)
                interp_radii.append(r_interp)
                interp_masses.append(masses[i])

                # Find first index where central H drops below threshold
                tams_idx = np.where(h1 < 1e-5)[0]

                if len(tams_idx) > 0:
                    tams_age = ages[tams_idx[0]]   # first time star is past TAMS
                else:
                    tams_age = None

                tams_ages.append(tams_age)


            # Now interpolate in masses:
            # Sort by increasing mass to avoid np.interp confusion
            if interp_masses[1] < interp_masses[0]:
                interp_masses[0], interp_masses[1] = interp_masses[1], interp_masses[0]
                interp_radii[0] , interp_radii[1]  = interp_radii[1] , interp_radii[0]

            radii_predictions = np.interp(target_mass, interp_masses, interp_radii)


            if central_h1s[0] < 10**(-5):
                # Use the earliest TAMS age among the matched tracks
                tams_ages = [a for a in tams_ages if a is not None]
                earliest_tams = min(tams_ages)
                tolerance = np.abs(target_age - earliest_tams)/earliest_tams  # 10 % relative tolerance
                if tolerance > 0.10:
                    raise ValueError(f" Star of mass {target_mass} and age {target_age/10**(9)} Gyr is past the TAMS with a central_h1 fraction of (~{np.mean(central_h1s)}) and approx TAMS age {earliest_tams/10**(9)}, we can't compute the collision.")

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
        E_orb = 0.5 * (mass1 * mass2 / (mass1 + mass2)) * v_inf**2 # Msun (km/s)^2

        tidal_capture_criterion = E_final <= 0

        # Classification
        regime = np.where(collision_criterion, 'collision',
                         np.where(tidal_capture_criterion, 'tidal_capture', 'flyby'))

        return regime

    def tidal_energy_loss(self, M1, M2, R1, R2, rp, vinf):

        G = self.G  # km^2 Rsun MSun^-1 s^-2

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
            # E_1 here is the n = 1.5 approx
            n1 = 3
            E_1_n3 =(G * M2**2. / R1) * ((R1/rp)**6. * self.T2(zeta1, n1) + (R1/rp)**8. * self.T3(zeta1, n1))
            E_1_interp = (E_1_n3 * (M1 - 0.4) + E_1 * (0.8 - M1)) / 0.4
            E_1 = E_1_interp

        if M2 >= 0.4 and M2 <= 0.8:
            # E_2 here is the n = 1.5 approx
            n2 = 3
            E_2_n3 =(G * M1**2. / R2) * ((R2/rp)**6. * self.T2(zeta2, n2) + (R2/rp)**8. * self.T3(zeta2, n2))
            E_2_interp = (E_2_n3 * (M2 - 0.4) + E_2 * (0.8 - M2)) / 0.4
            E_2 = E_2_interp

        # The fits from Portegies Zwart et al. 1993 can only be used for eta <= 10
        if eta1 > 10.:
            E_1 = 0.

        if eta2 > 10.:
            E_2 = 0.

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

        # Clip to avoid numerical overflows
        poly = A2 + B2*x + C2*x**2 + D2*x**3 + E2*x**4 + F2*x**5
        poly = np.clip(poly, -100, 100)

        T2 = max(10**(poly), 1e-5)

        return T2

    def T3(self, zeta, n):
        x = np.log10(zeta)
        if n == 3.:
            # n=3 polytrope coefficients for T2 and T3 (PZM fits)
            A3 =  -1.703; B3 = 2.653; C3 = -14.34; D3 = 12.85; E3 = -0.492; F3 = -3.600

        if n == 1.5:
            # n=1.5 polytrope coefficients for T2 and T3 (PZM fits)
            A3 =  -0.909; B3 = 1.574; C3 = 12.37; D3 = -57.40; E3 = 80.10; F3 = -46.43

        # Clip to avoid numerical overflows
        poly = A3 + B3*x + C3*x**2 + D3*x**3 + E3*x**4 + F3*x**5
        poly = np.clip(poly, -100, 100)

        T3 = max(10**(poly), 1e-5)

        return T3
