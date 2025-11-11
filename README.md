# Stellar Encounter Collision Model

A machine learning-based tool for predicting outcomes of stellar encounters using neural networks trained on SPH (Smoothed Particle Hydrodynamics) simulations.

## Overview

This model classifies stellar encounters into three physical regimes and predicts collision outcomes:

### Encounter Regimes

- **Collision (flag: -1)**: Physical contact between stars when pericenter < R₁ + R₂
- **Tidal Capture (flag: -2)**: Stars become gravitationally bound through tidal energy dissipation
- **Flyby (flag: -3)**: Stars pass without significant interaction

## Installation

### Requirements

```bash
pip install torch numpy h5py
```

### Required Files

- `classmodel.pt` - Classification model weights
- `regmodel.pt` - Regression model weights  
- `1e-02_Zsun.h5` - Posydon's Single Star MESA Models

## Quick Start

```python
from model import process_encounters

# Single encounter
results = process_encounters(
    ages=[1.0],           # Gyr
    masses1=[1.0],        # Msun
    masses2=[0.8],        # Msun
    pericenters=[5.0],    # Rsun
    velocities_inf=[50.0] # km/s
)

print(results[0])
# {'regime_flag': -1, 'predicted_class': 0, 'predicted_values': [1.2, 0.5, 0.1]}
```

### Batch Processing

```python
# Process multiple encounters
ages = [0.5, 1.0, 2.0, 5.0]
masses1 = [1.0, 1.2, 0.8, 1.5]
masses2 = [0.8, 0.9, 0.6, 1.0]
pericenters = [3.0, 5.0, 10.0, 2.0]
velocities = [100.0, 50.0, 30.0, 150.0]

results = process_encounters(ages, masses1, masses2, pericenters, velocities)
```

## Output Format

Each result is a dictionary containing:

- **`regime_flag`**: Integer indicating the encounter type
  - `-1`: Collision
  - `-2`: Tidal capture
  - `-3`: Flyby
- **`predicted_class`**: Classification outcome (0-3 for collisions, 1 for tidal capture, 2 for flyby)
- **`predicted_values`**: List of [mass_component1, mass_component2, mass_component3] in M☉

## Key Assumptions & Caveats

### ⚠️ Stellar Evolution
- **Main Sequence Only**: Stars must be on the main sequence (MS). Post-TAMS (Terminal Age Main Sequence) stars will raise a `ValueError`
- **Metallicity**: Model uses tracks at Z = 0.01 Z☉ (solar metallicity)
- Stellar radii are interpolated from POSYDON grids based on age and mass

### ⚠️ Mass Range
- Validated for masses between ~0.1 - 100 M☉
- Extrapolation outside this range may be unreliable

### ⚠️ Tidal Dissipation Physics
Uses polytrope approximations from Portegies Zwart & McMillan (1993) and Mardling & Aarseth (2001):
- **n = 1.5** for M < 0.8 M☉ (convective envelopes)
- **n = 3.0** for M > 0.8 M☉ (radiative envelopes)
- **Linear interpolation** for 0.4 < M < 0.8 M☉ (mixed structure)

### ⚠️ Simplified Regime Assumptions
- **Tidal Capture**: Assumes perfect merger with no mass loss
- **Flyby**: Assumes no mass transfer or interaction
- **All regimes**: No stellar rotation or stellar winds considered


## Example Use Cases

### 1. Single Encounter Analysis
```python
results = process_encounters([1.0], [1.0], [0.8], [5.0], [50.0])
```

### 2. Parameter Space Exploration
```python
import numpy as np

pericenters = np.linspace(1.0, 20.0, 100)
n = len(pericenters)

results = process_encounters(
    ages=[1.0] * n,
    masses1=[1.0] * n,
    masses2=[0.8] * n,
    pericenters=pericenters,
    velocities_inf=[50.0] * n
)
```

### 3. Classification Only
```python
from model import EncounterRegimeClassifier

classifier = EncounterRegimeClassifier()
regime = classifier.classify_encounter(
    age=1.0, mass1=1.0, mass2=0.8,
    pericenter=5.0, velocity_inf=50.0
)
```

## Error Handling

The code will raise errors for:
- **Post-TAMS stars**: `ValueError` when central H fraction < 10⁻⁵
- **Mismatched array lengths**: All input arrays must have same length
- **Missing model files**: Check that `.pt` files are in correct location

## Tutorial

See `tutorial.ipynb` for detailed examples and workflow demonstrations.

## References

- Portegies Zwart, S. F., & McMillan, S. L. W. (1993). *The evolution of close triple stars.* ApJ, 410, 759
- Mardling, R. A., & Aarseth, S. J. (2001). *Tidal interactions in star cluster simulations.* MNRAS, 321, 398  
- Fragos, T., et al. (2023). *POSYDON: A Population Synthesis Code*
- MESA stellar evolution code (Paxton et al. 2011, 2013, 2015, 2018, 2019)

## Citation

If you use this code in your research, please cite:
```
[Your paper citation here]
```

## License

[Add your license here]

## Contact

[Your contact information]

## Acknowledgments

This work uses:
- POSYDON v2 stellar evolution grids
- PyTorch for neural network implementation
- SPH simulation data for model training