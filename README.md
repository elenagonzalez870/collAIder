# CollAIder

A machine learning-based tool for predicting outcomes of stellar encounters using neural networks trained on SPH (Smoothed Particle Hydrodynamics) simulations.

## Installation

### Requirements

```bash
pip install torch numpy h5py
```

## Overview

This model classifies stellar encounters into three physical regimes and predicts collision outcomes:

### Encounter Regimes

- **Collision (flag: -1)**: Physical contact between stars when pericenter < R₁ + R₂
- **Tidal Capture (flag: -2)**: Stars become gravitationally bound through tidal energy dissipation
- **Flyby (flag: -3)**: Stars pass without significant interaction

## Output Format

Each result is a dictionary containing:

- **`regime_flag`**: Integer indicating the encounter type
  - `-1`: Collision
  - `-2`: Tidal capture
  - `-3`: Flyby
- **`predicted_class`**: Classification outcome (0-3 for collisions, 1 for tidal capture, 2 for flyby)
- **`predicted_values`**: List of [star_mass1, star_mass2, unbound_mass] in M☉


## Tutorial

- See `./examples/Tutorial.ipynb` for detailed examples and workflow demonstrations.
- See `./examples/NN_tutorial.ipynb` for detailed example on how to use the NNs independently.
- See `./examples/MoE_tutorial.ipynb` for detailed examples on how to use the MoE independently.

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

## Error Handling

The code will raise errors for:
- **Post-TAMS stars**: `ValueError` when central H fraction < 10⁻⁵
- **Mismatched array lengths**: All input arrays must have same length
- **Missing model files**: Check that `.pt` files are in correct location

## References

- Portegies Zwart, S. F., & McMillan, S. L. W. (1993). *The evolution of close triple stars.* ApJ, 410, 759
- Mardling, R. A., & Aarseth, S. J. (2001). *Tidal interactions in star cluster simulations.* MNRAS, 321, 398  
- Fragos, T., et al. (2023). *POSYDON: A Population Synthesis Code*
- MESA stellar evolution code (Paxton et al. 2011, 2013, 2015, 2018, 2019)

## Citation

If you use this code in your research, please cite:
```
González Prieto, E., et al., 2026, arXiv:2602.10191
```

## License

[Add your license here]

## Contact

elena.prieto[at]northwestern.edu

## Acknowledgments

This work uses:
- POSYDON v2 stellar evolution grids
- PyTorch for neural network implementation
- SPH simulation data for model training
