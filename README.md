# Stress–Strain Curve Prediction in Ductile Materials

## Project Overview
Goal: Explore machine learning approaches to predict stress–strain behavior of various ductile 
Data: Engineering stress–strain curves for Al 6061-T651 from 9 lots, multiple temperatures.  

## Versions / Milestones
- **v1 Baseline (torcher_simple.py)**  
  - Direct MLP predicting stress values at 250 resampled strain points.  
  - Result: Model underfit, predictions flat/averaged (see Figure 1).  

- **v2 Improved (torcher_heavy.py)**  
  - PCA compression of curves + larger MLP to predict PCA coefficients.  
  - Added normalization, dropout, scheduler, early stopping.  
  - Result: Predictions follow actual curve shape, outperform baselines.  

## Results
| Model | Validation MSE | Notes |
|-------|----------------|-------|
| Baseline | … | Underfit |
| PCA+MLP | … | Captures curve shape |

See `docs/figures/` for plots.

## Next Steps
- Try per-temperature models.  
- Increase PCA components (20–25).  
- Ensemble multiple seeds.  
- Add richer input features if available (composition, microstructure).
Project Overview