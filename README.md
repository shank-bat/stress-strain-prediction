# **Stress–Strain Prediction Project**

## **Overview**

This repository explores machine-learning approaches for predicting the mechanical behavior of ductile materials.  
The project has two major components:

1. **Keypoints** – predicts key mechanical properties such as Yield Strength, Tensile Strength, and Elongation.  
   This module is fully operational and forms the main focus of the current work.

2. **Curves** – models the full stress–strain curve using PCA compression and MLPs.  
   This component is experimental and not yet stable.

---

## **Keypoints Module**

**Location:** `Keypoints/`

### Purpose
Predicts material strength properties using alloy composition and processing parameters.

### Data
- **Steel:** Composition-only dataset (C, Mn, Si, Cr, Ni, Mo, V, N, Nb, Co, W, Al, Ti).  
- **Aluminium:** Composition and categorical processing (for example, “Solutionised + Artificially peak aged”).

Note: Data cleaning is not yet complete. Some fields contain inconsistencies and missing values.  
A dedicated round of dataset corrections and normalization will be carried out in the next phase.

### Models

| Model | Framework | Description |
|--------|------------|-------------|
| **Random Forest** | scikit-learn | Fast ensemble baseline with solid generalization. |
| **XGBoost** | xgboost | Gradient boosting model with strong R² performance. |
| **Neural Network (MLP)** | TensorFlow / Keras | Two-layer dense network for nonlinear feature interactions. |

All models perform **multi-output regression**, predicting:
- Yield Strength (MPa)  
- Tensile Strength (MPa)  
- Elongation (%)

```### Directory Structure

Keypoints/
├── Models/
│ ├── NeuralNet.py # Neural network implementation
│ ├── XGBoost.py # XGBoost regression
│ └── RandomForest.py # Random forest regression
├── Databases/
│ ├── steel_strength_cleaned.csv
│ └── al_data_cleaned.csv
└── backend.py # FastAPI backend for the web app
```


### Running the Models

To start the backend API:
uvicorn Keypoints.backend:app --reload

To run an individual model directly:
python3 Keypoints/Models/NeuralNet.py

---

## **Curves Module**

**Location:** `Curves/` or earlier `torcher_*.py` files

This component aimed to predict full stress–strain curves using PCA compression and direct MLP regression.  
While the method showed potential for capturing overall curve shape, the workflow is currently not reproducible with the cleaned datasets and remains under revision.

The Curves module will be revisited once the datasets are standardized and temperature-dependent normalization is complete.

---

## **Next Steps**

- Finalize and validate dataset cleaning and normalization.  
- Restore and extend curve-based prediction once data quality is sufficient.  
- Introduce configuration files for model hyperparameters and training options.  
- Integrate the Keypoints backend and frontend for public demonstration.

---

## **Current Status**

| Component | Status | Notes |
|------------|---------|-------|
| Keypoints | Stable | Operational with all three models |
| Curves | Experimental | Requires data consistency work |
| Data Cleaning(Within Keypoints) | In Progress | Planned refinement and revalidation |
