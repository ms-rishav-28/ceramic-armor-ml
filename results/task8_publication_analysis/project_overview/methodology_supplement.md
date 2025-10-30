# Methodology Supplement: Ceramic Armor ML Pipeline

## Detailed Implementation Specifications

### Model Architecture Details
- **XGBoost:** Intel MKL acceleration, n_estimators=500, max_depth=8, learning_rate=0.1
- **CatBoost:** Built-in uncertainty, iterations=1000, depth=6, learning_rate=0.1
- **Random Forest:** n_estimators=500, max_depth=None, bootstrap=True
- **Gradient Boosting:** n_estimators=300, max_depth=5, learning_rate=0.1

### Feature Engineering Specifications
- **Specific Hardness:** H / ρ (GPa·cm³/g)
- **Brittleness Index:** H / K_IC (GPa·m^(-1/2))
- **Ballistic Efficiency:** σ_c × √H (GPa^1.5)
- **Thermal Shock Resistance:** Complex multi-parameter calculation

### Validation Protocols
- **Cross-Validation:** 5-fold stratified with ceramic system stratification
- **Leave-One-Out:** Leave-one-ceramic-family-out validation
- **Performance Thresholds:** R² ≥ 0.85 (mechanical), R² ≥ 0.80 (ballistic)
- **Statistical Testing:** Paired t-tests for model comparison

### Uncertainty Quantification Methods
- **Random Forest:** Inter-tree variance estimation
- **CatBoost:** Built-in uncertainty quantification
- **Ensemble:** Uncertainty propagation through stacking
- **Confidence Intervals:** 95% prediction intervals for all outputs
