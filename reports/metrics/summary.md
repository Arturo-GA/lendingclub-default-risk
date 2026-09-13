# Resultados (test out-of-time)

Variables de precio de LendingClub (int_rate, grade): **incluidas**.  
Tasa de default en test: **14.91%**. Umbral elegido en validación con costo FN:FP = 5:1.

| Modelo        |    AUC |   Gini |     KS |   PR-AUC |   Brier |   AUC train |   Umbral | Aprobación   | Default aprobados   |
|:--------------|-------:|-------:|-------:|---------:|--------:|------------:|---------:|:-------------|:--------------------|
| lightgbm      | 0.7151 | 0.4302 | 0.3137 |   0.2986 |  0.1991 |      0.7334 |     0.51 | 65.8%        | 9.07%               |
| xgboost       | 0.7139 | 0.4278 | 0.3108 |   0.2966 |  0.1986 |      0.7373 |     0.53 | 69.1%        | 9.54%               |
| random_forest | 0.7043 | 0.4086 | 0.2936 |   0.2861 |  0.1924 |      0.7271 |     0.49 | 66.9%        | 9.55%               |
| logreg        | 0.7041 | 0.4081 | 0.298  |   0.2832 |  0.2004 |      0.6951 |     0.49 | 65.9%        | 9.44%               |
| scorecard     | 0.704  | 0.4079 | 0.2959 |   0.2835 |  0.2012 |      0.6938 |     0.51 | 66.1%        | 9.47%               |

AUC train vs test muestra el grado de sobreajuste. `Aprobación` y `Default aprobados` describen la cartera que resultaría de rechazar todo lo que supera el umbral.

## Estabilidad de variables (PSI train → test)

33 de 64 variables por encima de 0.1. Las 10 con mayor deriva:

| feature                    |   psi | status    |
|:---------------------------|------:|:----------|
| num_bc_tl                  | 0.988 | inestable |
| num_tl_op_past_12m         | 0.974 | inestable |
| pct_tl_nvr_dlq             | 0.97  | inestable |
| total_il_high_credit_limit | 0.968 | inestable |
| mo_sin_old_rev_tl_op       | 0.968 | inestable |
| num_actv_rev_tl            | 0.968 | inestable |
| total_rev_hi_lim           | 0.967 | inestable |
| num_rev_tl_bal_gt_0        | 0.966 | inestable |
| num_op_rev_tl              | 0.966 | inestable |
| num_actv_bc_tl             | 0.964 | inestable |