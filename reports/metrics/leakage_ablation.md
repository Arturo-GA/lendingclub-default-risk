# Ablación de fuga de información

Mismo modelo y mismo split temporal; solo cambian las columnas disponibles.

| Escenario                                     |   AUC test |   Gini |     KS |
|:----------------------------------------------|-----------:|-------:|-------:|
| Solo originación (este repo)                  |     0.7138 | 0.4277 | 0.3116 |
| + last_pymnt_amnt                             |     0.9005 | 0.8011 | 0.6611 |
| + last_pymnt_amnt, last_fico, total_rec_prncp |     0.9998 | 0.9996 | 0.997  |
