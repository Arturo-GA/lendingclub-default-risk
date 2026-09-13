# Predicción de default en préstamos P2P — LendingClub

> *Credit-risk default prediction on LendingClub loans (2010–2018): leakage-free origination features, out-of-time validation, banking metrics (AUC/Gini/KS/PSI), WoE scorecard, cost-based cut-off and SHAP explainability. Spanish-language repo.*

Modelo de admisión de crédito que estima la probabilidad de que un préstamo P2P termine en **charged off**, usando únicamente la información disponible el día de la originación y validando sobre cosechas posteriores a las de entrenamiento, que es como se evalúa un modelo de riesgo antes de ponerlo en producción.

## Resultados

<!-- RESULTS:START — esta sección la reescribe scripts/train.py con reports/metrics/summary.md; no editar a mano -->
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
<!-- RESULTS:END -->

Con variables de originación, la literatura sobre este dataset sitúa el AUC alcanzable en torno a **0.70–0.73**. Un valor muy por encima casi siempre significa que se coló información posterior a la originación; `scripts/leakage_ablation.py` lo demuestra sobre estos mismos datos.

Figuras y tablas en `reports/`: curvas ROC/PR, calibración, KS, tasa de default por decil, PSI de estabilidad y SHAP.

## Datos

Extracto público de LendingClub 2007–2018Q4 (Kaggle, `wordsforthewise/lending-club`, archivo `accepted_2007_to_2018Q4.csv.gz`, ~1.6 GB, 2.26 M filas × 151 columnas). No se versiona; ver [`data/README.md`](data/README.md).

**Target:** `default = 1` si `loan_status == "Charged Off"`, 0 si `Fully Paid`. Los préstamos vigentes se excluyen. En riesgo de crédito la clase positiva es el evento a detectar.

**Filtros:**
- Originados desde 2010 (antes LendingClub era otra empresa en volumen y política).
- **Madurez:** solo préstamos cuyo plazo (36 o 60 meses) ya venció al cierre del extracto (2018-12). Sin este filtro, las cosechas 2016–2018 contienen solo los préstamos que se resolvieron *antes* de tiempo (prepagos y defaults tempranos), lo que sesga la tasa de default.

Quedan del orden de **800 k préstamos** con una tasa de default cercana al **15–20 %**.

**Variables:** ~60 numéricas y 6 categóricas conocidas al originar, en tres grupos: términos del préstamo, perfil del solicitante y buró de crédito. Se declaran en **lista blanca** (`config.RAW_FEATURES`); cualquier columna no listada queda fuera por defecto.

## Decisiones de diseño

**Fuga de información.** El dataset trae ~40 columnas que describen la *vida* del préstamo (`last_pymnt_amnt`, `total_rec_prncp`, `recoveries`, `last_fico_range_*`, hardship, settlement…). Son el desenlace, no predictores. Están documentadas una por una en `config.LEAKAGE_COLUMNS` y `features.assert_no_leakage` hace fallar el entrenamiento si alguna llega a la matriz de features. `scripts/leakage_ablation.py` entrena el mismo modelo con y sin ellas para cuantificar el efecto; con `last_pymnt_amnt` el AUC se acerca a 1.

**Validación out-of-time.** Split por fecha de originación: 70 % más antiguo entrena, 15 % siguiente valida (early stopping, tuning, umbral), 15 % más reciente es test y no se toca hasta el final. Un split aleatorio mezclaría cosechas y sobreestimaría el desempeño.

**Sin remuestreo.** El desbalance (≈ 80/20) se maneja con `class_weight` / `scale_pos_weight`. El test conserva la distribución real, así que las tasas de aprobación y de default reportadas son las que vería negocio.

**Preprocesamiento dentro del `Pipeline`.** Imputación, indicadores de faltante, escalado, one-hot y binning WoE se ajustan solo con train. Los faltantes son informativos (LendingClub incorporó las variables de buró en 2012), por eso se conservan como indicador en lugar de eliminar filas o columnas.

**Métricas de banca, no accuracy.** Con 80 % de buenos, aprobar a todos da accuracy 0.80. Se reporta AUC, Gini, KS, PR-AUC, Brier y calibración, más una **tabla de deciles** ("si rechazo el 20 % de peor score, ¿qué fracción de defaults evito y a cuántos buenos pierdo?").

**Umbral por costo.** Un default aprobado cuesta capital; un buen cliente rechazado cuesta margen. Con un ratio configurable (5:1 por defecto, `config.COST_*`) el umbral se elige en validación y se reporta la cartera resultante.

**Estabilidad (PSI).** Para cada variable se calcula el PSI entre train y test. Variables con PSI > 0.25 cambiaron de distribución entre cosechas y son candidatas a revisión aunque el AUC se vea bien. En este dataset la deriva más fuerte viene del patrón de faltantes de las variables de buró incorporadas en 2012.

**Modelos.** Regresión logística (baseline), **scorecard** (regresión logística sobre WoE, el estándar auditable en banca), Random Forest, XGBoost y LightGBM con early stopping. Tuning con Optuna sobre la validación temporal. Explicabilidad con SHAP.

**Precio de LendingClub.** `int_rate`, `grade` y `sub_grade` son legítimas al originar, pero encapsulan el modelo interno de la plataforma. `--exclude-pricing` reporta el modelo sin ellas (ablación).

## Cómo reproducir

```bash
git clone <repo> && cd lendingclub-default-risk
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Colocar accepted_2007_to_2018Q4.csv.gz en data/raw/
# 2. Dataset procesado (parquet, ~2 min; requiere ~6 GB de RAM para leer el CSV)
python scripts/make_dataset.py

# 3. Entrenar los modelos base y generar reportes
python scripts/train.py

# 4. Tuning, explicabilidad, verificación de leakage y presentación
python scripts/tune.py --model lightgbm --trials 40
python scripts/train.py --params reports/metrics/best_params.json
python scripts/explain.py --model lightgbm
python scripts/leakage_ablation.py
python scripts/make_deck.py

# Tests (datos sintéticos, no requieren el CSV)
pytest -q
```

O todo de una vez: `make all` (dataset → entrenamiento → SHAP → ablación de leakage → presentación). `scripts/train.py` reescribe la sección *Resultados* de este README con las métricas reales; nunca lo hace con datos sintéticos.

Prueba rápida sin el dataset real: `python tests/synthetic.py && python scripts/make_dataset.py --raw data/raw/synthetic.csv.gz && python scripts/train.py`. El CI de GitHub ejecuta exactamente eso en cada push.

## Estructura

```
src/lcrisk/
  config.py      target, ventana temporal, lista blanca de features, lista negra de leakage, costos, umbrales PSI
  data.py        carga selectiva de columnas, target, filtro de madurez, split temporal
  features.py    ingeniería de variables + guarda contra leakage
  selection.py   WoE / IV, Cramér's V, poda por correlación, WoEEncoder (scorecard)
  pipeline.py    ColumnTransformer + modelos, early stopping
  evaluate.py    AUC, Gini, KS, PR-AUC, Brier, calibración, deciles, umbral por costo, PSI
  tune.py        Optuna sobre validación temporal
  explain.py     SHAP
scripts/         make_dataset · train · tune · explain · leakage_ablation · make_deck
notebooks/       01_eda · 02_resultados (interpretan, no entrenan)
tests/           leakage, datos, pipeline, WoE/PSI (con generador sintético)
reports/         metrics/*.json|md|csv · figures/*.png
.github/         CI: tests + smoke run
```

## Limitaciones y siguientes pasos

- Solo modela PD; para pricing o provisiones haría falta LGD/EAD.
- No hay validación por cosecha individual (*vintage curves*) ni monitoreo continuo; el PSI train→test es el primer paso.
- Sesgo de selección: solo se observan préstamos aprobados por LendingClub (*reject inference* pendiente).
- El scorecard usa binning por cuantiles; un binning óptimo con restricción de monotonía mejoraría su interpretabilidad.

## Presentación

`python scripts/make_deck.py` genera `reports/presentacion.pptx` (11 diapositivas) a partir de los reportes: cada número y figura sale de `reports/`, así que la presentación siempre coincide con la última corrida. Autores y programa se leen de `config.py`.

## Autores

Rosario Sánchez Mosquipa · Ruth Velarde Retamozo · Juan Gutiérrez Aguilar · Katherine Mauri Córdova · Densel Castillón Medina

Proyecto del Diplomado en Data Analytics, Pontificia Universidad Católica del Perú (2025). Datos: LendingClub 2007–2018Q4 vía Kaggle (`wordsforthewise/lending-club`).
