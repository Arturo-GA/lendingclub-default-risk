"""Configuración central del proyecto.

Todo lo que define *qué* se modela vive aquí: variable objetivo, ventana temporal,
lista blanca de variables de originación y lista negra de variables con fuga de
información (leakage). Cambiar una decisión de negocio debería implicar tocar
solo este archivo.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

RAW_FILE = DATA_RAW / "accepted_2007_to_2018Q4.csv.gz"
PROCESSED_FILE = DATA_PROCESSED / "loans.parquet"

RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Autoría (la usan README, presentación y CI; único lugar donde se edita)
# --------------------------------------------------------------------------- #
PROJECT_TITLE = "Predicción de default en préstamos P2P"
AUTHORS = [
    "Rosario Sánchez Mosquipa",
    "Ruth Velarde Retamozo",
    "Juan Gutiérrez Aguilar",
    "Katherine Mauri Córdova",
    "Densel Castillón Medina",
]
PROGRAM = "Diplomado en Data Analytics, Pontificia Universidad Católica del Perú (2025)"
DATA_SOURCE = "LendingClub 2007–2018Q4 vía Kaggle (wordsforthewise/lending-club)"

# Umbral de PSI (Population Stability Index) a partir del cual una variable se
# considera inestable entre train y test: < 0.10 estable, 0.10-0.25 vigilar, > 0.25 inestable.
PSI_WARN = 0.10
PSI_ALERT = 0.25

# --------------------------------------------------------------------------- #
# Variable objetivo
# --------------------------------------------------------------------------- #
# En riesgo de crédito la clase positiva es el evento que se quiere detectar:
# el default (Charged Off = 1).
TARGET = "default"
STATUS_COL = "loan_status"
STATUS_MAP = {"Fully Paid": 0, "Charged Off": 1}

# --------------------------------------------------------------------------- #
# Ventana temporal y madurez
# --------------------------------------------------------------------------- #
# El extracto de Kaggle termina en 2018Q4. Un préstamo a 60 meses emitido en 2016
# todavía no venció en esa fecha; los que aparecen como Fully Paid / Charged Off
# son los que se resolvieron *antes* de tiempo (prepago o default temprano), lo
# que sesga la tasa de default de las cosechas recientes. Por eso solo se
# conservan préstamos cuyo plazo ya venció a la fecha de corte.
DATE_COL = "issue_d"
AS_OF_DATE = "2018-12-31"
MIN_ISSUE_DATE = "2010-01-01"  # antes de 2010 LendingClub era muy pequeño y distinto

# Split temporal (out-of-time), en fracciones del volumen ordenado por issue_d.
# Train: primeros 70 %, Validación: siguientes 15 %, Test: últimos 15 %.
SPLIT_FRACTIONS = {"train": 0.70, "valid": 0.15, "test": 0.15}

# --------------------------------------------------------------------------- #
# Variables conocidas en el momento de la originación (lista blanca)
# --------------------------------------------------------------------------- #
# Se usa lista blanca y no lista negra: si LendingClub agrega una columna nueva
# que no conocemos, por defecto queda fuera del modelo.

# Términos y precio del préstamo. int_rate / grade / sub_grade son la salida del
# propio modelo interno de LendingClub; están disponibles al originar, pero
# vale la pena reportar el modelo con y sin ellos (ver flag --exclude-pricing).
LOAN_TERMS = ["loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade"]
PRICING_COLS = ["int_rate", "grade", "sub_grade"]

BORROWER_PROFILE = [
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
    "initial_list_status",
]

CREDIT_BUREAU = [
    "dti",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "delinq_2yrs",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "pub_rec",
    "pub_rec_bankruptcies",
    "revol_bal",
    "revol_util",
    "total_acc",
    "collections_12_mths_ex_med",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "chargeoff_within_12_mths",
    "delinq_amnt",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "mths_since_recent_bc",
    "mths_since_recent_inq",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "tax_liens",
    "tot_cur_bal",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
    "total_rev_hi_lim",
]

RAW_FEATURES = LOAN_TERMS + BORROWER_PROFILE + CREDIT_BUREAU
RAW_COLUMNS_TO_LOAD = [STATUS_COL, DATE_COL] + RAW_FEATURES

# --------------------------------------------------------------------------- #
# Variables con fuga de información (lista negra explícita)
# --------------------------------------------------------------------------- #
# Documentadas una por una: cualquier persona que revise el repositorio debe
# poder ver por qué cada columna quedó fuera. `lcrisk.features.assert_no_leakage`
# falla si alguna de ellas llega a la matriz de entrenamiento.
LEAKAGE_COLUMNS: dict[str, str] = {
    # Se conocen solo cuando el préstamo ya terminó o está avanzado
    "last_pymnt_amnt": "Monto del último pago. Un último pago grande = préstamo cancelado; "
    "pequeño = charged off. Es el desenlace, no un predictor; incluirla lleva el AUC a ~1 "
    "(ver scripts/leakage_ablation.py).",
    "last_pymnt_d": "Fecha del último pago (posterior a la originación).",
    "next_pymnt_d": "Fecha del siguiente pago.",
    "last_credit_pull_d": "Fecha de la última consulta de LC al buró, posterior a la originación.",
    "last_fico_range_high": "FICO *actualizado* durante la vida del préstamo (correlación 0.77 con el target).",
    "last_fico_range_low": "Idem.",
    "total_pymnt": "Total pagado a la fecha.",
    "total_pymnt_inv": "Total pagado a inversionistas.",
    "total_rec_prncp": "Principal recuperado a la fecha.",
    "total_rec_int": "Interés recuperado a la fecha.",
    "total_rec_late_fee": "Cargos por mora cobrados.",
    "recoveries": "Recuperaciones post charge-off: solo existe si hubo default.",
    "collection_recovery_fee": "Comisión de cobranza post charge-off.",
    "out_prncp": "Principal pendiente a la fecha.",
    "out_prncp_inv": "Idem para inversionistas.",
    "funded_amnt": "Monto financiado; en la práctica igual a loan_amnt pero se fija después de listar.",
    "funded_amnt_inv": "Monto financiado por inversionistas (depende de la demanda posterior).",
    # Hardship / settlement: eventos posteriores al default o a la mora
    "hardship_flag": "Programa de dificultad de pago, posterior.",
    "hardship_type": "Idem.",
    "hardship_reason": "Idem.",
    "hardship_status": "Idem.",
    "deferral_term": "Idem.",
    "hardship_amount": "Idem.",
    "hardship_start_date": "Idem.",
    "hardship_end_date": "Idem.",
    "payment_plan_start_date": "Idem.",
    "hardship_length": "Idem.",
    "hardship_dpd": "Idem.",
    "hardship_loan_status": "Idem.",
    "orig_projected_additional_accrued_interest": "Idem.",
    "hardship_payoff_balance_amount": "Idem.",
    "hardship_last_payment_amount": "Idem.",
    "debt_settlement_flag": "Acuerdo de liquidación de deuda, posterior al default.",
    "debt_settlement_flag_date": "Idem.",
    "settlement_status": "Idem.",
    "settlement_date": "Idem.",
    "settlement_amount": "Idem.",
    "settlement_percentage": "Idem.",
    "settlement_term": "Idem.",
    "pymnt_plan": "Plan de pagos, se activa con mora.",
    # Identificadores y texto libre
    "id": "Identificador.",
    "member_id": "Identificador.",
    "url": "URL del préstamo.",
    "title": "Texto libre; equivalente a purpose (Cramér's V ≈ 1).",
    "emp_title": "Texto libre con decenas de miles de categorías.",
    "desc": "Texto libre.",
    "zip_code": "Primeros 3 dígitos del ZIP: ~900 categorías; se usa addr_state.",
    "policy_code": "Constante.",
    # Estado del préstamo: es el target
    "loan_status": "Es la variable objetivo.",
    "issue_d": "Se usa solo para el split temporal, no como predictor.",
}

# --------------------------------------------------------------------------- #
# Costos de negocio para la elección de umbral (ilustrativos, en unidades relativas)
# --------------------------------------------------------------------------- #
# Aprobar un préstamo que hace default cuesta más (pérdida de capital) que
# rechazar a un buen cliente (margen no ganado). El ratio es un parámetro de
# negocio; 5:1 es un valor razonable para consumo sin garantía y se puede
# sensibilizar en reports/.
COST_FALSE_NEGATIVE = 5.0  # default aprobado
COST_FALSE_POSITIVE = 1.0  # buen cliente rechazado
