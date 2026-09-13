"""Genera la presentación del proyecto a partir de los reportes en reports/.

Todo número y figura del deck sale de los archivos que producen los scripts
(meta.json, metrics.json, leakage_ablation.md, psi, figuras). Si falta alguno,
la diapositiva lo indica en vez de inventar datos.

Uso:
    python scripts/make_deck.py [--out reports/presentacion.pptx]

Autores y programa se leen de lcrisk.config (AUTHORS, PROGRAM).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lcrisk import config as C  # noqa: E402

# --------------------------------------------------------------------------- #
# Estilo
# --------------------------------------------------------------------------- #
W, H = Inches(13.333), Inches(7.5)
NAVY = RGBColor(0x14, 0x2B, 0x4A)
INK = RGBColor(0x22, 0x22, 0x22)
GREY = RGBColor(0x6B, 0x72, 0x80)
ACCENT = RGBColor(0xC8, 0x3E, 0x2E)
LIGHT = RGBColor(0xF3, 0xF4, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"

MARGIN = Inches(0.6)
CONTENT_TOP = Inches(1.35)


def _text(slide, x, y, w, h, text, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT, font=FONT):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, font
        r.font.color.rgb = color
        p.space_after = Pt(4)
    return box


def _bullets(slide, x, y, w, h, items, size=15, color=INK):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        bold_head = None
        if isinstance(item, tuple):
            bold_head, item = item
        if bold_head:
            r = p.add_run(); r.text = f"{bold_head}  "; r.font.bold = True
            r.font.size, r.font.name = Pt(size), FONT; r.font.color.rgb = NAVY
        r = p.add_run(); r.text = item
        r.font.size, r.font.name = Pt(size), FONT; r.font.color.rgb = color
        p.space_after = Pt(9)
    return box


def _rect(slide, x, y, w, h, fill=LIGHT):
    from pptx.enum.shapes import MSO_SHAPE

    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def _title(slide, text, sub=None):
    _text(slide, MARGIN, Inches(0.45), W - 2 * MARGIN, Inches(0.6), text, size=28, bold=True, color=NAVY)
    if sub:
        _text(slide, MARGIN, Inches(1.0), W - 2 * MARGIN, Inches(0.35), sub, size=13, color=GREY)


def _footer(slide, n, demo):
    _text(slide, MARGIN, H - Inches(0.45), Inches(9), Inches(0.3),
          f"{C.PROJECT_TITLE} · LendingClub" + ("   —   DEMO CON DATOS SINTÉTICOS" if demo else ""),
          size=10, color=ACCENT if demo else GREY)
    _text(slide, W - MARGIN - Inches(1), H - Inches(0.45), Inches(1), Inches(0.3), str(n), size=10, color=GREY, align=PP_ALIGN.RIGHT)


def _table(slide, x, y, w, df: pd.DataFrame, size=12, col_widths=None, row_h=Inches(0.36)):
    rows, cols = len(df) + 1, len(df.columns)
    shape = slide.shapes.add_table(rows, cols, x, y, w, row_h * rows)
    tbl = shape.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = cw
    for j, col in enumerate(df.columns):
        cell = tbl.cell(0, j); cell.text = str(col)
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.size, r.font.bold, r.font.name = Pt(size), True, FONT; r.font.color.rgb = WHITE
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j); cell.text = str(val)
            cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size, r.font.name = Pt(size), FONT; r.font.color.rgb = INK
    return shape


def _image(slide, path: Path, x, y, w=None, h=None, missing_msg=""):
    if path.exists():
        return slide.shapes.add_picture(str(path), x, y, width=w, height=h)
    _rect(slide, x, y, w or Inches(5), h or Inches(3.5))
    _text(slide, x + Inches(0.2), y + Inches(0.2), (w or Inches(5)) - Inches(0.4), Inches(1), missing_msg or f"Falta {path.name}", size=12, color=GREY)


def _fit_image(slide, path: Path, x, y, max_w, max_h, missing_msg=""):
    """Inserta la imagen respetando proporción dentro de la caja."""
    if not path.exists():
        return _image(slide, path, x, y, max_w, max_h, missing_msg)
    from PIL import Image

    iw, ih = Image.open(path).size
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    return slide.shapes.add_picture(str(path), x + (max_w - w) // 2, y, width=Emu(w), height=Emu(h))


# --------------------------------------------------------------------------- #
# Carga de reportes
# --------------------------------------------------------------------------- #
def load_reports() -> dict:
    r: dict = {}
    meta_p = C.PROCESSED_FILE.parent / "meta.json"
    r["meta"] = json.loads(meta_p.read_text()) if meta_p.exists() else None
    m_p = C.METRICS_DIR / "metrics.json"
    r["metrics"] = json.loads(m_p.read_text()) if m_p.exists() else None
    l_p = C.METRICS_DIR / "leakage_ablation.md"
    r["leakage"] = _md_table(l_p.read_text()) if l_p.exists() else None
    psi_p = C.METRICS_DIR / "psi_train_vs_test.csv"
    r["psi"] = pd.read_csv(psi_p) if psi_p.exists() else None
    r["best"] = None
    if r["metrics"]:
        r["best"] = max(r["metrics"]["metrics"], key=lambda k: r["metrics"]["metrics"][k]["test"]["auc"])
        dec_p = C.METRICS_DIR / f"decile_{r['best']}.csv"
        r["deciles"] = pd.read_csv(dec_p) if dec_p.exists() else None
        shap_p = C.METRICS_DIR / f"shap_importance_{r['best']}.csv"
        r["shap"] = pd.read_csv(shap_p) if shap_p.exists() else None
    return r


def _md_table(md: str) -> pd.DataFrame | None:
    lines = [ln for ln in md.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return None
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]
    return pd.DataFrame(rows, columns=header)


# --------------------------------------------------------------------------- #
# Diapositivas
# --------------------------------------------------------------------------- #
def build(out: Path) -> None:
    authors, program = C.AUTHORS, C.PROGRAM
    R = load_reports()
    demo = bool(R["meta"] and R["meta"].get("synthetic"))
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]
    n = 0

    def new():
        nonlocal n
        n += 1
        s = prs.slides.add_slide(blank)
        if n > 1:
            _footer(s, n, demo)
        return s

    # 1 · Portada -------------------------------------------------------------
    s = new()
    _rect(s, 0, 0, W, H, NAVY)
    _text(s, MARGIN, Inches(2.2), W - 2 * MARGIN, Inches(1.4), "Predicción de default en préstamos P2P", size=40, bold=True, color=WHITE)
    _text(s, MARGIN, Inches(3.5), W - 2 * MARGIN, Inches(0.8),
          "Modelo de admisión de crédito sobre datos de LendingClub, validado out-of-time", size=20, color=RGBColor(0xC9, 0xD3, 0xE0))
    _text(s, MARGIN, Inches(5.05), W - 2 * MARGIN, Inches(0.9), " · ".join(authors), size=15, color=WHITE)
    _text(s, MARGIN, Inches(6.05), W - 2 * MARGIN, Inches(0.4), program, size=13, color=RGBColor(0xC9, 0xD3, 0xE0))
    if demo:
        _text(s, MARGIN, Inches(6.7), W - 2 * MARGIN, Inches(0.4), "DEMO GENERADA CON DATOS SINTÉTICOS — no son resultados reales", size=13, bold=True, color=RGBColor(0xFF, 0xB4, 0xA8))

    # 2 · Problema ------------------------------------------------------------
    s = new()
    _title(s, "El problema", "Por qué predecir el default importa en una plataforma P2P")
    _bullets(s, MARGIN, CONTENT_TOP, Inches(6.2), Inches(5), [
        ("Contexto.", "En un préstamo P2P el inversionista asume el riesgo directamente; no hay banco que absorba las pérdidas."),
        ("Pregunta.", "Dado un solicitante, ¿qué probabilidad tiene de no pagar? Y con eso, ¿a quién aprobar y a qué precio?"),
        ("Restricción clave.", "La decisión se toma el día de la originación. El modelo solo puede usar lo que se sabía ese día."),
        ("Cómo se evalúa.", "Entrenando con cosechas antiguas y midiendo sobre cosechas posteriores (out-of-time), como se valida un modelo antes de producción."),
    ])
    _rect(s, Inches(7.3), CONTENT_TOP, Inches(5.4), Inches(4.6))
    _text(s, Inches(7.6), CONTENT_TOP + Inches(0.3), Inches(4.8), Inches(0.4), "Entregables del modelo", size=15, bold=True, color=NAVY)
    _bullets(s, Inches(7.6), CONTENT_TOP + Inches(0.85), Inches(4.8), Inches(3.5), [
        "Probabilidad de default por solicitud (PD)",
        "Ordenamiento de riesgo: AUC / Gini / KS",
        "Punto de corte según costo de negocio",
        "Tabla de deciles para discutir política de aprobación",
        "Explicabilidad por variable (SHAP) y scorecard auditable",
    ], size=13)

    # 3 · Datos ---------------------------------------------------------------
    s = new()
    _title(s, "Datos", "LendingClub 2007–2018Q4 (Kaggle) · 2.26 M préstamos · 151 columnas")
    meta = R["meta"]
    left = [
        ("Target.", "default = 1 si el préstamo terminó Charged Off; 0 si Fully Paid. Los vigentes se excluyen."),
        ("Desde 2010.", "Antes, LendingClub era otra empresa en volumen y política de crédito."),
        ("Madurez.", "Solo préstamos cuyo plazo (36/60 m) venció antes del cierre del extracto. Sin esto, las cosechas recientes solo tienen prepagos y defaults tempranos."),
        ("Variables.", "~60 numéricas + 6 categóricas conocidas al originar: términos, perfil del solicitante y buró. Lista blanca explícita."),
    ]
    _bullets(s, MARGIN, CONTENT_TOP, Inches(6.4), Inches(5), left, size=14)
    if meta:
        tab = pd.DataFrame([
            {"Conjunto": k, "Préstamos": f"{v['n']:,}", "Originación": f"{v['issue_min'][:7]} → {v['issue_max'][:7]}", "Tasa default": f"{100*v['default_rate']:.1f}%"}
            for k, v in [(k, meta["splits"][k]) for k in ("train", "valid", "test") if k in meta["splits"]]
        ])
        _text(s, Inches(7.4), CONTENT_TOP, Inches(5.3), Inches(0.4), f"Después de filtros: {meta['n_rows']:,} préstamos · tasa de default {100*meta['default_rate']:.1f}%", size=13, bold=True, color=NAVY)
        _table(s, Inches(7.4), CONTENT_TOP + Inches(0.55), Inches(5.3), tab, size=12,
               col_widths=[Inches(0.9), Inches(1.2), Inches(2.0), Inches(1.2)])
        _text(s, Inches(7.4), CONTENT_TOP + Inches(2.3), Inches(5.3), Inches(1.2),
              "Split temporal: los préstamos más antiguos entrenan; los más recientes son test y no se tocan hasta el final. "
              "Validación se usa para early stopping, tuning y elección del umbral.", size=12, color=GREY)
    else:
        _text(s, Inches(7.4), CONTENT_TOP, Inches(5.3), Inches(1), "Ejecuta scripts/make_dataset.py para completar esta tabla.", size=13, color=ACCENT)

    # 4 · Leakage -------------------------------------------------------------
    s = new()
    _title(s, "Fuga de información: la trampa de este dataset", "Cuarenta columnas describen la vida del préstamo, no al solicitante")
    _bullets(s, MARGIN, CONTENT_TOP, Inches(6.2), Inches(5), [
        ("Ejemplos.", "last_pymnt_amnt, total_rec_prncp, recoveries, last_fico_range_*, hardship_*, settlement_*."),
        ("Por qué engañan.", "Un último pago grande = préstamo cancelado; pequeño = charged off. Es el desenlace, no un predictor."),
        ("Defensa.", "Lista negra documentada columna por columna en config.py y una guarda (assert_no_leakage) que hace fallar el entrenamiento si alguna llega a la matriz de features."),
        ("Verificación.", "Mismo modelo y mismo split, cambiando solo qué columnas se permiten:"),
    ], size=14)
    if R["leakage"] is not None:
        _table(s, Inches(7.0), CONTENT_TOP + Inches(0.2), Inches(5.7), R["leakage"], size=12,
               col_widths=[Inches(3.2), Inches(0.9), Inches(0.8), Inches(0.8)])
        _text(s, Inches(7.0), CONTENT_TOP + Inches(2.1), Inches(5.7), Inches(1),
              "Un AUC cercano a 1 en un problema de crédito de consumo no es un buen modelo: es información del futuro.", size=13, color=ACCENT, bold=True)
    else:
        _text(s, Inches(7.0), CONTENT_TOP + Inches(0.2), Inches(5.7), Inches(1), "Ejecuta scripts/leakage_ablation.py para completar esta tabla.", size=13, color=ACCENT)

    # 5 · Metodología ---------------------------------------------------------
    s = new()
    _title(s, "Metodología", "Decisiones que un modelo de riesgo necesita antes de elegir algoritmo")
    colw = Inches(3.9)
    for i, (head, items) in enumerate([
        ("Preprocesamiento", ["Todo dentro de un Pipeline de scikit-learn: se ajusta solo con train",
                              "Faltantes informativos (buró desde 2012): mediana + indicador",
                              "Categorías raras agrupadas (< 0.5 %)",
                              "Variables derivadas: FICO medio, antigüedad crediticia, cuota/ingreso, préstamo/ingreso"]),
        ("Desbalance y métricas", ["Sin remuestreo: class_weight / scale_pos_weight",
                                   "El test conserva la tasa real (~15–20 %)",
                                   "Se reporta AUC, Gini, KS, PR-AUC, Brier y calibración",
                                   "Accuracy no: aprobar a todos ya da ~0.80"]),
        ("Modelos", ["Regresión logística (baseline)",
                     "Scorecard: logística sobre WoE, auditable",
                     "Random Forest, XGBoost, LightGBM con early stopping",
                     "Optuna sobre la validación temporal · SHAP"]),
    ]):
        x = MARGIN + i * (colw + Inches(0.25))
        _rect(s, x, CONTENT_TOP, colw, Inches(5.2))
        _text(s, x + Inches(0.25), CONTENT_TOP + Inches(0.25), colw - Inches(0.5), Inches(0.4), head, size=16, bold=True, color=NAVY)
        _bullets(s, x + Inches(0.25), CONTENT_TOP + Inches(0.85), colw - Inches(0.5), Inches(4.2), items, size=13)

    # 6 · Resultados ----------------------------------------------------------
    s = new()
    M = R["metrics"]
    if M:
        base = next(iter(M["metrics"].values()))["test"]["base_rate"]
        _title(s, "Resultados (test out-of-time)", f"Tasa de default en test: {100*base:.1f}% · variables de precio de LendingClub {'excluidas' if M['exclude_pricing'] else 'incluidas'}")
        rows = []
        for name, m in M["metrics"].items():
            rows.append({"Modelo": name, "AUC": f"{m['test']['auc']:.3f}", "Gini": f"{m['test']['gini']:.3f}", "KS": f"{m['test']['ks']:.3f}",
                         "PR-AUC": f"{m['test']['pr_auc']:.3f}", "Brier": f"{m['test']['brier']:.3f}", "AUC train": f"{m['train']['auc']:.3f}"})
        tab = pd.DataFrame(rows).sort_values("AUC", ascending=False)
        _table(s, MARGIN, CONTENT_TOP, Inches(7.4), tab, size=13,
               col_widths=[Inches(1.9), Inches(0.9), Inches(0.9), Inches(0.9), Inches(0.95), Inches(0.9), Inches(0.95)])
        best = R["best"]; bm = M["metrics"][best]["test"]
        _rect(s, Inches(8.4), CONTENT_TOP, Inches(4.3), Inches(3.2))
        _text(s, Inches(8.65), CONTENT_TOP + Inches(0.25), Inches(3.8), Inches(0.4), f"Mejor modelo: {best}", size=16, bold=True, color=NAVY)
        _bullets(s, Inches(8.65), CONTENT_TOP + Inches(0.8), Inches(3.8), Inches(2.4), [
            f"AUC {bm['auc']:.3f} → Gini {bm['gini']:.3f}",
            f"KS {bm['ks']:.3f}: separación máxima entre buenos y malos",
            f"Brecha train–test AUC: {M['metrics'][best]['train']['auc'] - bm['auc']:+.3f}",
        ], size=13)
        _text(s, MARGIN, CONTENT_TOP + Inches(3.6), Inches(12), Inches(1.2),
              "La brecha AUC train vs test mide sobreajuste. Con variables de originación, en este dataset la literatura sitúa el AUC alcanzable "
              "alrededor de 0.70–0.73; valores muy superiores indican fuga de información.", size=12, color=GREY)
    else:
        _title(s, "Resultados", "Pendiente")
        _text(s, MARGIN, CONTENT_TOP, Inches(12), Inches(1), "Ejecuta scripts/train.py para completar esta diapositiva.", size=14, color=ACCENT)

    # 7 · Curvas --------------------------------------------------------------
    s = new()
    _title(s, "Ordenamiento, precisión sobre la clase minoritaria y calibración")
    _fit_image(s, C.FIGURES_DIR / "roc_pr.png", MARGIN, CONTENT_TOP, Inches(8.1), Inches(5.0), "Falta reports/figures/roc_pr.png (scripts/train.py)")
    _fit_image(s, C.FIGURES_DIR / "calibration.png", Inches(8.9), CONTENT_TOP, Inches(3.9), Inches(3.9), "Falta calibration.png")
    _text(s, Inches(8.9), CONTENT_TOP + Inches(4.05), Inches(3.9), Inches(1),
          "Calibración: si el modelo dice 20 %, ~20 % de esos clientes deben hacer default. Necesario para fijar tasas y límites.", size=11, color=GREY)

    # 8 · Deciles y umbral ----------------------------------------------------
    s = new()
    _title(s, "De la probabilidad a la decisión", "Tabla de deciles y punto de corte por costo")
    if R.get("deciles") is not None:
        d = R["deciles"].copy()
        d = pd.DataFrame({"Decil": d["decile"], "Préstamos": d["n"].map("{:,}".format), "Tasa default": (d["default_rate"] * 100).map("{:.1f}%".format),
                          "Defaults acum.": (d["cum_defaults_pct"] * 100).map("{:.0f}%".format), "Buenos acum.": (d["cum_goods_pct"] * 100).map("{:.0f}%".format), "Lift": d["lift"].map("{:.2f}".format)})
        _table(s, MARGIN, CONTENT_TOP, Inches(6.6), d, size=11, row_h=Inches(0.33),
               col_widths=[Inches(0.7), Inches(1.2), Inches(1.2), Inches(1.3), Inches(1.2), Inches(0.9)])
        _text(s, MARGIN, CONTENT_TOP + Inches(3.9), Inches(6.6), Inches(0.9),
              "Decil 1 = 10 % de mayor score. \"Defaults acum.\" es la fracción de todos los defaults que se evita rechazando hasta ese decil; \"Buenos acum.\" lo que cuesta en clientes buenos.", size=11, color=GREY)
    else:
        _text(s, MARGIN, CONTENT_TOP, Inches(6.6), Inches(1), "Ejecuta scripts/train.py para la tabla de deciles.", size=13, color=ACCENT)
    if M:
        t = M["thresholds"][R["best"]]
        _rect(s, Inches(7.5), CONTENT_TOP, Inches(5.2), Inches(3.0))
        _text(s, Inches(7.75), CONTENT_TOP + Inches(0.25), Inches(4.7), Inches(0.4), f"Punto de corte ({R['best']})", size=16, bold=True, color=NAVY)
        _bullets(s, Inches(7.75), CONTENT_TOP + Inches(0.8), Inches(4.7), Inches(2.2), [
            f"Costo FN : FP = {C.COST_FALSE_NEGATIVE:.0f} : {C.COST_FALSE_POSITIVE:.0f}, umbral elegido en validación: {t['threshold']:.2f}",
            f"Aprobación: {100*t['approval_rate']:.1f}% de las solicitudes",
            f"Tasa de default de los aprobados: {100*t['default_rate_approved']:.1f}% (vs {100*t['default_rate_baseline']:.1f}% sin modelo)",
            f"Defaults detectados (recall): {100*t['recall_default']:.0f}%",
        ], size=12)
        _fit_image(s, C.FIGURES_DIR / f"deciles_{R['best']}.png", Inches(7.5), CONTENT_TOP + Inches(3.15), Inches(5.2), Inches(2.3), "")

    # 9 · Estabilidad y explicabilidad ---------------------------------------
    s = new()
    _title(s, "Estabilidad y explicabilidad", "PSI train → test por variable · contribución SHAP del mejor modelo")
    if R["psi"] is not None:
        psi = R["psi"]
        n_warn = int((psi["psi"] > C.PSI_WARN).sum())
        _text(s, MARGIN, CONTENT_TOP, Inches(5.2), Inches(0.4), f"{n_warn} de {len(psi)} variables con PSI > {C.PSI_WARN}", size=14, bold=True, color=NAVY)
        top = psi.head(8)[["feature", "psi", "status"]].copy(); top["psi"] = top["psi"].round(2)
        top.columns = ["Variable", "PSI", "Estado"]
        _table(s, MARGIN, CONTENT_TOP + Inches(0.5), Inches(5.2), top, size=11, row_h=Inches(0.32), col_widths=[Inches(2.9), Inches(0.9), Inches(1.4)])
        _text(s, MARGIN, CONTENT_TOP + Inches(3.6), Inches(5.2), Inches(1.4),
              "PSI < 0.10 estable · 0.10–0.25 vigilar · > 0.25 cambio significativo. En este dataset la deriva más fuerte viene del patrón de faltantes: "
              "las variables de buró se incorporaron en 2012 y están vacías en las cosechas antiguas.", size=11, color=GREY)
    else:
        _text(s, MARGIN, CONTENT_TOP, Inches(5.2), Inches(1), "Ejecuta scripts/train.py para el PSI.", size=13, color=ACCENT)
    shap_png = C.FIGURES_DIR / f"shap_summary_{R['best']}.png" if R["best"] else C.FIGURES_DIR / "shap_summary.png"
    _fit_image(s, shap_png, Inches(6.2), CONTENT_TOP, Inches(6.5), Inches(5.2), "Falta el summary plot de SHAP (scripts/explain.py)")

    # 10 · Conclusiones -------------------------------------------------------
    s = new()
    _title(s, "Conclusiones y siguientes pasos")
    concl = [
        ("Validación honesta.", "Split temporal, sin remuestreo del test y sin variables posteriores a la originación. El AUC resultante es el que se puede esperar en producción."),
        ("Decisión, no solo score.", "Umbral por costo y tabla de deciles permiten discutir con negocio cuánto riesgo aceptar."),
        ("Auditable.", "Scorecard WoE al lado de los boosters; SHAP para el modelo elegido; lista negra de leakage con test automatizado."),
    ]
    if M:
        concl.insert(0, ("Resultado.", f"Mejor modelo {R['best']} con AUC {M['metrics'][R['best']]['test']['auc']:.3f} / KS {M['metrics'][R['best']]['test']['ks']:.3f} en test out-of-time."))
    _bullets(s, MARGIN, CONTENT_TOP, Inches(6.4), Inches(5), concl, size=14)
    _rect(s, Inches(7.4), CONTENT_TOP, Inches(5.3), Inches(4.4))
    _text(s, Inches(7.65), CONTENT_TOP + Inches(0.25), Inches(4.8), Inches(0.4), "Siguientes pasos", size=16, bold=True, color=NAVY)
    _bullets(s, Inches(7.65), CONTENT_TOP + Inches(0.8), Inches(4.8), Inches(3.4), [
        "Curvas por cosecha (vintage) y monitoreo continuo de PSI",
        "Binning óptimo con monotonía para el scorecard",
        "Reject inference: solo se observan préstamos aprobados",
        "LGD / EAD para pasar de PD a pérdida esperada y pricing",
    ], size=13)

    # 11 · Créditos -----------------------------------------------------------
    s = new()
    _title(s, "Créditos")
    _text(s, MARGIN, CONTENT_TOP, Inches(12), Inches(0.4), "Autores", size=16, bold=True, color=NAVY)
    _bullets(s, MARGIN, CONTENT_TOP + Inches(0.5), Inches(12), Inches(2.6), authors, size=14)
    _bullets(s, MARGIN, CONTENT_TOP + Inches(3.0), Inches(12), Inches(2), [
        ("Programa.", program + "."),
        ("Datos.", C.DATA_SOURCE + "."),
        ("Código.", "Repositorio reproducible: tests, CI y generación automática de reportes y de esta presentación."),
    ], size=14)

    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(f"Presentación guardada en {out} ({n} diapositivas){' — DEMO sintética' if demo else ''}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(C.REPORTS_DIR / "presentacion.pptx"))
    a = p.parse_args()
    build(Path(a.out))


if __name__ == "__main__":
    main()
