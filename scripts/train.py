"""Entrena los modelos base, evalúa out-of-time y genera reportes.

Uso:
    python scripts/train.py
    python scripts/train.py --models logreg lightgbm --exclude-pricing
    python scripts/train.py --params reports/metrics/best_params_lightgbm.json

Salidas:
    models/<modelo>.joblib
    reports/metrics/metrics.json, decile_<modelo>.csv, summary.md
    reports/figures/roc_pr.png, calibration.png, ks_<modelo>.png, deciles_<modelo>.png
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcrisk import config as C  # noqa: E402
from lcrisk.data import load_processed  # noqa: E402
from lcrisk.evaluate import (  # noqa: E402
    apply_threshold_report,
    choose_threshold,
    compute_metrics,
    decile_table,
    plot_calibration,
    plot_decile_table,
    plot_ks,
    plot_roc_pr,
    psi_table,
)
from lcrisk.features import engineer, get_xy  # noqa: E402
from lcrisk.logging_utils import setup_logging  # noqa: E402
from lcrisk.pipeline import MODEL_NAMES, fit, make_model, pos_weight_from  # noqa: E402

log = logging.getLogger("train")


def write_summary(metrics: dict, thresholds: dict, out: Path, exclude_pricing: bool, stability: pd.DataFrame | None = None) -> None:
    rows = []
    for name, m in metrics.items():
        t = thresholds[name]
        rows.append({
            "Modelo": name,
            "AUC": f"{m['test']['auc']:.4f}",
            "Gini": f"{m['test']['gini']:.4f}",
            "KS": f"{m['test']['ks']:.4f}",
            "PR-AUC": f"{m['test']['pr_auc']:.4f}",
            "Brier": f"{m['test']['brier']:.4f}",
            "AUC train": f"{m['train']['auc']:.4f}",
            "Umbral": f"{t['threshold']:.2f}",
            "Aprobación": f"{100*t['approval_rate']:.1f}%",
            "Default aprobados": f"{100*t['default_rate_approved']:.2f}%",
        })
    tab = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    base = next(iter(metrics.values()))["test"]["base_rate"]
    lines = [
        "# Resultados (test out-of-time)",
        "",
        f"Variables de precio de LendingClub (int_rate, grade): **{'excluidas' if exclude_pricing else 'incluidas'}**.  ",
        f"Tasa de default en test: **{100*base:.2f}%**. Umbral elegido en validación con costo FN:FP = {C.COST_FALSE_NEGATIVE:.0f}:{C.COST_FALSE_POSITIVE:.0f}.",
        "",
        tab.to_markdown(index=False),
        "",
        "AUC train vs test muestra el grado de sobreajuste. `Aprobación` y `Default aprobados` "
        "describen la cartera que resultaría de rechazar todo lo que supera el umbral.",
    ]
    if stability is not None:
        top = stability.head(10).copy()
        top["psi"] = top["psi"].round(3)
        lines += ["", "## Estabilidad de variables (PSI train → test)", "",
                  f"{int((stability['psi'] > C.PSI_WARN).sum())} de {len(stability)} variables por encima de {C.PSI_WARN}. Las 10 con mayor deriva:", "",
                  top.to_markdown(index=False)]
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Resumen escrito en %s", out)


def update_readme(summary_path: Path, readme: Path = C.ROOT / "README.md") -> None:
    """Reemplaza la sección de resultados del README con el contenido de summary.md."""
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    start_tag, end_tag = "<!-- RESULTS:START", "<!-- RESULTS:END -->"
    if start_tag not in text or end_tag not in text:
        log.warning("README sin marcadores RESULTS; no se actualiza")
        return
    start = text.index(start_tag)
    start = text.index("-->", start) + 3
    end = text.index(end_tag)
    body = summary_path.read_text(encoding="utf-8").split("\n", 1)[1]  # sin el título "# Resultados"
    readme.write_text(text[:start] + "\n" + body.strip() + "\n" + text[end:], encoding="utf-8")
    log.info("README actualizado con los resultados")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(C.PROCESSED_FILE))
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--exclude-pricing", action="store_true", help="Quitar int_rate/grade (ablación)")
    parser.add_argument("--params", default=None, help="JSON {modelo: {hiperparámetros}} (p. ej. salida de tune.py)")
    parser.add_argument("--sample-frac", type=float, default=None, help="Submuestrear train (debug)")
    parser.add_argument("--no-readme", action="store_true", help="No reescribir la sección Resultados del README")
    args = parser.parse_args()

    setup_logging()
    C.MODELS_DIR.mkdir(exist_ok=True, parents=True)
    C.FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    C.METRICS_DIR.mkdir(exist_ok=True, parents=True)

    df = engineer(load_processed(args.data))
    X, y, num_cols, cat_cols = get_xy(df, exclude_pricing=args.exclude_pricing)
    split = df["split"]
    log.info("Features: %d numéricas + %d categóricas", len(num_cols), len(cat_cols))

    parts = {s: (X[split == s], y[split == s]) for s in ("train", "valid", "test")}
    if args.sample_frac:
        Xt, yt = parts["train"]
        idx = Xt.sample(frac=args.sample_frac, random_state=C.RANDOM_STATE).index
        parts["train"] = (Xt.loc[idx], yt.loc[idx])
    pos_weight = pos_weight_from(parts["train"][1])
    log.info("pos_weight (neg/pos) = %.2f", pos_weight)

    # Estabilidad de las variables entre train y test (out-of-time): si una variable
    # cambió mucho de distribución, el modelo puede degradarse aunque el AUC se vea bien.
    stability = psi_table(parts["train"][0], parts["test"][0])
    stability.to_csv(C.METRICS_DIR / "psi_train_vs_test.csv", index=False)
    unstable = stability[stability["psi"] > C.PSI_WARN]
    log.info("PSI train→test: %d variables por encima de %.2f: %s", len(unstable), C.PSI_WARN,
             ", ".join(f"{r.feature}={r.psi:.2f}" for r in unstable.head(8).itertuples()) or "ninguna")

    params_all = json.loads(Path(args.params).read_text()) if args.params else {}
    metrics, thresholds, curves = {}, {}, {}

    for name in args.models:
        t0 = time.time()
        pipe = make_model(name, num_cols, cat_cols, pos_weight, params_all.get(name))
        fit(pipe, *parts["train"], *parts["valid"])
        scores = {s: pipe.predict_proba(parts[s][0])[:, 1] for s in parts}
        metrics[name] = {s: compute_metrics(parts[s][1], scores[s]) for s in parts}

        thr = choose_threshold(parts["valid"][1], scores["valid"])
        thresholds[name] = apply_threshold_report(parts["test"][1], scores["test"], thr["threshold"])
        curves[name] = (parts["test"][1].to_numpy(), scores["test"])

        dec = decile_table(parts["test"][1], scores["test"])
        dec.to_csv(C.METRICS_DIR / f"decile_{name}.csv", index=False)
        plot_decile_table(dec, C.FIGURES_DIR / f"deciles_{name}.png", title=f"— {name}")
        plot_ks(parts["test"][1], scores["test"], C.FIGURES_DIR / f"ks_{name}.png", title=f"— {name}")
        joblib.dump(pipe, C.MODELS_DIR / f"{name}.joblib")

        m = metrics[name]["test"]
        log.info(
            "%-13s AUC test %.4f | Gini %.4f | KS %.4f | PR-AUC %.4f | AUC train %.4f | %.0fs",
            name, m["auc"], m["gini"], m["ks"], m["pr_auc"], metrics[name]["train"]["auc"], time.time() - t0,
        )

    plot_roc_pr(curves, C.FIGURES_DIR / "roc_pr.png")
    plot_calibration(curves, C.FIGURES_DIR / "calibration.png")
    (C.METRICS_DIR / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "thresholds": thresholds, "exclude_pricing": args.exclude_pricing,
                    "features": {"numeric": num_cols, "categorical": cat_cols}}, indent=2),
        encoding="utf-8",
    )
    write_summary(metrics, thresholds, C.METRICS_DIR / "summary.md", args.exclude_pricing, stability)
    meta_p = C.PROCESSED_FILE.parent / "meta.json"
    synthetic = meta_p.exists() and json.loads(meta_p.read_text()).get("synthetic", False)
    if synthetic:
        log.warning("Dataset sintético: no se actualiza el README")
    elif not args.no_readme and not args.sample_frac:
        update_readme(C.METRICS_DIR / "summary.md")


if __name__ == "__main__":
    main()
