"""
V7 — one-click econometrics/research assistant
===============================================
The culmination of the roadmap: from a raw data file to a complete
research deliverable in (almost) one click.

Pieces:
- suggest_design(): the AI reads your columns and proposes a regression
  (Y and X's), with reasons. You confirm or tweak — the researcher
  still decides.
- smart_guards(): automatic sanity checks (ID-like columns, tiny
  samples, multicollinearity, dead predictors).
- run_everything(): chains the whole pipeline — clean → describe →
  charts → regression → diagnostics → assembled full report.
- build_full_report(): one Markdown document with everything.

Command line:
    python src/v7_assistant.py data/input/housing.csv --y price --x sqft age
    python src/v7_assistant.py data/input/housing.csv --suggest   # AI picks Y/X
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import core
import ai_chat

DESIGN_PROMPT = """You are a research design assistant for an econometrics \
automation app. A dataset has been cleaned; its columns are listed \
below (name, dtype, % missing).

Propose ONE sensible OLS regression: a dependent variable Y and 2-5 \
explanatory X variables.

Rules:
- Y must be a substantive outcome (income, price, score, sales...). \
NEVER pick an ID, serial number, or row counter as Y.
- X's must be plausible explanatory variables. NEVER pick IDs, serial \
numbers, or constants.
- Prefer numeric columns.

Reply with ONLY a JSON object, no other text:
{{"y": "column_name", "x": ["col_a", "col_b"], \
"why_y": "one sentence", "why_x": "one sentence"}}

Columns:
{columns}"""

FULL_INTERPRET_TASK = (
    "You are writing the interpretation section of a research report. "
    "Using the full results below (data overview, cleaning, regression "
    "output, diagnostics, and automatic checks), write a clear, "
    "thesis-ready interpretation: what was done, what each key result "
    "means in plain words, which findings are statistically significant, "
    "what the diagnostics say, and what caveats a researcher must "
    "report. Be concrete — quote the actual numbers. End with 3-5 "
    "bullet takeaways for the thesis."
)


def _parse_design(text):
    """Extract the JSON design from the AI's reply (tolerates fences)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("The AI did not return a usable design.")
    d = json.loads(m.group(0))
    return {
        "y": d["y"],
        "x": list(d["x"]),
        "why_y": d.get("why_y", ""),
        "why_x": d.get("why_x", ""),
    }


def suggest_design(result, api_key, base_url=None, model=None):
    """Ask the AI to propose Y and X's. Returns validated design dict."""
    overview = ai_chat.build_column_overview(result)
    messages = [
        {"role": "system",
         "content": "You propose regression designs. Reply with only JSON."},
        {"role": "user",
         "content": DESIGN_PROMPT.format(columns=overview)},
    ]
    text = ai_chat.chat_reply(
        api_key, ai_chat.trim_messages(messages),
        base_url=base_url or ai_chat.DEFAULT_BASE_URL,
        model=model or ai_chat.DEFAULT_MODEL)
    design = _parse_design(text)
    df = result["df"]
    cols = set(df.columns)
    if design["y"] not in cols:
        raise ValueError(f"AI suggested an unknown Y: {design['y']}")
    numeric = set(result["numeric_cols"])
    design["x"] = [c for c in design["x"]
                   if c in cols and c != design["y"] and c in numeric]
    if not design["x"]:
        raise ValueError("AI suggested no usable X variables.")
    return design


def smart_guards(result, reg):
    """Automatic sanity checks on the regression. Returns [warnings]."""
    warnings = []
    df = result["df"]
    n, k = reg["n_obs"], len(reg["x"])
    for c in reg["x"]:
        s = df[c]
        # Integer, unique in every row, and monotonic → almost certainly
        # a serial/ID column, not an explanatory variable. (Floats are
        # exempt: continuous measurements are often unique per row.)
        if (pd.api.types.is_integer_dtype(s) and s.nunique() == len(df)
                and (s.is_monotonic_increasing
                     or s.is_monotonic_decreasing)):
            warnings.append(
                f"'{c}' is unique in every row — it looks like an "
                f"ID/serial number, not an explanatory variable. "
                f"Consider dropping it.")
    if n < 10 * k:
        warnings.append(
            f"Only {n} observations for {k} predictors (rule of thumb: "
            f"at least {10 * k}). Estimates may be unstable.")
    for _, r in reg["vif"].iterrows():
        if r["VIF"] > 10:
            warnings.append(
                f"VIF for '{r['variable']}' is {r['VIF']:.1f} (> 10) — "
                f"strong multicollinearity.")
    for c in reg["x"]:
        if df[c].std() == 0:
            warnings.append(f"'{c}' is constant — it cannot explain anything.")
    if reg["r2"] < 0.10:
        warnings.append(
            f"R² is only {reg['r2']:.3f} — the X's explain little of "
            f"'{reg['y']}'.")
    if reg["f_pvalue"] >= 0.05:
        warnings.append(
            f"The overall F-test is not significant "
            f"(p = {reg['f_pvalue']:.4g}) — the model adds no explanatory "
            f"power beyond the mean of '{reg['y']}'.")
    return warnings


def build_full_report(name, result, reg, guards):
    """Assemble the one deliverable: a complete Markdown research report."""
    df = result["df"]
    d = reg["diagnostics"]
    L = [f"# Research report: {name}", ""]
    L += ["## 1. Data overview", "",
          f"- Rows after cleaning: **{df.shape[0]}**, "
          f"columns: **{df.shape[1]}**",
          f"- Numeric columns: {', '.join(result['numeric_cols']) or 'none'}",
          f"- Categorical columns: {', '.join(result['cat_cols']) or 'none'}",
          ""]
    L += ["## 2. Cleaning log", ""]
    L += [f"- {line}" for line in result["clean_lines"]] + [""]
    L += ["## 3. Descriptive snapshot", "",
          "Missing values per column:", "```",
          result["missing"].round(2).to_string(), "```", ""]
    if not result["desc"].empty:
        L += ["Numeric summary:", "```",
              result["desc"].round(3).to_string(), "```", ""]
    L += ["## 4. Regression", "",
          f"**OLS:** `{reg['y']} ~ {' + '.join(reg['x'])}`",
          f"- Observations: {reg['n_obs']}",
          f"- R² = **{reg['r2']:.4f}**, adj. R² = {reg['adj_r2']:.4f}",
          f"- F = {reg['f_stat']:.2f} (p = {reg['f_pvalue']:.4g})", "",
          "Coefficients:", "```",
          reg["coef"].round(4).to_string(), "```", "",
          "VIF:", "```",
          reg["vif"].round(3).to_string(index=False), "```", ""]
    L += ["## 5. Diagnostics", "",
          f"- Breusch-Pagan p = {d['breusch_pagan_p']:.4g} "
          f"({'possible heteroskedasticity'
             if d['breusch_pagan_p'] < 0.05
             else 'no sign of heteroskedasticity'})",
          f"- Jarque-Bera p = {d['jarque_bera_p']:.4g} "
          f"({'residuals look non-normal'
             if d['jarque_bera_p'] < 0.05
             else 'residuals look normal'})",
          ""]
    L += ["## 6. Automatic checks", ""]
    if guards:
        L += [f"- ⚠️ {g}" for g in guards]
    else:
        L += ["- ✅ No issues flagged."]
    L += ["", "## 7. AI interpretation", "",
          "_(Generate from the website's 🚀 One-Click tab, or interpret "
          "each section with the ✨ buttons.)_", ""]
    L += ["---",
          f"_Generated by the Auto-Research Assistant (V7)._"]
    return "\n".join(L)


def _save_bundle(out_dir, result, reg, report, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = result["df"]
    df.to_csv(out_dir / "cleaned.csv", index=False)
    (out_dir / "cleaning_log.txt").write_text(
        "\n".join(result["clean_lines"]), encoding="utf-8")
    if not result["desc"].empty:
        result["desc"].round(4).to_csv(out_dir / "summary_statistics.csv")
    result["missing"].to_csv(out_dir / "missing_values.csv")
    (out_dir / "report.md").write_text(result["report"], encoding="utf-8")
    core.save_figures(result["figs"], out_dir / "charts")
    reg_dir = out_dir / "regression"
    reg_dir.mkdir(exist_ok=True)
    reg["coef"].round(4).to_csv(reg_dir / "coefficients.csv")
    reg["vif"].round(3).to_csv(reg_dir / "vif.csv", index=False)
    reg["corr"].round(4).to_csv(reg_dir / "correlation.csv")
    core.save_figures(reg["figs"], reg_dir / "charts")
    (out_dir / "FULL_REPORT.md").write_text(report, encoding="utf-8")
    return out_dir


def run_everything(source=None, y_col=None, x_cols=None, *, result=None,
                   name="dataset", clean_opts=None, out_dir=None):
    """One call: clean → describe → charts → regress → report.

    Pass either `source` (a file path) or a pre-analyzed `result` dict.
    Returns {"result", "reg", "guards", "report", "out_dir"}.
    """
    if result is None:
        if source is None:
            raise ValueError("run_everything needs a source file or result.")
        # core detects the file type from the name: fall back to the
        # source's own filename when `name` has no extension.
        safe_name = (name if Path(str(name)).suffix
                     else Path(str(source)).name)
        result = core.analyze(source, name=safe_name, clean_opts=clean_opts)
    reg = core.run_regression(result["df"], y_col, x_cols)
    guards = smart_guards(result, reg)
    report = build_full_report(name, result, reg, guards)
    saved = None
    if out_dir is not None or source is not None:
        stem = Path(str(source)).stem if source else "dataset"
        out = Path(out_dir) if out_dir else Path("data/output") / f"{stem}_full"
        saved = _save_bundle(out, result, reg, report, name)
    return {"result": result, "reg": reg, "guards": guards,
            "report": report, "out_dir": saved}


def main():
    ap = argparse.ArgumentParser(
        description="V7: one-click econometrics/research assistant.")
    ap.add_argument("input_file", help="data file (csv, txt, xlsx, xls)")
    ap.add_argument("--y", default=None, help="dependent variable")
    ap.add_argument("--x", default=None, nargs="+",
                    help="independent variables")
    ap.add_argument("--suggest", action="store_true",
                    help="let the AI propose Y and X (needs GROQ_API_KEY)")
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args()

    src = Path(args.input_file)
    if not src.exists():
        sys.exit(f"Input file not found: {src}")

    y_col, x_cols = args.y, args.x
    if args.suggest:
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            sys.exit("Set the GROQ_API_KEY environment variable first.")
        print("Analyzing, then asking the AI for a research design…")
        result = core.analyze(src, name=src.name)
        try:
            design = suggest_design(result, key)
        except (RuntimeError, ValueError) as e:
            sys.exit(f"Could not get an AI design: {e}")
        y_col, x_cols = design["y"], design["x"]
        print(f"AI design: {y_col} ~ {' + '.join(x_cols)}")
        print(f"  why Y: {design['why_y']}")
        print(f"  why X: {design['why_x']}")
    if not (y_col and x_cols):
        sys.exit("Give --y YCOL and --x XCOL [XCOL ...], or use --suggest.")

    try:
        bundle = run_everything(src, y_col, x_cols, name=src.name,
                                out_dir=args.out)
    except (ValueError, RuntimeError) as e:
        sys.exit(f"V7 failed: {e}")

    print(f"\nR² = {bundle['reg']['r2']:.3f}, "
          f"n = {bundle['reg']['n_obs']}")
    for g in bundle["guards"]:
        print(f"  ⚠️ {g}")
    print(f"\nDone. Full bundle in {bundle['out_dir']}/")
    print("  → FULL_REPORT.md is the one deliverable.")


if __name__ == "__main__":
    main()
