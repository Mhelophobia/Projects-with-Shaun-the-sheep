"""
V1 — data file → cleaned dataset + summary statistics + charts (command line)
==============================================================================
Usage:
    python3 src/v1_clean_analyze.py <input_file> [--out <output_dir>]
                                    [--sheet <sheet_name>]

Accepts CSV, TXT (auto-detected delimiter), and Excel (.xlsx/.xls).

The data work lives in src/core.py (shared with the web app);
this script is just the command-line wrapper.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core


def main():
    ap = argparse.ArgumentParser(
        description="V1: clean a data file and produce summary stats + charts.")
    ap.add_argument("input_file",
                    help="path to the input file (csv, txt, xlsx, xls)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: "
                         "data/output/<file-stem>)")
    ap.add_argument("--sheet", default=None,
                    help="Excel sheet name (default: first sheet)")
    ap.add_argument("--remove-duplicates", action="store_true",
                    help="drop exact duplicate rows")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="do not drop exact duplicate rows")
    ap.add_argument("--no-numeric-conversion", "--no-auto-numeric", action="store_true",
                    help="do not auto-convert numeric-looking text columns")
    ap.add_argument("--missing-tokens", action="store_true",
                    help="standardize common missing-value markers")
    ap.add_argument("--no-missing-standardization", action="store_true",
                    help="do not standardize NA/N/A/-/? markers to missing")
    ap.add_argument("--y", default=None, metavar="YCOL",
                    help="V4: dependent variable for OLS regression")
    ap.add_argument("--x", default=None, nargs="+", metavar="XCOL",
                    help="V4: independent variable(s) for OLS regression")
    args = ap.parse_args()

    src = Path(args.input_file)
    if not src.exists():
        sys.exit(f"Input file not found: {src}")

    out_dir = (Path(args.out) if args.out
               else Path("data/output") / src.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_opts = {
        "drop_dupes": args.remove_duplicates and not args.keep_duplicates,
        "convert_numerics": not args.no_numeric_conversion,
        "fix_missing_tokens": args.missing_tokens and not args.no_missing_standardization,
    }
    result = core.analyze(src, name=src.name, sheet=args.sheet,
                          clean_opts=clean_opts)
    if args.sheet:
        print(f"Using Excel sheet: {args.sheet}")
    for line in result["clean_lines"]:
        print(line)

    df = result["df"]
    df.to_csv(out_dir / "cleaned.csv", index=False)
    (out_dir / "cleaning_log.txt").write_text(
        "\n".join(result["clean_lines"]), encoding="utf-8")
    result["desc"].round(4).to_csv(out_dir / "summary_statistics.csv")
    result["missing"].to_csv(out_dir / "missing_values.csv")
    result["top_cats"].to_csv(out_dir / "top_categories.csv", index=False)
    (out_dir / "report.md").write_text(result["report"], encoding="utf-8")
    chart_paths = core.save_figures(result["figs"], out_dir / "charts")

    print(f"\nDone. Outputs in {out_dir}/")
    print(" - cleaned.csv, cleaning_log.txt, summary_statistics.csv,")
    print("   missing_values.csv, top_categories.csv, report.md, charts/")

    if args.y or args.x:
        if not (args.y and args.x):
            sys.exit("Regression needs both --y YCOL and --x XCOL [XCOL ...].")
        try:
            reg = core.run_regression(df, args.y, args.x)
        except (ValueError, RuntimeError) as e:
            sys.exit(f"Regression failed: {e}")
        reg_dir = out_dir / "regression"
        reg_dir.mkdir(exist_ok=True)
        reg["coef"].round(4).to_csv(reg_dir / "coefficients.csv")
        reg["vif"].round(3).to_csv(reg_dir / "vif.csv", index=False)
        reg["corr"].round(4).to_csv(reg_dir / "correlation.csv")
        d = reg["diagnostics"]
        (reg_dir / "diagnostics.txt").write_text(
            f"OLS: {args.y} ~ {' + '.join(args.x)}\n"
            f"n = {reg['n_obs']}\n"
            f"R-squared = {reg['r2']:.4f} | adj. R-squared = {reg['adj_r2']:.4f}\n"
            f"F = {reg['f_stat']:.2f} (p = {reg['f_pvalue']:.4g})\n"
            f"Breusch-Pagan p = {d['breusch_pagan_p']:.4g} "
            f"(heteroskedasticity if < 0.05)\n"
            f"Jarque-Bera p = {d['jarque_bera_p']:.4g} "
            f"(non-normal residuals if < 0.05)\n",
            encoding="utf-8")
        core.save_figures(reg["figs"], reg_dir / "charts")
        print(f" - regression/: coefficients.csv, vif.csv, correlation.csv,")
        print(f"   diagnostics.txt, charts/ (R² = {reg['r2']:.3f})")


if __name__ == "__main__":
    main()
