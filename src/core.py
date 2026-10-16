"""
core.py — shared engine for the auto-research project.

Used by both the V1 command-line script and the Streamlit web app.
All data work lives here; the CLI / web app are thin wrappers.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed; we save/show PNGs directly
import matplotlib.pyplot as plt

MISSING_TOKENS = ["", " ", "NA", "N/A", "n/a", "na", "NULL", "null",
                  "None", "none", "-", "?", "#N/A", "#n/a"]

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".xlsx", ".xls"}
EXCEL_EXTENSIONS = {".xlsx", ".xls"}


def _ext_of(filename):
    return Path(filename).suffix.lower()


def get_sheet_names(source):
    """List sheet names in an Excel file (path or file-like)."""
    return pd.ExcelFile(source).sheet_names


def _read_txt(source):
    """Read a .txt as delimited text, auto-detecting the separator."""
    def _try(**kw):
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, engine="python", dtype=str, keep_default_na=False, **kw)

    try:  # auto-detect: comma, tab, semicolon, pipe, ...
        return _try(sep=None, encoding="utf-8")
    except Exception:
        pass
    try:
        return _try(sep=None, encoding="latin-1")
    except Exception:
        pass
    # last resort: whitespace-delimited
    return _try(sep=r"\s+", encoding="utf-8")


def load_table(source, filename=None, sheet=None):
    """Load a CSV, TXT, or Excel file into a DataFrame.

    source: path or file-like object. filename: original file name
    (needed when source is file-like). sheet: Excel sheet name or
    index (defaults to the first sheet).
    """
    name = filename or (str(source) if isinstance(source, (str, Path))
                        else getattr(source, "name", ""))
    ext = _ext_of(name)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    if ext == ".csv":
        return load_csv(source)
    if ext == ".txt":
        return _read_txt(source)
    # Excel
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_excel(source, sheet_name=0 if sheet is None else sheet, dtype=object, keep_default_na=False)


def load_csv(source):
    """Load a CSV from a path or an uploaded file-like object."""
    try:
        return pd.read_csv(source, encoding="utf-8", dtype=str, keep_default_na=False)
    except UnicodeDecodeError:
        if isinstance(source, (str, Path)):
            return pd.read_csv(source, encoding="latin-1", dtype=str, keep_default_na=False)
        source.seek(0)
        return pd.read_csv(source, encoding="latin-1", dtype=str, keep_default_na=False)


def _log(lines, msg):
    lines.append(msg)
    return msg


def _is_text_dtype(dtype):
    """True for object/string columns (works on pandas 2.x and 3.x)."""
    return (pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype))


def clean_column_names(df, lines):
    original = list(df.columns)
    normalized = [re.sub(r"_+", "_", str(c).strip().replace(" ", "_"))
                  for c in original]
    reserved, used, cleaned = set(normalized), set(), []
    for index, original_name in enumerate(normalized, start=1):
        base = original_name or f"column_{index}"
        candidate, suffix = base, 2
        # A suffix must not take the name of a later original column.
        while candidate in used or (candidate != original_name and candidate in reserved):
            candidate = f"{base}_{suffix}"
            suffix += 1
        cleaned.append(candidate)
        used.add(candidate)
    df.columns = cleaned
    renamed = {o: n for o, n in zip(original, cleaned) if str(o) != n}
    if renamed:
        _log(lines, f"Renamed columns for tidiness: {renamed}")
    return df


def standardize_missing(df, lines):
    before = int(df.isna().sum().sum())
    df = df.replace(MISSING_TOKENS, np.nan)
    df = df.map(lambda value: value.strip() if isinstance(value, str) else value)
    df = df.replace("", np.nan)
    after = int(df.isna().sum().sum())
    if after > before:
        _log(lines, f"Standardized {after - before} missing-value tokens "
                    f"(e.g. 'NA', 'N/A', '-', '?') to NaN.")
    return df


def drop_empty(df, lines):
    r0, c0 = df.shape
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    r1, c1 = df.shape
    if (r0, c0) != (r1, c1):
        _log(lines, f"Dropped {r0 - r1} all-empty rows and "
                    f"{c0 - c1} all-empty columns.")
    return df


def drop_duplicates(df, lines):
    n = int(df.duplicated().sum())
    if n:
        df = df.drop_duplicates().reset_index(drop=True)
        _log(lines, f"Dropped {n} exact duplicate rows.")
    return df


def convert_numeric(df, lines):
    converted = []
    for col in df.columns:
        if _is_text_dtype(df[col].dtype):
            values = df[col].dropna()
            strings = values.map(str)
            identifier = bool(re.search(r"(^|_)(id|identifier|serial|code|zip|postal)($|_)",
                                        str(col).lower()))
            protected = strings.map(lambda v: bool(re.fullmatch(r"[+-]?0\d+", v))
                                    or len(re.sub(r"\D", "", v)) > 15).any()
            if identifier or protected:
                _log(lines, f"Kept '{col}' as text to preserve identifiers or precision.")
                continue
            coerced = pd.to_numeric(df[col], errors="coerce")
            if len(values) and coerced.notna().sum() == len(values) and np.isfinite(coerced.dropna()).all():
                df[col] = coerced
                converted.append(col)
            elif len(values):
                _log(lines, f"Kept '{col}' unchanged because some values are not numeric.")
    if converted:
        _log(lines, f"Auto-converted text columns to numeric: {converted}")
    return df


def clean(df, drop_dupes=False, convert_numerics=True,
          fix_missing_tokens=False, *, remove_duplicates=None,
          auto_numeric=None, use_missing_tokens=None):
    """Clean a DataFrame. Returns (cleaned_df, log_lines).

    The three flags are the user-facing "cleaning options":
    drop_dupes         — drop exact duplicate rows
    convert_numerics   — auto-convert numeric-looking text columns
    fix_missing_tokens — standardize NA/N/A/-/? … markers to NaN
    """
    if remove_duplicates is not None:
        drop_dupes = remove_duplicates
    if auto_numeric is not None:
        convert_numerics = auto_numeric
    if use_missing_tokens is not None:
        fix_missing_tokens = use_missing_tokens
    df = df.copy()
    lines = [f"Starting shape: {df.shape[0]} rows × {df.shape[1]} columns"]
    df = clean_column_names(df, lines)
    df = df.map(lambda value: value.strip() if isinstance(value, str) else value)
    df = df.replace("", np.nan)
    if fix_missing_tokens:
        df = standardize_missing(df, lines)
    else:
        _log(lines, "Skipped missing-marker standardization (option off).")
    df = drop_empty(df, lines)
    if drop_dupes:
        df = drop_duplicates(df, lines)
    else:
        _log(lines, "Skipped duplicate removal (option off).")
        _log(lines, f"Retained {int(df.duplicated().sum())} duplicate rows.")
    if convert_numerics:
        df = convert_numeric(df, lines)
    else:
        _log(lines, "Skipped numeric auto-conversion (option off).")
    lines.append(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df, lines


def summarize(df):
    """Returns (desc, missing, top_cats, numeric_cols, cat_cols)."""
    numeric = df.select_dtypes(include=np.number)
    desc = numeric.describe().T if not numeric.empty else pd.DataFrame()
    if not numeric.empty:
        desc["skew"] = numeric.skew()
        desc["kurtosis"] = numeric.kurtosis()
        desc["missing"] = numeric.isna().sum()
        desc["missing_%"] = (numeric.isna().mean() * 100).round(2)
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
    }).sort_values("missing_%", ascending=False)
    cat_cols = [c for c in df.columns
                if _is_text_dtype(df[c].dtype)
                or isinstance(df[c].dtype, pd.CategoricalDtype)]
    top_cats = []
    for col in cat_cols:
        for val, cnt in df[col].value_counts(dropna=False).head(5).items():
            top_cats.append({"column": col, "value": val, "count": int(cnt)})
    top_cats = pd.DataFrame(top_cats)
    return desc, missing, top_cats, numeric.columns.tolist(), cat_cols


def make_figures(df, numeric_cols, cat_cols):
    """Build matplotlib figures. Returns list of (title, figure)."""
    figs = []

    if numeric_cols:
        cols = numeric_cols[:12]
        n = len(cols)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, col in zip(axes, cols):
            ax.hist(df[col].dropna(), bins=20, edgecolor="white")
            ax.set_title(col)
            ax.set_ylabel("count")
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle("Distributions of numeric columns", fontsize=13)
        fig.tight_layout()
        figs.append(("Histograms", fig))

    cat_plot = [c for c in cat_cols if df[c].nunique() <= 20][:9]
    if cat_plot:
        n = len(cat_plot)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 3.5 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, col in zip(axes, cat_plot):
            vc = df[col].value_counts(dropna=False).head(10)
            labels = ["(missing)" if pd.isna(v) else str(v)
                      for v in vc.index]
            ax.barh(labels, vc.values)
            ax.set_title(col)
            ax.invert_yaxis()
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle("Top values per categorical column", fontsize=13)
        fig.tight_layout()
        figs.append(("Categorical bars", fig))

    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if not miss.empty:
        fig, ax = plt.subplots(figsize=(max(6, len(miss) * 0.6), 4))
        ax.bar(miss.index.astype(str), (miss * 100).round(1))
        ax.set_ylabel("missing %")
        ax.set_title("Missing values by column")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        figs.append(("Missing values", fig))

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(
            figsize=(max(6, len(numeric_cols) * 0.8),
                     max(5, len(numeric_cols) * 0.7)))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="correlation")
        ax.set_title("Correlation matrix (numeric columns)")
        fig.tight_layout()
        figs.append(("Correlation", fig))

    return figs


def save_figures(figs, chart_dir):
    """Save (title, fig) pairs as PNGs. Returns list of Paths."""
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for title, fig in figs:
        fname = title.lower().replace(" ", "_") + ".png"
        p = chart_dir / fname
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def build_report(name, df, clean_lines, desc, missing,
                 numeric_cols, cat_cols, chart_names):
    """Returns the Markdown report text."""
    lines = [f"# Data report: {name}", ""]
    lines.append("## Cleaning log")
    lines += [f"- {l}" for l in clean_lines]
    lines += ["", "## Dataset overview",
              f"- Rows: {df.shape[0]}, Columns: {df.shape[1]}",
              f"- Numeric columns ({len(numeric_cols)}): "
              f"{', '.join(numeric_cols) if numeric_cols else 'none'}",
              f"- Categorical columns ({len(cat_cols)}): "
              f"{', '.join(cat_cols) if cat_cols else 'none'}", ""]
    lines.append("## Missing values (worst 10)")
    lines.append("```")
    lines.append(missing.head(10).to_string() if not missing.empty
                 else "No columns.")
    lines.append("```")
    lines += ["", "## Numeric summary"]
    lines.append("```")
    lines.append(desc.round(3).to_string() if not desc.empty
                 else "No numeric columns.")
    lines.append("```")
    if chart_names:
        lines += ["", "## Charts"]
        lines += [f"- {c}" for c in chart_names]
    return "\n".join(lines)


def analyze(source, name=None, sheet=None, clean_opts=None):
    """Full V1 pipeline. Returns dict with everything the UIs need.

    clean_opts: dict with any of drop_dupes, convert_numerics,
    fix_missing_tokens. Numeric conversion defaults to True; duplicate removal
    and missing-token standardization are opt-in. The UI supplies explicit options.
    """
    raw = load_table(source, filename=name, sheet=sheet)
    opts = {"drop_dupes": False, "convert_numerics": True,
            "fix_missing_tokens": False}
    if clean_opts:
        opts.update(clean_opts)
    df, clean_lines = clean(raw, **opts)
    desc, missing, top_cats, numeric_cols, cat_cols = summarize(df)
    figs = make_figures(df, numeric_cols, cat_cols)
    report = build_report(name or "dataset", df, clean_lines, desc,
                          missing, numeric_cols, cat_cols,
                          [t for t, _ in figs])
    return {
        "raw_shape": raw.shape,
        "clean_opts": opts.copy(),
        "df": df,
        "clean_lines": clean_lines,
        "desc": desc,
        "missing": missing,
        "top_cats": top_cats,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "figs": figs,
        "report": report,
    }


def run_regression(df, y_col, x_cols):
    """V4: OLS regression of y_col on x_cols + VIF + correlation + diagnostics.

    Returns a dict with coef table, fit stats, VIF table, correlation
    matrix, diagnostic test results, and figures. Raises ValueError on bad
    input, RuntimeError if statsmodels is missing.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor)
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.stats.stattools import jarque_bera
        from statsmodels.graphics.gofplots import qqplot
    except ImportError:
        raise RuntimeError(
            "Regression needs the 'statsmodels' package. "
            "Install it with: pip install statsmodels")

    if y_col not in df.columns:
        raise ValueError(f"Y variable '{y_col}' not found.")
    if not x_cols:
        raise ValueError("Pick at least one X variable.")
    bad = [c for c in [y_col] + list(x_cols) if c not in df.columns]
    if bad:
        raise ValueError(f"Column(s) not found: {', '.join(bad)}")
    if y_col in x_cols:
        raise ValueError("Y cannot also be one of the X variables.")
    non_num = [c for c in [y_col] + list(x_cols)
               if not pd.api.types.is_numeric_dtype(df[c])]
    if non_num:
        raise ValueError(
            f"Regression needs numeric columns; not numeric: "
            f"{', '.join(non_num)}")

    data = df[[y_col] + list(x_cols)].dropna()
    n, k = data.shape[0], len(x_cols)
    if n < k + 2:
        raise ValueError(
            f"Not enough complete rows ({n}) for {k} X variables.")

    X = sm.add_constant(data[list(x_cols)], has_constant="add")
    y = data[y_col]
    model = sm.OLS(y, X).fit()

    coef = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t": model.tvalues,
        "p_value": model.pvalues,
    })

    vif = pd.DataFrame({
        "variable": list(x_cols),
        "VIF": [variance_inflation_factor(X.values, i + 1)
                for i in range(k)],
    })

    corr = data[[y_col] + list(x_cols)].corr()

    # Diagnostics: heteroskedasticity + residual normality tests
    bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
    jb_stat, jb_p, _, _ = jarque_bera(model.resid)
    diagnostics = {
        "n_obs": int(n),
        "breusch_pagan_stat": float(bp_stat),
        "breusch_pagan_p": float(bp_p),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_p": float(jb_p),
    }

    figs = []
    fitted, resid = model.fittedvalues, model.resid

    fig, ax = plt.subplots()
    ax.scatter(fitted, resid, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    figs.append(("Residuals vs Fitted", fig))

    fig = qqplot(resid, line="s")
    fig.suptitle("Q-Q plot of residuals")
    figs.append(("Q-Q plot of residuals", fig))

    fig, ax = plt.subplots()
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45,
                  ha="right")
    ax.set_yticks(range(len(corr.columns)), corr.columns)
    ax.set_title("Correlation matrix")
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center",
                    va="center", fontsize=8,
                    color="white" if abs(corr.values[i, j]) > 0.5
                    else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    figs.append(("Correlation heatmap", fig))

    return {
        "y": y_col,
        "x": list(x_cols),
        "n_obs": int(n),
        "coef": coef,
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "f_stat": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "vif": vif,
        "corr": corr,
        "diagnostics": diagnostics,
        "figs": figs,
    }
