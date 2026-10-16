"""
ai_chat.py — AI Assistant for the Auto-Research website.

Builds a compact "data brief" from an analysis result and sends it
to an OpenAI-compatible chat API (Groq by default) so the
assistant answers questions about the user's own dataset.

The web app reads its owner-provided API key from server-side secrets.
Credentials are never rendered in widgets or included in AI prompts.
"""

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a friendly data-analysis assistant inside a \
research automation app. The user has just cleaned and summarized a \
dataset; a briefing about it follows. Answer their questions about \
THIS dataset: its columns, statistics, data quality, and what the \
numbers mean. You can also draft short manuscript-ready \
interpretations when asked.

Rules:
- Be concise and concrete; quote actual numbers from the briefing.
- If a question can't be answered from the briefing, say so instead \
of guessing.
- Never invent columns, values, or statistics that aren't in the briefing.
"""


EXPLAIN_TASKS = {
    "cleaning": (
        "Explain the cleaning results below for a researcher: what was "
        "done to the dataset, why each step matters for analysis, and "
        "what data-quality concerns remain. Be concrete — quote the "
        "actual numbers."
    ),
    "statistics": (
        "Interpret the summary statistics below for a researcher writing "
        "a thesis: describe the key patterns, notable values, skew, "
        "spread, and missing-data concerns. Be concrete — quote the "
        "actual numbers, and flag anything that looks odd or worth "
        "investigating further."
    ),
    "charts": (
        "The researcher generated charts from this data: distributions "
        "(histograms), categorical bar charts, a missing-values chart, "
        "and a correlation heatmap. Using the numbers below, describe "
        "what each chart shows and the key takeaways a researcher "
        "should report. Be concrete — quote the actual numbers."
    ),
    "regression": (
        "Interpret the OLS regression results below for a researcher "
        "writing a thesis: explain what each coefficient means in plain "
        "words (the expected change in Y per unit of X, holding other "
        "variables constant), which predictors are statistically "
        "significant, how well the model fits (R-squared), whether "
        "multicollinearity is a concern (VIF), and what the diagnostic "
        "tests say about heteroskedasticity and residual normality. "
        "Be concrete — quote the actual numbers, and flag anything "
        "that looks odd or worth investigating further."
    ),
}


def build_section_brief(result, name, section, reg=None):
    """Focused briefing for one results tab.

    reg: the run_regression() dict, required when section='regression'.
    """
    df = result["df"]
    if section == "cleaning":
        lines = [
            f"Dataset: {name}",
            f"Raw shape: {result['raw_shape'][0]} rows x "
            f"{result['raw_shape'][1]} columns",
            f"Cleaned shape: {df.shape[0]} rows x {df.shape[1]} columns",
            "", "Cleaning log:",
        ]
        lines += [f"- {l}" for l in result["clean_lines"]]
        return "\n".join(lines)
    if section == "statistics":
        lines = [f"Dataset: {name} "
                 f"({df.shape[0]} rows x {df.shape[1]} columns)",
                 "", "Missing values per column:"]
        lines.append(_cap_table(result["missing"].round(2), 20, "columns"))
        if not result["desc"].empty:
            lines += ["", "Numeric summary:"]
            lines.append(_cap_table(result["desc"].round(3), 25, "columns"))
        if not result["top_cats"].empty:
            lines += ["", "Top categorical values:"]
            for _, r in result["top_cats"].iterrows():
                lines.append(f"- {r['column']}: '{r['value']}' "
                             f"(n={r['count']})")
        return "\n".join(lines)
    if section == "charts":
        lines = [f"Dataset: {name} "
                 f"({df.shape[0]} rows x {df.shape[1]} columns)",
                 "Charts generated: "
                 + ", ".join(t for t, _ in result["figs"])]
        num = result["numeric_cols"]
        if len(num) >= 2:
            lines += ["", "Correlation matrix:"]
            lines.append(df[num].corr(numeric_only=True).round(2)
                         .to_string())
        desc = result["desc"]
        if not desc.empty and "skew" in desc.columns:
            lines += ["", "Skew per numeric column:"]
            for col in desc.index:
                lines.append(f"- {col}: skew={desc.loc[col, 'skew']:.2f}")
        return "\n".join(lines)
    if section == "regression":
        if reg is None:
            raise ValueError("Regression brief needs the reg result.")
        d = reg["diagnostics"]
        lines = [
            f"Dataset: {name}",
            f"OLS regression: {reg['y']} ~ {' + '.join(reg['x'])}",
            f"n = {reg['n_obs']} complete observations",
            f"R-squared = {reg['r2']:.4f}, "
            f"adj. R-squared = {reg['adj_r2']:.4f}",
            f"F-statistic = {reg['f_stat']:.2f} (p = {reg['f_pvalue']:.4g})",
            "", "Coefficients:",
            _cap_table(reg["coef"].round(4), 20, "variables"),
            "", "VIF (multicollinearity; worry if > 5-10):",
            _cap_table(reg["vif"].round(3), 20, "variables"),
            "", "Diagnostics:",
            f"- Breusch-Pagan p = {d['breusch_pagan_p']:.4g} "
            f"(< 0.05 suggests heteroskedasticity)",
            f"- Jarque-Bera p = {d['jarque_bera_p']:.4g} "
            f"(< 0.05 suggests non-normal residuals)",
        ]
        return "\n".join(lines)
    raise ValueError(f"Unknown section: {section}")


def estimate_tokens(text):
    """Rough token estimate (~4 chars per token). Good enough for budgets."""
    return max(1, len(str(text)) // 4)


def trim_messages(messages, max_tokens=5000):
    """Drop the oldest conversation turns to fit a token budget.

    Always keeps the system message (index 0) and the newest turns.
    Conservative default (5000) stays under small models' per-minute
    token caps (e.g. 6000 TPM on Groq's free tier).
    """
    if len(messages) <= 1:
        return messages
    system, rest = messages[0], messages[1:]
    kept = []
    used = estimate_tokens(system.get("content", ""))
    for m in reversed(rest):
        t = estimate_tokens(m.get("content", ""))
        if used + t > max_tokens:
            break
        kept.append(m)
        used += t
    kept.reverse()
    return [system] + kept


def build_column_overview(result):
    """One line per column: name, dtype, missing %. Compact."""
    df = result["df"]
    lines = ["Columns:"]
    for col in df.columns:
        miss = df[col].isna().mean() * 100
        lines.append(f"- {col} ({df[col].dtype}, {miss:.1f}% missing)")
    return "\n".join(lines)


def _cap_table(df, max_rows, label):
    """Render a DataFrame as text, noting when rows were cut."""
    if len(df) > max_rows:
        return (df.head(max_rows).to_string()
                + f"\n… ({len(df) - max_rows} more {label} not shown)")
    return df.to_string()


def build_data_brief(result, name):
    """Compact text briefing about the analyzed dataset."""
    df = result["df"]
    lines = [f"Dataset: {name}",
             f"Shape after cleaning: {df.shape[0]} rows x {df.shape[1]} columns",
             "",
             "Columns (name | dtype | missing % | unique values):"]
    for col in df.columns:
        miss = df[col].isna().mean() * 100
        lines.append(
            f"- {col} | {df[col].dtype} | {miss:.1f}% missing | "
            f"{df[col].nunique()} unique")
    lines += ["", "Cleaning actions taken:"]
    lines += [f"- {l}" for l in result["clean_lines"]]
    desc = result["desc"]
    if not desc.empty:
        lines += ["", "Numeric summary (rounded):",
                  desc.round(2).to_string()]
    if not result["top_cats"].empty:
        lines += ["", "Top categorical values:"]
        for _, r in result["top_cats"].iterrows():
            lines.append(f"- {r['column']}: '{r['value']}' "
                         f"(n={r['count']})")
    return "\n".join(lines)


def _client(api_key, base_url):
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "The 'openai' package is not installed. Run: "
            "pip install openai")
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)


def _friendly_api_error(e):
    """Turn raw API exceptions into human-readable messages."""
    try:
        from openai import RateLimitError
        if isinstance(e, RateLimitError):
            return ("The AI service is temporarily at its usage limit. "
                    "Try again later or contact the app owner.")
    except ImportError:
        pass
    # Provider errors can include request details. Never echo them to testers.
    return "The AI service could not complete this request. Try again or contact the app owner."


def list_models(api_key, base_url=DEFAULT_BASE_URL, timeout=30):
    """Return sorted model IDs this key can access.

    Raises RuntimeError with a human-readable message on failure.
    """
    client = _client(api_key, base_url)
    try:
        models = client.models.list(timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Could not list models: {_friendly_api_error(e)}")
    ids = sorted(m.id for m in models.data)
    if not ids:
        raise RuntimeError("The API returned an empty model list.")
    return ids


def chat_reply(api_key, messages, base_url=DEFAULT_BASE_URL,
               model=DEFAULT_MODEL, timeout=120):
    """Send chat messages to the API. Returns the assistant's text.

    Raises RuntimeError with a human-readable message on failure.
    """
    client = _client(api_key, base_url)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout,
        )
    except Exception as e:
        raise RuntimeError(_friendly_api_error(e))
    try:
        return resp.choices[0].message.content or "(empty reply)"
    except Exception:
        raise RuntimeError("API returned an unexpected response.")
