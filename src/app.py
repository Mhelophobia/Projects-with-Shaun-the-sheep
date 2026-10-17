"""
Auto-Research Assistant — web app (V1 + V6 GUI, early)
======================================================
Run it with:
    streamlit run src/app.py
then open the address it prints (usually http://localhost:8501).

Upload a CSV → get a cleaned dataset, summary statistics,
charts, and downloadable results. All the data work lives in
src/core.py, shared with the command-line version.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import matplotlib.pyplot as plt
import io
from datetime import datetime
import core
import ai_chat
import v7_assistant
import variable_types
from chat_rendering import render_message

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "input" / \
    "sample_survey.csv"
HERO_IMAGE = Path(__file__).resolve().parent.parent / "assets" / "research-discovery-v1.png"

def _server_setting(name, default=""):
    """Read owner-managed configuration without putting it in browser widgets."""
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


def _current_api_key():
    return str(_server_setting("GROQ_API_KEY")).strip()


def _current_base_url():
    return str(_server_setting("AI_BASE_URL", ai_chat.DEFAULT_BASE_URL)).strip()


def _current_model():
    return str(_server_setting("AI_MODEL", ai_chat.DEFAULT_MODEL)).strip()


CHAT_SUGGESTIONS = {
    "cleaning": [
        "What was the biggest data quality issue here?",
        "Is there anything I should still worry about?",
    ],
    "statistics": [
        "Summarize the key findings in 3 bullets.",
        "Which numbers look odd or worth investigating?",
    ],
    "charts": [
        "What does the correlation heatmap show?",
        "Give me the takeaways I should report in my thesis.",
    ],
    "regression": [
        "Which predictors are significant, and what do they mean?",
        "Is multicollinearity or heteroskedasticity a concern here?",
    ],
    "report": ["Help me plan my research report.", "Summarize the findings for my report."],
    "downloads": ["Which file should I download?", "What is included in my report?"],
}

CHAT_SECTIONS = {
    "cleaning": "Overview",
    "statistics": "Statistics",
    "charts": "Charts",
    "regression": "Regression",
    "report": "Research report",
    "downloads": "Downloads",
}


def _chat_system(section, result, name, reg=None):
    if section == "report":
        bundle = st.session_state.get("v7_bundle")
        brief = (bundle["report"] if bundle else
                 "No research report has been built yet. Help the user choose an outcome and predictors.")
    elif section == "downloads":
        brief = (f"Dataset: {name}. Downloads offers a cleaned CSV and the analysis report in Markdown. "
                 "The Research report tab offers the full report after it is built.")
    elif section == "regression" and reg is None:
        brief = "No regression has been run yet. Help the user select an outcome and predictors; do not invent fitted results."
    else:
        brief = ai_chat.build_section_brief(result, name, section, reg=reg)
    return (ai_chat.SYSTEM_PROMPT
            + f"\n\nThe user is currently viewing the {section.upper()} "
              f"section of the results."
            + "\n\nSECTION BRIEF:\n"
            + brief
            + "\n\n"
            + ai_chat.build_column_overview(result))


def _section_history(section):
    """Per-section chat history: each tab keeps its own conversation."""
    histories = st.session_state.setdefault("chat_histories", {})
    return histories.setdefault(section, [])


def _send_chat_message(prompt, section, result, name, reg=None):
    """Append a user message, get the AI reply, append it.

    History is per-section: chats in Charts never leak into Statistics.
    """
    api_key = _current_api_key()
    hist = _section_history(section)
    hist.append({"role": "user", "content": prompt})
    messages = [{"role": "system",
                 "content": _chat_system(section, result, name, reg=reg)}]
    messages += hist
    messages = ai_chat.trim_messages(messages)  # fit context window
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("AI is thinking…", show_time=True):
            try:
                reply = ai_chat.chat_reply(
                    api_key, messages,
                    base_url=_current_base_url(),
                    model=_current_model())
            except RuntimeError as e:
                reply = f"⚠️ {e}"
        render_message(reply, key=f"{section}_reply_{len(hist)}")
    hist.append({"role": "assistant", "content": reply})


def _set_assistant_open(opened):
    st.session_state["assistant_open"] = opened


def _new_section_chat(section):
    st.session_state.setdefault("chat_histories", {})[section] = []
    st.session_state.pop(f"chat_{section}", None)
    generations = st.session_state.setdefault("chat_generations", {})
    generations[section] = generations.get(section, 0) + 1


def _chat_ui(section, result, name, reg=None):
    """A docked right-hand assistant with a separate conversation for each tab."""
    if not st.session_state.setdefault("assistant_open", False):
        return
    label = CHAT_SECTIONS[section]
    # Streamlit has one native left sidebar; scope the right-dock layout to our own keys.
    st.html(Path(__file__).with_name("assistant_panel.css"))
    with st.container(key="research_chat_panel"):
        with st.container(horizontal=True, vertical_alignment="center", key="assistant_header"):
            st.markdown(":blue[:material/auto_awesome:] **Research assistant**")
            st.space("stretch")
            st.button("New chat", icon=":material/add_comment:", type="tertiary",
                      key=f"new_chat_{section}", help=f"Start a new {label.lower()} conversation",
                      on_click=_new_section_chat, args=(section,))
            st.button("Close assistant", icon=":material/close:", type="tertiary",
                      key="close_assistant", on_click=_set_assistant_open, args=(False,))
        st.caption(f"{label} · Separate conversation for this section")
        api_key = _current_api_key()
        history = _section_history(section)
        generation = st.session_state.get("chat_generations", {}).get(section, 0)
        conversation = st.container(key=f"assistant_messages_{section}_{generation}",
                                    height=400, border=False, autoscroll=bool(history))
        with conversation:
            for message_index, msg in enumerate(history):
                with st.chat_message(msg["role"]):
                    render_message(msg["content"], key=f"{section}_history_{message_index}")
            if not history:
                st.markdown("### :blue[Hi there.]\n## How can I help?")
                st.caption(f"Let's take a closer look at your {label.lower()} results.")
                st.space("small")
                questions = CHAT_SUGGESTIONS.get(section, []) + ["Explain this section in simple terms."]
                for q in questions:
                    if st.button(q, key=f"sugg_{section}_{q[:20]}", width="stretch",
                                 icon=":material/auto_awesome:", disabled=not api_key):
                        _send_chat_message(q, section, result, name, reg=reg)
                        st.rerun()
            if not api_key:
                st.info("AI is not configured yet. Contact the app owner to enable chat.")
        if prompt := st.chat_input(f"Ask about {label.lower()}…",
                                   key=f"chat_{section}", submit_mode="disable",
                                   disabled=not api_key):
            with conversation:
                _send_chat_message(prompt, section, result, name, reg=reg)
            st.rerun()
        st.caption("AI can make mistakes. Check important findings against your data.")


def _explain_section(label, section, result, name, reg=None):
    """'Explain with AI' button for a results tab. The explanation
    persists in session state until a new dataset is analyzed."""
    key = f"explanation_{section}"
    if st.button(label, key=f"btn_{section}"):
        api_key = _current_api_key()
        if not api_key:
            st.warning("AI is not configured yet. Contact the app owner to enable it.")
        else:
            brief = ai_chat.build_section_brief(result, name, section,
                                                reg=reg)
            messages = [
                {"role": "system",
                 "content": ai_chat.SYSTEM_PROMPT
                 + "\n\nDATA BRIEF:\n" + brief},
                {"role": "user", "content": ai_chat.EXPLAIN_TASKS[section]},
            ]
            messages = ai_chat.trim_messages(messages)  # fit context window
            with st.spinner("The AI is reading your results…"):
                try:
                    reply = ai_chat.chat_reply(
                        api_key, messages,
                        base_url=_current_base_url(),
                        model=_current_model())
                except RuntimeError as e:
                    reply = f"⚠️ {e}"
            st.session_state[key] = reply
    if st.session_state.get(key):
        with st.expander("✨ AI interpretation", expanded=True):
            st.markdown(st.session_state[key])


def _remember_upload(source, name):
    """Keep a small in-memory list of recent uploads for quick re-runs."""
    try:
        data = source.getvalue() if hasattr(source, "getvalue") else None
    except Exception:
        data = None
    if not data or len(data) > 10 * 1024 * 1024:
        return
    recents = st.session_state.setdefault("recent_uploads", [])
    recents = [r for r in recents if r["name"] != name]
    recents.insert(0, {"name": name, "data": data,
                       "time": datetime.now().strftime("%I:%M %p")})
    st.session_state["recent_uploads"] = recents[:5]


def _clear_analysis(reset_variables=True):
    """Reset dataset-specific state before displaying a changed upload."""
    st.session_state["assistant_open"] = False
    for key in ("result", "regression"):
        previous = st.session_state.get(key)
        if previous:
            for _, fig in previous.get("figs", []):
                plt.close(fig)
    for key in ("result", "name", "regression", "chat_histories", "chat_history",
                "v7_design", "v7_bundle", "v7_interpretation",
                "reg_y", "reg_x", "v7_y", "v7_x", "dataset_sheet",
                "results_tabs", "expanded_table_chat_panel"):
        st.session_state.pop(key, None)
    for key in list(st.session_state):
        if reset_variables and key.startswith("variable_"):
            st.session_state.pop(key, None)
    for section in CHAT_SECTIONS:
        st.session_state.pop(f"explanation_{section}", None)
        st.session_state.pop(f"chat_{section}", None)


def analyze_and_store(source, name, sheet=None, clean_opts=None,
                      encode_cols=None):
    with st.spinner("Cleaning your data and crunching the numbers…"):
        result = core.analyze(source, name=name, sheet=sheet,
                              clean_opts=clean_opts,
                              encode_cols=encode_cols)
    _remember_upload(source, name)
    _clear_analysis()
    st.session_state["result"] = result
    st.session_state["name"] = name
    st.session_state["chat_histories"] = {}


def _applied_cleaning_options(result):
    """Read the analysis snapshot, including results from before snapshots existed."""
    if "clean_opts" in result:
        return result["clean_opts"]
    lines = result.get("clean_lines", [])
    return {
        "drop_dupes": "Skipped duplicate removal (option off)." not in lines,
        "convert_numerics": "Skipped numeric auto-conversion (option off)." not in lines,
        "fix_missing_tokens": "Skipped missing-marker standardization (option off)." not in lines,
    }


def _variable_review(result, name):
    base = st.session_state.setdefault("variable_base", result)
    source = base["df"]
    with st.expander("Review variable types & dummy encoding", expanded=True,
                     icon=":material/checklist:"):
        st.caption("Numeric text conversion changes storage, not meaning. Review suggestions "
                   "below: numbers can be quantities or category codes. Applying this step "
                   "updates all results and clears previous models and AI responses.")
        binary = [c for c in source if variable_types.suggest(source[c], c) ==
                  "Binary / existing dummy"]
        if binary:
            st.info("Existing 0/1 binary variables detected: " + ", ".join(binary) +
                    ". No additional encoding is needed for these columns.")
        st.caption("Categories produce one dummy per non-reference category. Original category "
                   "labels are retained for exploration; identifiers are excluded from regression.")
        roles, references = {}, {}
        for index, col in enumerate(source.columns):
            left, right = st.columns(2)
            suggested = variable_types.suggest(source[col], col)
            roles[col] = left.selectbox(
                f"{col} — variable type", variable_types.ROLES,
                index=variable_types.ROLES.index(suggested), key=f"variable_role_{index}",
                help="Suggestions use values and names, not a codebook. Override them when needed.")
            if roles[col] == "Categorical":
                cats = variable_types.levels(source[col])
                if 2 <= len(cats) <= 50:
                    references[col] = right.selectbox(
                        f"{col} — reference category", cats, key=f"variable_ref_{index}")
                    right.caption(f"Will create {len(cats) - 1} dummy column(s).")
                else:
                    right.warning("Encoding requires 2–50 categories. Review/group values or choose Identifier.")
            elif roles[col] == "Needs review":
                right.warning("These values may be category codes or quantities (possibly with invalid text). Choose their meaning.")
            else:
                right.caption("Examples: " + ", ".join(variable_types.levels(source[col])[:5]))
        if st.button("Apply variable types", key="variable_apply", type="primary"):
            try:
                prepared, log = variable_types.prepare(source, roles, references)
                with st.spinner("Updating analysis with your variable types…"):
                    updated = core.summarize_frame(
                        prepared, name, base["clean_lines"] + log,
                        base["raw_shape"], _applied_cleaning_options(base))
            except ValueError as exc:
                st.error(str(exc))
            else:
                # Preserve the original cleaned source so repeated reviews never double-encode.
                _clear_analysis(reset_variables=False)
                st.session_state["result"] = updated
                st.session_state["name"] = name
                st.session_state["variable_applied"] = log
                st.session_state["variable_choices"] = (roles, references)
                st.rerun()
        if st.session_state.get("variable_applied"):
            if st.session_state.get("variable_choices") != (roles, references):
                st.warning("You have unapplied changes. Results still use your last applied variable types.")
            else:
                st.success("Variable types applied. Results below use your last applied review.")
            for line in st.session_state["variable_applied"]:
                st.caption(line)
        else:
            st.caption("Suggestions have not been applied. Results currently use the cleaning settings only.")


def _regression_equation(reg):
    """Format the fitted model's variables as an OLS specification."""
    def variable(name):
        escapes = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
                   "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#",
                   "$": r"\$", "^": r"\textasciicircum{}",
                   "~": r"\textasciitilde{}"}
        label = "".join(escapes.get(char, char) for char in str(name))
        return rf"\text{{{label}}}_i"

    terms = [rf"\beta_{{{index}}}\,{variable(name)}"
             for index, name in enumerate(reg["x"], start=1)]
    return (variable(reg["y"]) + r" = \beta_0 + "
            + " + ".join(terms) + r" + \varepsilon_i")


def _start_analysis(source, name, sheet=None, clean_opts=None):
    """Keep file errors in the upload flow, then refresh the workspace."""
    try:
        analyze_and_store(source, name, sheet, clean_opts)
    except Exception as exc:
        st.error(f"Could not analyze this file: {exc}")
        return
    st.rerun()


def main():
    st.set_page_config(page_title="Auto-Research Assistant",
                       page_icon=":material/query_stats:", layout="wide")
    # Discard credentials/configuration left by older browser sessions.
    for legacy_key in ("ai_key", "ai_base", "ai_model_select", "ai_model_text",
                       "ai_models", "ai_models_cache_key"):
        st.session_state.pop(legacy_key, None)

    with st.sidebar:
        st.subheader(":material/query_stats: Auto-Research")
        st.caption("A little clarity for your next discovery.")
        st.badge("Research workspace", icon=":material/science:", color="blue")
        st.space("medium")
        with st.expander("Cleaning settings", icon=":material/tune:",
                         expanded=True):
            st.caption("Applied when you analyze a dataset.")
            opt_dupes = st.checkbox(
                "Remove duplicate rows", value=True, key="clean_dupes",
                help="Drop rows that are exact copies of another row.")
            opt_numeric = st.checkbox(
                "Convert numeric text", value=True, key="clean_numeric",
                help="Convert numeric-looking text into numbers, including existing 0/1 dummies. "
                     "This does not encode categories. Use Review variable types to mark numeric category codes.")
            opt_missing = st.checkbox(
                "Standardize missing values", value=True, key="clean_missing",
                help="Treat NA, N/A, -, ? and similar markers as missing values.")
        clean_opts = {"drop_dupes": opt_dupes,
                      "convert_numerics": opt_numeric,
                      "fix_missing_tokens": opt_missing}

        with st.container(border=True):
            st.markdown(":material/auto_awesome: **Research companion**")
            st.caption("Ask questions and get help interpreting your results." if _current_api_key()
                       else "Cleaning, statistics, and exports are ready. AI explanations become available when the app owner enables them.")

        recents = st.session_state.get("recent_uploads", [])
        if recents:
            with st.expander("Recent uploads", icon=":material/history:"):
                for r in recents:
                    if st.button(r["name"], icon=":material/description:",
                                 key=f"recent_{r['name']}_{r['time']}",
                                 width="stretch"):
                        bio = io.BytesIO(r["data"])
                        bio.name = r["name"]
                        _start_analysis(bio, r["name"], clean_opts=clean_opts)
        st.space("medium")
        st.caption("Built for curious minds.\n\nProject by Mhel Garret L. Suerte")

    result = st.session_state.get("result")
    if result is None:
        intro, artwork = st.columns([1.1, 1], gap="large", vertical_alignment="center")
        with intro:
            st.badge("Your next insight starts here", icon=":material/auto_awesome:", color="blue")
            st.title("Meet your data before you :blue[marry your model.]")
            st.markdown("Bring your dataset. Find the patterns. Build a clearer research story — all in one workspace.")
            st.caption("CSV, TXT & Excel  ·  Cleaning, analysis & reports")
        with artwork:
            st.image(str(HERO_IMAGE), width="stretch",
                     alt="Paper research charts connected to blue glass columns by a flowing ribbon.")
        st.space("small")
    else:
        st.caption("YOUR RESEARCH WORKSPACE")
        st.title("Make sense of your data")
        st.write("Explore the findings, check the details, and build your report.")
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.button("AI assistant", icon=":material/auto_awesome:", key="open_assistant",
                      on_click=_set_assistant_open, args=(True,))

    upload_area = (st.expander("Analyze another dataset", icon=":material/upload_file:")
                   if result is not None else st.container(border=True))
    with upload_area:
        st.subheader("Start with your dataset", anchor=False, icon=":material/upload_file:")
        st.caption("Drop in your data to begin. Just exploring? Try the sample survey below.")
        uploaded = st.file_uploader(
            "Choose a data file", type=["csv", "txt", "xlsx", "xls"],
            key="dataset_upload", label_visibility="collapsed",
            on_change=_clear_analysis)
        sheet = None
        sheet_error = False
        if uploaded is not None and Path(uploaded.name).suffix.lower() in (".xlsx", ".xls"):
            try:
                sheet = st.selectbox("Excel sheet", core.get_sheet_names(uploaded),
                                     key="dataset_sheet")
            except Exception as exc:
                st.error(f"Could not read the Excel sheets: {exc}")
                sheet_error = True
        with st.container(horizontal=True):
            if st.button("Analyze dataset", type="primary", key="analyze_dataset",
                         icon=":material/arrow_forward:",
                         disabled=uploaded is None or sheet_error):
                _start_analysis(uploaded, uploaded.name, sheet, clean_opts)
            if st.button("Try sample dataset", key="try_sample",
                         icon=":material/science:"):
                _start_analysis(SAMPLE, SAMPLE.name, clean_opts=clean_opts)
        st.caption(":material/tune: Cleaning settings are in the sidebar. Your uploaded source file is never overwritten.")

    if result is None:
        st.space("small")
        st.subheader("A clear path from data to findings", anchor=False)
        features = [
            ("01", "cleaning_services", "Prepare with confidence",
             "Tidy columns, remove duplicates, and standardize missing values."),
            ("02", "query_stats", "Find the useful patterns",
             "Summary statistics, charts, correlations, and regression diagnostics."),
            ("03", "description", "Bring your findings together",
             "Build a research report and download your cleaned data and findings."),
        ]
        for col, (step, icon, title, description) in zip(st.columns(3), features):
            with col.container(border=True, height="stretch"):
                st.caption(f"STEP {step}")
                st.markdown(f":blue[:material/{icon}:] **{title}**")
                st.caption(description)
        return

    name = st.session_state.get("name", "dataset")
    _variable_review(result, name)
    df = result["df"]
    st.subheader("Dataset overview", anchor=False)
    st.caption(f"Analyzed file: {name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}", border=True)
    c2.metric("Columns", f"{df.shape[1]:,}", border=True)
    c3.metric("Numeric columns", len(result["numeric_cols"]), border=True)
    c4.metric("Categorical columns", len(result["cat_cols"]), border=True)

    with st.container(border=True, key="results_navigation"):
        st.subheader("Explore your results", anchor=False)
        st.caption("Select a tab below to switch sections. "
                   "Scroll the tab bar sideways if more tabs are off-screen.")
        tab_overview, tab_stats, tab_charts, tab_reg, tab_v7, tab_download = st.tabs(
            [":material/table_chart: **Overview**",
             ":material/functions: **Statistics**",
             ":material/bar_chart: **Charts**",
             ":material/query_stats: **Regression**",
             ":material/description: **Research report**",
             ":material/download: **Downloads**"],
            key="results_tabs", on_change="rerun")

    with tab_overview:
        applied_opts = _applied_cleaning_options(result)
        skipped = [label for key, label in (
            ("drop_dupes", "duplicate removal"),
            ("convert_numerics", "numeric text conversion"),
            ("fix_missing_tokens", "missing-value standardization"),
        ) if not applied_opts[key]]
        st.subheader("Data preview" if skipped else "Cleaned data", anchor=False)
        if skipped:
            st.warning(
                "Skipped " + ", ".join(skipped) + ". "
                "Basic preparation still runs: column names are tidied, empty rows "
                "and columns are removed, and text whitespace is trimmed. "
                "Enable the options in Cleaning settings and analyze the dataset "
                "again to apply them.",
                title="Cleaning options were off" if len(skipped) == 3
                      else "Some cleaning steps were skipped",
                icon=":material/warning:")
        st.caption("Preview of the first 50 rows. Download the full dataset from Downloads.")
        st.dataframe(df.head(50), width="stretch", hide_index=True)
        with st.expander("Cleaning log", icon=":material/checklist:"):
            for line in result["clean_lines"]:
                st.write("• " + line)
        _explain_section("Explain cleaning", "cleaning", result, name)

    with tab_stats:
        st.subheader("Missing values")
        st.dataframe(result["missing"], width="stretch")
        st.subheader("Numeric summary")
        if not result["desc"].empty:
            st.dataframe(result["desc"].round(3), width="stretch")
        else:
            st.write("No numeric columns in this dataset.")
        if not result["top_cats"].empty:
            st.subheader("Top categories")
            st.dataframe(result["top_cats"], width="stretch")
        _explain_section("✨ Interpret the statistics with AI", "statistics",
                         result, name)

    with tab_charts:
        if result["figs"]:
            for title, fig in result["figs"]:
                st.subheader(title)
                st.pyplot(fig, width="stretch")
        else:
            st.write("No charts could be drawn for this dataset.")
        _explain_section("✨ Interpret the charts with AI", "charts",
                         result, name)

    with tab_reg:
        st.subheader("OLS regression")
        num_cols = result["numeric_cols"]
        if len(num_cols) < 2:
            st.info("Regression needs at least 2 numeric columns.")
        else:
            c1, c2 = st.columns(2)
            y_col = c1.selectbox("Y — the variable to explain",
                                 num_cols, key="reg_y")
            x_cols = c2.multiselect("X's — the explanatory variables",
                                    [c for c in num_cols if c != y_col],
                                    key="reg_x")
            if st.button("▶ Run regression", key="reg_run"):
                try:
                    reg = core.run_regression(df, y_col, x_cols)
                    st.session_state["regression"] = reg
                    st.session_state.pop("explanation_regression", None)
                    st.session_state.get("chat_histories", {}).pop(
                        "regression", None)
                except (ValueError, RuntimeError) as e:
                    st.error(str(e))
            reg = st.session_state.get("regression")
            if reg:
                st.latex(_regression_equation(reg))
                st.caption(f"OLS model · {reg['n_obs']:,} complete observations")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R²", f"{reg['r2']:.3f}")
                m2.metric("Adj. R²", f"{reg['adj_r2']:.3f}")
                m3.metric("F-statistic", f"{reg['f_stat']:.1f}")
                m4.metric("F p-value", f"{reg['f_pvalue']:.4g}")
                st.subheader("Coefficients")
                st.dataframe(reg["coef"].round(4), width="stretch")
                st.subheader("VIF — multicollinearity check")
                vif = reg["vif"].round(3)
                st.dataframe(vif, width="stretch")
                if (vif["VIF"] > 10).any():
                    st.warning("VIF above 10 — strong multicollinearity in "
                               "those variables. Consider dropping one.")
                elif (vif["VIF"] > 5).any():
                    st.warning("VIF above 5 — moderate multicollinearity. "
                               "Worth a look.")
                else:
                    st.success("All VIFs look fine (≤ 5).")
                st.subheader("Diagnostics")
                d = reg["diagnostics"]
                st.write(f"• Breusch-Pagan p = {d['breusch_pagan_p']:.4g} — "
                         f"{'possible heteroskedasticity'
                            if d['breusch_pagan_p'] < 0.05
                            else 'no sign of heteroskedasticity'}")
                st.write(f"• Jarque-Bera p = {d['jarque_bera_p']:.4g} — "
                         f"{'residuals look non-normal'
                            if d['jarque_bera_p'] < 0.05
                            else 'residuals look normal'}")
                for title, fig in reg["figs"]:
                    st.subheader(title)
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
                _explain_section("✨ Interpret the regression with AI",
                                 "regression", result, name, reg=reg)

    with tab_v7:
        st.subheader("One-click research assistant")
        st.write("From your cleaned dataset to a full research report. "
                 "The AI can suggest the research design — you confirm it, "
                 "then one click runs everything.")
        api_key_v7 = _current_api_key()
        if "v7_design" not in st.session_state:
            st.session_state["v7_design"] = None
        if api_key_v7 and st.button("🤖 Suggest research design",
                                   key="v7_suggest"):
            with st.spinner("The AI is studying your columns…"):
                try:
                    st.session_state["v7_design"] = \
                        v7_assistant.suggest_design(
                            result, api_key_v7,
                            base_url=_current_base_url(),
                            model=_current_model())
                except (RuntimeError, ValueError) as e:
                    st.warning(f"Could not suggest a design: {e}")
        design = st.session_state.get("v7_design")
        if design:
            st.success(f"AI suggestion: **{design['y']}** ~ "
                       f"{' + '.join(design['x'])}")
            st.caption(f"Why this Y: {design['why_y']}")
            st.caption(f"Why these X's: {design['why_x']}")
        num_cols_v7 = result["numeric_cols"]
        if len(num_cols_v7) < 2:
            st.info("One-click needs at least 2 numeric columns.")
        else:
            y_idx = (num_cols_v7.index(design["y"])
                     if design and design["y"] in num_cols_v7 else 0)
            y_v7 = st.selectbox("Y — the variable to explain",
                                num_cols_v7, index=y_idx, key="v7_y")
            x_default = [c for c in (design["x"] if design else [])
                         if c in num_cols_v7 and c != y_v7]
            x_v7 = st.multiselect(
                "X's — the explanatory variables",
                [c for c in num_cols_v7 if c != y_v7],
                default=x_default, key="v7_x")
            if st.button("🚀 Generate full report", key="v7_go"):
                with st.spinner("Cleaning → stats → charts → regression → "
                                "report…"):
                    try:
                        st.session_state["v7_bundle"] = \
                            v7_assistant.run_everything(
                                result=result, y_col=y_v7, x_cols=x_v7,
                                name=name)
                        st.session_state.pop("v7_interpretation", None)
                    except (ValueError, RuntimeError) as e:
                        st.error(str(e))
            bundle = st.session_state.get("v7_bundle")
            if bundle:
                if bundle["guards"]:
                    with st.expander("⚠️ Automatic checks", expanded=True):
                        for g in bundle["guards"]:
                            st.warning(g)
                else:
                    st.success("✅ Automatic checks passed — "
                               "no issues flagged.")
                st.markdown(bundle["report"])
                st.download_button(
                    "⬇️ Download FULL_REPORT.md",
                    bundle["report"].encode("utf-8"),
                    file_name=f"FULL_REPORT_{Path(name).stem}.md",
                    mime="text/markdown",
                )
                if st.button("✨ Write AI interpretation", key="v7_write"):
                    if not api_key_v7:
                        st.warning("AI is not configured yet. Contact the app owner to enable it.")
                    else:
                        messages = [
                            {"role": "system",
                             "content": ai_chat.SYSTEM_PROMPT
                             + "\n\nFULL RESULTS:\n" + bundle["report"]},
                            {"role": "user",
                             "content":
                             v7_assistant.FULL_INTERPRET_TASK},
                        ]
                        messages = ai_chat.trim_messages(messages)
                        with st.spinner("The AI is writing the "
                                        "interpretation…"):
                            try:
                                st.session_state["v7_interpretation"] = \
                                    ai_chat.chat_reply(
                                        api_key_v7, messages,
                                        base_url=_current_base_url(),
                                        model=_current_model())
                            except RuntimeError as e:
                                st.warning(f"⚠️ {e}")
                if st.session_state.get("v7_interpretation"):
                    with st.expander("✨ AI interpretation", expanded=True):
                        st.markdown(
                            st.session_state["v7_interpretation"])

    with tab_download:
        st.subheader("Take your results with you")
        st.download_button(
            "⬇️ Download cleaned CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"cleaned_{Path(name).stem}.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇️ Download report (Markdown)",
            result["report"].encode("utf-8"),
            file_name=f"report_{Path(name).stem}.md",
            mime="text/markdown",
        )
        st.caption("The full file bundle (charts as PNGs included) is "
                   "available from the command-line version: "
                   "`python src/v1_clean_analyze.py data/input/yourfile.csv`")



    for section, tab in zip(CHAT_SECTIONS, (
            tab_overview, tab_stats, tab_charts, tab_reg, tab_v7, tab_download)):
        if tab.open:
            _chat_ui(section, result, name, reg=st.session_state.get("regression"))
            break


if __name__ == "__main__":
    main()
