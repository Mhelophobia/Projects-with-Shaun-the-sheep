"""Manual visual fixture: streamlit run tests/chat_table_preview.py. No AI requests."""
from pathlib import Path
import sys
from unittest.mock import patch

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from app import _chat_ui

REPLY = """Here are a few checks to consider:

| Check | Why it matters | Suggested next step |
| --- | --- | --- |
| **Missing values** | Missing entries can change how many observations are available for your analysis. | Review the missing-value summary before selecting variables. |
| **Unusual values** | A valid observation may still deserve a closer look. | Inspect distributions and compare unusual values with the original source. |
| **Category codes** | Numeric labels do not always represent quantities. | Review variable types before fitting a model. |

Keep these checks in mind when interpreting your results.
"""

st.set_page_config(layout="wide")
st.title("Chat table preview")
st.caption("Visual check using an example response. No AI requests are sent.")
st.session_state["preview_runs"] = st.session_state.get("preview_runs", 0) + 1
st.caption(f"Page render count: {st.session_state['preview_runs']}")
st.session_state.setdefault("chat_histories", {"cleaning": [
    {"role": "assistant", "content": REPLY}]})
if st.button("AI assistant"):
    st.session_state["assistant_open"] = True
with (patch("app._current_api_key", return_value="preview"),
      patch("app._chat_system", return_value="Example table preview"),
      patch("app.ai_chat.chat_reply", return_value=REPLY)):
    _chat_ui("cleaning", {}, "example.csv")
