"""Browser fixture for fragment updates. No external AI calls are made."""
from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import app

# Persistent stubs also apply when Streamlit reruns a registered fragment.
app._current_api_key = lambda: "preview"
app.ai_chat.chat_reply = lambda *args, **kwargs: (
    "Here is a small example table.\n\n"
    "| Check | Result |\n| --- | --- |\n| Missing values | Review the summary |\n"
)
st.session_state["preview_page_runs"] = st.session_state.get("preview_page_runs", 0) + 1
st.caption(f"Full page renders: {st.session_state['preview_page_runs']}")
app.main()
