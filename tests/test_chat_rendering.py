"""Chat table rendering preserves content without widening message text."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chat_rendering import message_blocks
from streamlit.testing.v1 import AppTest


class ChatRenderingTests(unittest.TestCase):
    def test_tables_preserve_strings_and_surrounding_text(self):
        blocks = list(message_blocks(
            "Before\n\n| Code | Detail |\n| --- | :---: |\n"
            "| 001 | left \\| right |\n| 0.0001 | **Important** |\n\nAfter"))
        self.assertEqual([kind for kind, _ in blocks], ["markdown", "table", "markdown"])
        self.assertEqual(blocks[1][1][1], [["001", "left | right"], ["0.0001", "**Important**"]])
        self.assertIn("Before", blocks[0][1])
        self.assertIn("After", blocks[2][1])

    def test_fenced_and_invalid_tables_remain_markdown(self):
        for message in ("```markdown\n| A | B |\n| --- | --- |\n| 1 | 2 |\n```",
                        "| A | B |\n| --- |\n| 1 | 2 |"):
            self.assertEqual(list(message_blocks(message)), [("markdown", message)])

    def test_inline_code_pipes_do_not_split_cells(self):
        blocks = list(message_blocks("| Check | Code |\n| --- | --- |\n"
                                     "| Filter | `df[(x > 1) | (y < 2)]` |\n"
                                     "| Next | unchanged |"))
        self.assertEqual(blocks[0][1][1], [
            ["Filter", "`df[(x > 1) | (y < 2)]`"], ["Next", "unchanged"]])

    def test_wide_table_renders_as_grid_including_duplicate_headers(self):
        app = AppTest.from_string(
            'from chat_rendering import render_message\n'
            'render_message("Intro\\n| A | A | C | D |\\n| --- | --- | --- | --- |\\n'
            '| 001 | 2 | 3 | Long description |\\n\\nEnd")'
        ).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(app.dataframe[0].value.iloc[0].tolist(), ["001", "2", "3", "Long description"])
        self.assertTrue(any(m.value == "Intro" for m in app.markdown))
        self.assertTrue(any("End" in m.value for m in app.markdown))
        self.assertEqual(len(app.get("bidi_component")), 1)
        self.assertFalse(app.button)  # Expansion has no server-side click handler.

    def test_table_readers_are_isolated_and_leave_chat_state_unchanged(self):
        app = AppTest.from_string('''
import streamlit as st
from chat_rendering import render_message
st.session_state.setdefault("history", ["saved reply"])
st.session_state.setdefault("panel", True)
st.session_state.setdefault("other_panel", False)
with st.popover("Assistant", key="panel", on_change="rerun"):
    render_message("| A | B |\\n| --- | --- |\\n| 1 | 2 |", key="overview")
    render_message("| A | B |\\n| --- | --- |\\n| 3 | 4 |", key="statistics")
''').run()
        self.assertFalse(app.exception)
        self.assertEqual([e.key for e in app.get("bidi_component")],
                         ["overview_table_0", "statistics_table_0"])
        self.assertTrue(app.session_state["panel"])
        self.assertFalse(app.session_state["other_panel"])
        self.assertEqual(app.session_state["history"], ["saved reply"])
        self.assertFalse(app.button)
