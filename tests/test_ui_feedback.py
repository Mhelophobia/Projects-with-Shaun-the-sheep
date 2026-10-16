"""UI feedback checks without calling an external AI service."""

from contextlib import contextmanager
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ai_chat


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.server_secrets = {}
        self.secrets = patch("streamlit.secrets", self.server_secrets)
        self.secrets.start()
        self.addCleanup(self.secrets.stop)
        self.app = AppTest.from_file(str(ROOT / "src/app.py"), default_timeout=40).run()

    def test_warning_tracks_analysis_not_current_checkboxes(self):
        for key in ("clean_dupes", "clean_numeric", "clean_missing"):
            self.app.checkbox(key=key).uncheck()
        self.app.button(key="try_sample").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any(w.proto.title == "Cleaning options were off"
                            for w in self.app.warning))
        # Existing browser sessions must also work without the new metadata.
        self.app.session_state["result"].pop("clean_opts")
        for key in ("clean_dupes", "clean_numeric", "clean_missing"):
            self.app.checkbox(key=key).check()
        self.app.run()
        self.assertTrue(any(w.proto.title == "Cleaning options were off"
                            for w in self.app.warning))
        self.app.button(key="try_sample").click().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(any("cleaning" in w.proto.title.lower()
                             for w in self.app.warning))

    def test_owner_key_and_connection_are_never_browser_controls(self):
        secret = "private-test-key-never-render"
        self.server_secrets.update({"GROQ_API_KEY": secret,
                                    "AI_MODEL": "owner-model"})
        self.app.session_state["ai_key"] = "legacy-key"
        self.app.session_state["ai_base"] = "https://untrusted.example/v1"
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertNotIn("ai_key", self.app.session_state)
        self.assertNotIn("ai_base", self.app.session_state)
        self.assertFalse(self.app.text_input)
        self.assertFalse(any(e.label == "AI setup" for e in self.app.expander))
        self.app.button(key="try_sample").click().run()
        with patch.object(ai_chat, "chat_reply", return_value="Reply") as request:
            self.app.chat_input(key="chat_cleaning").set_value("Summarize").run()
            self.assertEqual(request.call_args.args[0], secret)
            self.assertEqual(request.call_args.kwargs["base_url"], ai_chat.DEFAULT_BASE_URL)
            self.assertEqual(request.call_args.kwargs["model"], "owner-model")
            self.assertNotIn(secret, str(request.call_args.args[1]))
        for element in self.app:
            if hasattr(element, "proto"):
                self.assertNotIn(secret, str(element.proto))

    def test_upload_removal_and_replacement_clear_all_results(self):
        data = b"x,y\n1,3\n2,5\n3,8\n4,9\n5,12\n6,13\n"
        self.server_secrets["GROQ_API_KEY"] = "test-key"
        self.app.run()
        for action in ("remove", "replace"):
            with self.subTest(action=action):
                self.app.get("file_uploader")[0].upload("data.csv", data).run()
                self.app.button(key="analyze_dataset").click().run()
                self.assertFalse(self.app.exception)
                self.app.multiselect(key="reg_x").set_value(["y"]).run()
                self.app.button(key="reg_run").click().run()
                self.assertFalse(self.app.exception)
                self.app.session_state["v7_bundle"] = {"report": "Old report"}
                self.app.session_state["v7_design"] = {"y": "x", "x": ["y"]}
                self.app.session_state["v7_interpretation"] = "Old interpretation"
                self.app.session_state["explanation_cleaning"] = "Old explanation"
                self.app.session_state["chat_histories"] = {
                    "cleaning": [{"role": "user", "content": "Old question"}]}
                if action == "remove":
                    self.app.get("file_uploader")[0].clear().run()
                else:
                    # A new file can share the previous upload's name.
                    self.app.get("file_uploader")[0].upload(
                        "data.csv", data.replace(b"1,3", b"1,4")).run()
                self.assertFalse(self.app.exception)
                self.assertEqual(len(self.app.tabs), 0)
                self.assertEqual(len(self.app.metric), 0)
                for key in ("result", "name", "regression", "chat_histories",
                            "v7_bundle", "v7_design", "v7_interpretation",
                            "explanation_cleaning", "reg_x", "reg_y"):
                    self.assertNotIn(key, self.app.session_state)
                self.assertNotIn("ai_key", self.app.session_state)
                self.assertTrue(self.app.session_state["recent_uploads"])
                self.assertEqual(self.app.button(key="analyze_dataset").disabled,
                                 action == "remove")

        # The empty uploader must not erase a deliberately loaded sample on rerun.
        self.app.get("file_uploader")[0].clear().run()
        self.app.button(key="try_sample").click().run()
        self.app.checkbox(key="clean_dupes").uncheck().run()
        self.assertFalse(self.app.exception)
        self.assertIn("result", self.app.session_state)

    def test_chat_shows_loading_during_request_and_recovers_after_error(self):
        self.app.button(key="try_sample").click().run()
        self.server_secrets["GROQ_API_KEY"] = "test-key"
        self.app.run()
        original_spinner = st.spinner
        loading = []

        @contextmanager
        def observed_spinner(text, **kwargs):
            is_chat = text == "AI is thinking…"
            if is_chat:
                loading.append(True)
                self.assertTrue(kwargs["show_time"])
            try:
                with original_spinner(text, **kwargs):
                    yield
            finally:
                if is_chat:
                    loading.pop()

        def response(*args, **kwargs):
            self.assertTrue(loading, "AI request must run inside the loading indicator")
            return "Test reply"

        with patch("streamlit.spinner", observed_spinner), patch.object(
                ai_chat, "chat_reply", side_effect=response) as request:
            self.app.chat_input(key="chat_cleaning").set_value("Explain this data").run()
            self.assertFalse(self.app.exception)
            request.assert_called_once()
            self.assertEqual(len(self.app.session_state["chat_histories"]["cleaning"]), 2)
            self.assertEqual(self.app.session_state["chat_histories"]["cleaning"][-1]["content"],
                             "Test reply")
            self.assertFalse(loading)

        with patch("streamlit.spinner", observed_spinner), patch.object(
                ai_chat, "chat_reply", side_effect=RuntimeError("Service unavailable")):
            self.app.chat_input(key="chat_cleaning").set_value("Try again").run()
            self.assertFalse(self.app.exception)
            self.assertIn("Service unavailable",
                          self.app.session_state["chat_histories"]["cleaning"][-1]["content"])
            self.assertFalse(loading)


if __name__ == "__main__":
    unittest.main()
