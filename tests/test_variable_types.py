from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import variable_types as vt


class VariableTests(unittest.TestCase):
    def test_binary_and_numeric_categories_independent_of_storage(self):
        for data in ([0, 1, None], ["0", "1", None]):
            self.assertEqual(vt.suggest(pd.Series(data), "female"), "Binary / existing dummy")
        self.assertEqual(vt.suggest(pd.Series([1, 2, 3]), "region"), "Needs review")
        self.assertEqual(vt.suggest(pd.Series([1, 2, 3]), "person_id"), "Identifier")
        df = pd.DataFrame({"female": ["0", "1", None], "region": [1, 2, None],
                           "region__2.0": [5, 6, 7]})
        out, log = vt.prepare(df, {"female": "Binary / existing dummy",
            "region": "Categorical", "region__2.0": "Numeric"}, {"region": "1.0"})
        self.assertEqual(list(out.columns), list(df.columns) + ["region__2.0_2"])
        self.assertEqual(out["female"].dropna().tolist(), [0, 1])
        self.assertEqual(out["region__2.0_2"].dropna().tolist(), [0, 1])
        self.assertTrue(np.isnan(out["region__2.0_2"].iloc[2]))
        self.assertEqual(df["female"].iloc[0], "0")

    def test_invalid_decisions_do_not_coerce_or_encode_silently(self):
        df = pd.DataFrame({"x": ["1", "no"]})
        for role in ("Needs review", "Numeric", "Binary / existing dummy"):
            with self.assertRaises(ValueError):
                vt.prepare(df, {"x": role}, {})
        with self.assertRaises(ValueError):
            vt.prepare(df, {"x": "Categorical"}, {"x": "absent"})

    def test_review_apply_repeat_and_reset(self):
        with patch("streamlit.secrets", {}):
            app = AppTest.from_file(str(ROOT / "src/app.py"), default_timeout=40).run()
            app.get("file_uploader")[0].upload("review.csv",
                b"wage,female,region,person_id\n10,0,1,001\n20,1,2,002\n30,0,3,003\n40,1,1,004\n").run()
            app.button(key="analyze_dataset").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Existing 0/1" in x.value for x in app.info))
            app.selectbox(key="variable_role_0").select("Numeric")
            app.selectbox(key="variable_role_2").select("Categorical").run()
            app.selectbox(key="variable_ref_2").select("2").run()
            app.session_state["v7_bundle"] = {"report": "stale"}
            app.button(key="variable_apply").click().run()
            self.assertFalse(app.exception)
            result = app.session_state["result"]
            self.assertIn("region__1", result["numeric_cols"])
            self.assertNotIn("region__2", result["numeric_cols"])
            self.assertNotIn("person_id", result["numeric_cols"])
            self.assertNotIn("region", result["numeric_cols"])
            self.assertNotIn("v7_bundle", app.session_state)
            columns = result["df"].columns.tolist()
            app.button(key="variable_apply").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(columns, app.session_state["result"]["df"].columns.tolist())
            app.get("file_uploader")[0].clear().run()
            self.assertFalse(app.exception)
            self.assertNotIn("variable_base", app.session_state)


if __name__ == "__main__":
    unittest.main()
