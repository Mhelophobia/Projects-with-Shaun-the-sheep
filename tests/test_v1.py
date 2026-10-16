import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import core
import pandas as pd


class DataIntegrityTests(unittest.TestCase):
    def tearDown(self):
        core.plt.close("all")

    def test_mixed_numeric_text_is_preserved(self):
        values = ["1", "2", "3", "4", "suppressed"]
        df, log = core.clean(pd.DataFrame({"value": values}))
        self.assertEqual(df.value.tolist(), values)
        self.assertTrue(any("Kept 'value'" in line for line in log))

    def test_mixed_excel_cells_do_not_disappear(self):
        df, _ = core.clean(pd.DataFrame({"value": [1, 2, "3", "unknown"]}))
        self.assertEqual(df.value.tolist(), [1, 2, "3", "unknown"])

    def test_fully_numeric_text_converts(self):
        df, _ = core.clean(pd.DataFrame({"value": ["1", "2.5", ""]}))
        self.assertEqual(df.value.iloc[:2].tolist(), [1, 2.5])
        self.assertTrue(pd.api.types.is_numeric_dtype(df.value))

    def test_conversion_can_be_disabled(self):
        df, _ = core.clean(pd.DataFrame({"value": ["1", "2"]}), auto_numeric=False)
        self.assertEqual(df.value.tolist(), ["1", "2"])

    def test_duplicate_rows_default_retained(self):
        df, log = core.clean(pd.DataFrame({"value": [1, 1, 2]}))
        self.assertEqual(len(df), 3)
        self.assertTrue(any("Retained 1 duplicate" in line for line in log))

    def test_duplicate_rows_opt_in_removed(self):
        df, _ = core.clean(pd.DataFrame({"value": [1, 1, 2]}), remove_duplicates=True)
        self.assertEqual(df.value.tolist(), [1, 2])

    def test_column_name_collisions(self):
        headers = ["a", " a", "a_2", "a", "a_3", " ", "column"]
        df, _ = core.clean(pd.DataFrame([[1] * len(headers)], columns=headers))
        self.assertTrue(df.columns.is_unique)
        self.assertEqual(df.columns[2], "a_2")
        self.assertEqual(df.columns[4], "a_3")
        self.assertEqual(df.columns[6], "column")

    def test_original_dataframe_unchanged(self):
        raw = pd.DataFrame({" value ": [" 1 ", " 1 "]})
        snapshot = raw.copy(deep=True)
        core.clean(raw, remove_duplicates=True)
        pd.testing.assert_frame_equal(raw, snapshot)

    def test_missing_markers_are_opt_in_and_trimmed_first(self):
        raw = pd.DataFrame({"value": [" NA ", " N/A ", "valid"], "row": [1, 2, 3]})
        kept, _ = core.clean(raw)
        self.assertEqual(kept.value.tolist(), ["NA", "N/A", "valid"])
        missing, _ = core.clean(raw, use_missing_tokens=True)
        self.assertEqual(int(missing.value.isna().sum()), 2)

    def test_identifiers_and_long_strings_survive_csv_loading(self):
        source = io.StringIO("id,serial,large,region,value\n001,0004,12345678901234567890,NA,4\n002,0005,12345678901234567891,US,6\n")
        result = core.analyze(source, name="example.csv")
        self.assertEqual(result["df"].id.tolist(), ["001", "002"])
        self.assertEqual(result["df"].serial.iloc[0], "0004")
        self.assertEqual(result["df"].large.iloc[0], "12345678901234567890")
        self.assertEqual(result["df"].region.iloc[0], "NA")
        self.assertEqual(result["desc"].loc["value", "mean"], 5)

    def test_xlsx_sheet_and_mixed_cells(self):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame({"other": [99]}).to_excel(writer, sheet_name="First", index=False)
            pd.DataFrame({"value": [1, "text", 3], "id": ["001", "002", "003"]}).to_excel(writer, sheet_name="Data", index=False)
        buffer.seek(0)
        result = core.analyze(buffer, name="example.xlsx", sheet="Data")
        self.assertEqual(result["df"].value.tolist(), [1, "text", 3])
        self.assertEqual(result["df"].id.tolist(), ["001", "002", "003"])

    def test_txt_delimiter(self):
        result = core.analyze(io.StringIO("x;y\n1;3\n2;4\n"), name="example.txt")
        self.assertEqual(result["df"].shape, (2, 2))
        self.assertEqual(result["desc"].loc["x", "mean"], 1.5)

    def test_all_empty_data(self):
        result = core.analyze(io.StringIO("a,b\n,\n"), name="empty.csv")
        self.assertTrue(result["df"].empty)
        self.assertEqual(result["figs"], [])

    def test_empty_file_raises(self):
        with self.assertRaises(pd.errors.EmptyDataError):
            core.analyze(io.StringIO(""), name="empty.csv")

    def test_cli_outputs_and_duplicate_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.csv"
            source.write_text("x,y\n1,3\n1,3\n2,5\n", encoding="utf-8")
            output = root / "report"
            run = subprocess.run([sys.executable, "-B", str(ROOT / "src/v1_clean_analyze.py"),
                                  str(source), "--out", str(output), "--remove-duplicates"],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            for filename in ["cleaned.csv", "cleaning_log.txt", "summary_statistics.csv",
                             "missing_values.csv", "top_categories.csv", "report.md"]:
                self.assertTrue((output / filename).is_file(), filename)
            self.assertEqual(len(pd.read_csv(output / "cleaned.csv")), 2)
            self.assertTrue(list((output / "charts").glob("*.png")))

    @unittest.skipUnless(os.environ.get("RUN_STREAMLIT_TESTS") == "1",
                         "Optional UI check requires a local socket-capable runtime")
    def test_streamlit_sample(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "src/app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("result", app.session_state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
