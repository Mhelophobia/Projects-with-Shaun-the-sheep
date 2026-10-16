# Reviewing variable types

After analyzing a dataset, open **Review variable types & dummy encoding**.
Suggestions describe possible meanings, independently of whether values were
loaded as text or numbers. They are heuristics, not a replacement for a codebook.

- **Binary / existing dummy:** 0/1 values are kept numeric without re-encoding.
- **Numeric:** quantities are converted to numbers; invalid values cause an error.
- **Categorical:** choose a reference category. The app retains the original
  labels for exploration and creates one numeric dummy for every other category.
  The reference category has all zeros; missing values remain missing in every
  dummy. Encoding supports 2–50 observed categories per column.
- **Identifier:** retained as text and excluded from numeric regression choices.
- **Needs review:** a small set of integers could be quantities or category codes.
  Choose a meaning explicitly before applying the review.

Click **Apply variable types** to rebuild statistics, charts, reports and downloads.
Previous fitted models and AI responses are cleared. Edits to the selectors are
only drafts until applied. Existing analysis remains available before applying.
Repeated reviews always start from the original cleaned data, so they do not
encode previously generated dummies again. Uploading a different file resets the
review. To reverse encoding, choose Numeric or Binary for the original column
and apply again; reload/reanalyze the file to reset the entire review.

For an already prepared dataset such as cps4c_small.csv, keep numeric text
conversion enabled and keep existing 0/1 columns as Binary / existing dummy.
For numeric region codes, select Categorical even if numeric conversion was on.

Generated dummies omit the selected reference category for models with an
intercept. Existing dummy sets are not altered automatically: users still need
to avoid selecting every mutually exclusive group indicator with an intercept.
