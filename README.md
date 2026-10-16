# Auto-Research Assistant — private beta

A research workspace by Mhel & Shaun for CSV, TXT, and Excel datasets.
Upload data, inspect cleaning and descriptive statistics, view charts,
run OLS regression, and export a research report. AI explanations and
section-specific chat use an owner-configured service.

## Run locally

Use Python 3.12 or newer (validated locally on Python 3.14).

```sh
python -m pip install -r requirements.txt
python -m streamlit run src/app.py
```

The app starts at http://localhost:8501. The bundled sample survey works
without an API key. The folder watcher additionally requires `watchdog`.

## Owner-managed AI

The key, model, and provider URL are server-side settings; there is no AI
configuration panel for testers. Configure `GROQ_API_KEY`, and optionally
`AI_BASE_URL` and `AI_MODEL`, in Streamlit Cloud's Secrets settings. See
[the blank example](.streamlit/secrets.example.toml). For local use, set these
in `~/.streamlit/secrets.toml` or `.streamlit/secrets.toml` (ignored by Git).
Never put real credentials in the repository, chat, issue, or screenshot.

AI requests send dataset summaries, column information, and conversation
messages to the configured provider. Use non-sensitive sample data during
beta testing. The app keeps uploads and recent-upload copies in server-side
session memory; it does not write web uploads to disk. The hosting provider
and AI provider have their own handling and retention policies.

## Cleaning behavior

The website's three checkboxes explicitly control duplicate removal,
numeric-text conversion, and missing-marker standardization. The preview
shows the options used for the completed analysis. Basic preparation still
trims whitespace, tidies column names, and removes empty rows and columns.
Numeric conversion preserves mixed text, common identifiers, leading zeros,
and long digit strings. Uploaded source files are never overwritten.

For direct Python/CLI use, duplicate removal and extra missing markers are
opt-in; numeric conversion is enabled by default. Removing or replacing an
uploaded file resets its displayed results, regression, reports, and chats.

## Private beta deployment

See [BETA.md](docs/BETA.md) for deployment, private access, and verification.
App entry point: `src/app.py`. Keep the repository and app private.

## Checks

```sh
python -m unittest discover -s tests -q
```

Tests cover parsing, value preservation, cleaning options, exports, upload
resets, AI loading feedback, and keeping owner credentials out of the UI.
AI tests use a mocked provider and do not spend API credits.

## CLI

```sh
python src/v1_clean_analyze.py data/input/sample_survey.csv
python src/v1_clean_analyze.py data/input/sample_survey.csv --remove-duplicates --missing-tokens
```

Results are exploratory. Verify important findings against the underlying
statistics; AI-generated explanations can be incorrect.
