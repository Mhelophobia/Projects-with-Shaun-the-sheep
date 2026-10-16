# Private beta release

## Deployment

1. Deploy the private GitHub repository in Streamlit Community Cloud.
2. Select `main`, entry point `src/app.py`, and Python 3.14 (or validate the
   pinned dependencies on another supported Python version first).
3. In Advanced settings > Secrets, set `GROQ_API_KEY` privately. Optional
   settings: `AI_BASE_URL` and `AI_MODEL`; see the blank example in `.streamlit`.
   The existing local secrets file is not part of the repository.
4. Confirm app Sharing is **Only specific people can view this app** before
   distributing a link. Add approved testers as viewers, not developers.
5. Confirm an unauthenticated browser cannot open the app. Streamlit Community
   Cloud currently permits one private app per workspace/account limitation;
   check account eligibility before relying on this deployment route.

Official guides:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app
- https://docs.streamlit.io/deploy/concepts/secrets

## Beta verification

- Open the sample dataset, check tables and charts, run a regression, and export.
- Upload CSV/TXT/Excel; test sheet selection and cleaning options on/off.
- Remove and replace an upload; old results and chats must disappear.
- Ask AI a question; verify the loading indicator, response, and error handling.
- Open two separate browser sessions; uploads and chats must remain separate.
- Confirm no API-key, provider-URL, or model input is displayed to a tester.
- Check provider usage during the beta and set an appropriate provider-side
  spending/usage limit where supported. The app has no per-user billing quota.
- Collect feedback with the section, expected/actual behavior, and a redacted
  screenshot. Do not include API keys, confidential data, or full datasets.

## Release scope and limitations

Private beta, intended for a small invited group using non-sensitive data.
Cloud authentication and viewer restrictions are required: the application
itself does not implement a login screen. Do not expose it as a public app.
Cloud upload-size and memory limits may require smaller datasets. There is no
load-test or uptime guarantee. Local tests mock the AI service; the deployed
provider connection still needs a live smoke test by the owner.

The old data-cleaning issues (mixed values, identifier loss, column-name
collisions, and incomplete CLI exports) were addressed during beta preparation.
Numeric category codes and ambiguous identifiers still require user judgment.
