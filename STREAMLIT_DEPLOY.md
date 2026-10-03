# Deploy to Streamlit Cloud (Single App)

This is the refactored version that runs as a single process - perfect for Streamlit Cloud.

## Files created
- `streamlit_app.py` - Standalone Streamlit app (no FastAPI needed)
- `requirements_streamlit.txt` - Minimal deps for Streamlit Cloud

## Steps to deploy

1. **Push to GitHub**: Push this entire folder to a GitHub repo (exclude `.env`, `__pycache__`, `*.db`).
2. **Go to Streamlit Cloud**: Open [share.streamlit.io](https://share.streamlit.io) and sign in.
3. **Create app**: Click "New app" → "Deploy a public app from GitHub".
4. **Configure**:
   - Repository: select your repo
   - Branch: `main` (or your default)
   - Main file path: `streamlit_app.py`
   - App URL: choose as desired
5. **Add secrets**: Click "Advanced settings → Secrets". Add:

   ```toml
   GROQ_API_KEY = "gsk_your_real_groq_key_here"
   GROQ_MODEL = "openai/gpt-oss-120b"
   ```

6. **Deploy**: Click "Deploy". Streamlit will install dependencies from `requirements.txt` by default. If you want minimal deps, rename `requirements_streamlit.txt` to `requirements.txt` before pushing, or keep as-is - Streamlit will use existing `requirements.txt` (it has extra deps like fastapi/uvicorn but that's fine, they just take a bit longer to install).

## Notes
- Groq key stays in Streamlit secrets (never committed).
- SQLite `lease_register.db` persists on Streamlit Cloud's filesystem for your app instance.
- Sample leases folder works as-is (shows in dropdown).
- PDF + TXT upload both supported.

## Alternative: use existing requirements.txt
The current `requirements.txt` includes FastAPI/uvicorn etc - Streamlit Cloud will still install them even if unused. That's harmless but slightly slower. Using `requirements_streamlit.txt` is cleaner.

## Local testing
```bash
streamlit run streamlit_app.py
```

It will read from `.env` if secrets aren't set (we have fallback in code).
