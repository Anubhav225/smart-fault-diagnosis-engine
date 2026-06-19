# ⚙️ Smart Fault Diagnosis System

AI-powered industrial fault detection using Groq

---

## Setup (Windows + VS Code)

### 1. Get a FREE Groq API Key
→ **https://console.groq.com** · Sign in · API Keys · Create Key · Copy it

### 2. Open project in VS Code
Unzip → `File → Open Folder` → select the folder

### 3. Terminal (`Ctrl + `` ` ``)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure API key
```bash
python setup_env.py
```
Paste your key when prompted. Saved to `.env`, never shown in the app.

### 5. Generate sample data (optional)
```bash
python generate_samples.py
```

### 6. Run
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## Troubleshooting

**`TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`**
This is caused by an `httpx` version newer than what `groq` expects.
Fix:
```bash
pip install -r requirements.txt --force-reinstall
```
This project pins `httpx==0.27.2` to avoid it.

**`UnicodeEncodeError` when running `generate_samples.py`**
Already fixed — all file writes use `encoding="utf-8"` explicitly.

---

## Streamlit Cloud Deploy
1. Push to GitHub (`.env` is gitignored)
2. **share.streamlit.io** → New app → select repo + `app.py`
3. Settings → Secrets → add: `GROQ_API_KEY = "gsk_xxx"`
4. Deploy ✅
