# Blast Fragmentation Calculator + Rock-Factor ML + Vision

A blast-fragmentation prediction tool. Three components, each doing the job it's
suited for:

1. **Kuz-Ram model (in the browser)** — predicts muckpile fragment size
   distribution from blast design parameters. Verified formula, runs as JS.
2. **Machine-learning rock-factor estimator (server-side)** — a RandomForest
   trained on the Lilly blastability relationship; predicts rock factor *A*
   from five rock-mass properties.
3. **Spatial / vision model (server-side, via OpenRouter)** — reads a photo of a
   rock face and estimates those five properties, which feed the ML model.

Pipeline:

    photo -> [vision model] -> 5 rock properties -> [ML model] -> rock factor A -> [Kuz-Ram] -> fragment size

The vision and ML pieces run in a small Python (Flask) server because a browser
can't run the trained model or hold an API key securely.

---

## Files

- `server.py` — the Flask server (trains the model on startup, serves the page,
  handles the vision call and the ML prediction).
- `blast_calculator_server.html` — the calculator page + photo/ML panel.
- `rock_factor_training_data.csv` — training data; the model is rebuilt from this
  every time the server starts.
- `requirements.txt` — dependencies (for hosting).

---

## Run it locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Set your OpenRouter API key (this terminal only):
   - Windows:  `set OPENROUTER_API_KEY=sk-or-your-key-here`
   - Mac/Linux: `export OPENROUTER_API_KEY=sk-or-your-key-here`
3. Start it:
   ```
   python server.py
   ```
   It opens `http://127.0.0.1:5000` automatically.

The calculator and ML prediction work without a key; only **Analyze photo**
needs the key.

---

## Deploy it as a live link (Render free tier)

This gives you a public URL anyone can open — no install, no key on their end.

1. **Put these files in a GitHub repo** (server.py, the HTML, the CSV,
   requirements.txt). Do NOT put your API key in the repo.
2. Go to **render.com**, sign up (free), and click **New > Web Service**.
3. Connect your GitHub repo.
4. Set the configuration:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn server:app`
   - **Instance type:** Free
5. Under the **Environment** tab, add an environment variable:
   - Key: `OPENROUTER_API_KEY`
   - Value: your OpenRouter key
   (Render injects this at runtime — it stays server-side, never in the code.)
6. Click **Create Web Service**. After the build finishes you get a URL like
   `https://your-app-name.onrender.com`.

### Things to know about the free tier
- The service **sleeps after 15 minutes of inactivity** and takes ~30–60 seconds
  to wake up on the next visit. The first load after idle will hang briefly —
  that's normal, not a bug.
- The filesystem resets on restart. That's fine here: the model retrains from the
  CSV on every startup, and nothing is saved to disk.
- **Cost watch:** because the URL is public, anyone who has it can press *Analyze
  photo*, and each press uses *your* OpenRouter credits. Keep the vision model on
  a free OpenRouter model (the default `meta-llama/llama-4-maverick:free`) so this
  stays free/rate-limited. If you want, add a simple password gate before sharing
  widely.

---

## How it works / honest notes (for the write-up)

- The five rock-mass properties read from a photo are **estimates**, not
  measurements — the vision model is giving a first-pass read. Every value stays
  editable on the page so a human can correct it.
- Because the ML model's training labels come from the Lilly formula, the model
  **reproduces and generalizes the Lilly blastability relationship** — it does not
  claim to beat it or discover anything new.
- The Kuz-Ram math is the verified, established part. The judgment the project
  shows is putting ML/vision on the genuinely uncertain input (rock factor /
  rock properties) while keeping the reliable physics as a formula.

### References
- Cunningham, C.V.B. (1983, 1987) — Kuz-Ram fragmentation model.
- Lilly blastability index / Cunningham rock-factor formulation
  (e.g. Gheibie et al. 2009, *Int. J. Rock Mechanics & Mining Sciences* 46, 967–973).
