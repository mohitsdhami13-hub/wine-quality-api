# Wine Quality API — Vercel version

Folder layout Vercel expects:

```
wine-api-vercel/
  api/
    index.py              <- the FastAPI app (already here)
    wine_red_model.joblib  <- YOU add this (from Colab)
    wine_white_model.joblib <- YOU add this (from Colab)
    feature_names.json     <- YOU add this (from Colab)
    metrics.json            <- YOU add this (from Colab)
  requirements.txt
  vercel.json
```

**Important:** the 4 files you downloaded from Colab must go INSIDE the
`api/` folder, next to `index.py` — not in the root. Vercel only bundles
files that live alongside the function.

## Deploy steps

1. Create a GitHub repo (e.g. `wine-quality-api`) and upload this entire
   folder's contents, keeping the same structure above (including your
   4 Colab files inside `api/`).
2. Go to https://vercel.com and sign up/log in with GitHub.
3. Click "Add New" -> "Project", select your repo, click "Import".
4. Vercel auto-detects the Python function — you don't need to change any
   build settings. Click "Deploy".
5. After ~1 minute you'll get a URL like `https://wine-quality-api.vercel.app`.
6. Test it: open `https://wine-quality-api.vercel.app/api/health` — you
   should see `{"status":"ok"}`.

Note the routes are under `/api/...` here (`/api/health`, `/api/predict`,
`/api/metrics`) rather than at the root — that's a Vercel convention.
