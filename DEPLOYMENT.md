# 🚀 Project Deployment Guide: GitHub & Render

This document provides a comprehensive, step-by-step guide to deploying the **Accident Severity Predictor & Hotspots Analysis** project to **GitHub** and hosting it live on **Render**.

---

## 📋 System Architecture & Overview

- **Backend**: FastAPI (`backend/app.py`) providing:
  - Severity prediction (`POST /predict`) with calibrated multi-factor risk inference.
  - Explainable risk factors (e.g., Highway Junctions, Adverse Weather, Low Visibility).
  - Health check endpoint (`GET /health`).
  - Interactive OpenAPI Swagger docs (`GET /docs`).
  - Direct static delivery of the frontend interface (`GET /`).
- **Frontend**: Responsive modern UI (`frontend/index.html`) featuring:
  - Visible car collision showcase banner and styled accident backdrop.
  - Authentic Google Maps tiles (Streets, Hybrid/Satellite, Terrain).
  - Historical accident severity hotspots (Levels 1–4) with pulsing markers.
  - Resilient geolocation detection (Hardware GPS + Network IP fallback).
  - Route calculation with distance and Estimated Time of Arrival (ETA).
- **Deployment Strategy**: **Unified Single-Service Deployment** on Render. The FastAPI application serves both the backend API and the frontend HTML at a single public HTTPS URL, eliminating CORS issues and hosting overhead.

---

## 🛠️ Prerequisites

Before you start, make sure you have:
1. **Git** installed on your system ([git-scm.com](https://git-scm.com/)).
2. A **GitHub account** ([github.com](https://github.com/)).
3. A **Render account** ([render.com](https://render.com/)) — free tier available with no credit card required.

---

## 📁 Phase 1: Preparing the Local Project

### 1.1 Verify `.gitignore`
Large files (such as datasets over 100MB and local virtual environments) must **never** be committed to GitHub.

Ensure your root `.gitignore` contains:
```gitignore
# Python bytecode & cache
__pycache__/
*.py[cod]

# Virtual environments
road_env/
.venv/
backend/.venv/
env/
venv/

# Large Datasets (GitHub blocks files > 100MB)
data/
*.xls
*.xlsx
*.csv

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# IDE & OS files
.vscode/
.idea/
.DS_Store
Thumbs.db
```

### 1.2 Verify `backend/requirements.txt`
Ensure `backend/requirements.txt` specifies production dependencies:
```txt
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.0.0
scikit-learn>=1.4.0
xgboost>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
```

---

## 🐙 Phase 2: Pushing Code to GitHub

### Step 2.1: Initialize Git Repository
Open your terminal (PowerShell or Bash) in the project root directory:

```bash
cd c:\Users\Ranaveer\Downloads\SDC
git init
```

### Step 2.2: Stage and Commit the Files
```bash
git add .
git commit -m "Initial release: Accident Severity Predictor with Google Maps & Hotspots"
```

> **Note**: If `data/SDC_DATASET.xls` (3.19 GB) was previously tracked, verify that `git status` shows it is excluded by `.gitignore`.

### Step 2.3: Create a New GitHub Repository
1. Go to [github.com/new](https://github.com/new).
2. Set **Repository Name**: `accident-severity-predictor` (or your choice).
3. Set Visibility: **Public** (recommended for free Render deployment) or **Private**.
4. Leave **Initialize with README** unchecked (since your project already exists).
5. Click **Create repository**.

### Step 2.4: Connect Remote and Push
Copy the GitHub repository URL and execute:

```bash
# Rename the default branch to main
git branch -M main

# Link your local repo to GitHub (replace with your actual GitHub username)
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/accident-severity-predictor.git

# Push the code to GitHub
git push -u origin main
```

---

## ☁️ Phase 3: Deploying on Render

### Step 3.1: Create a New Web Service on Render
1. Log in to [dashboard.render.com](https://dashboard.render.com/).
2. In the top right corner, click **New +** and select **Web Service**.
3. Under **Connect a repository**, select your GitHub account and choose `accident-severity-predictor`.
   *(If your repo does not appear, click "Configure GitHub App" to grant Render access to your repository).*

### Step 3.2: Configure Web Service Settings
Fill in the configuration fields as follows:

| Field | Setting / Value | Notes |
|---|---|---|
| **Name** | `accident-severity-predictor` | Becomes your subdomain: `https://accident-severity-predictor.onrender.com` |
| **Region** | *Choose closest region* (e.g., `Oregon (US West)` or `Frankfurt (EU)`) | |
| **Branch** | `main` | |
| **Root Directory** | *(Leave blank)* | Uses the repository root |
| **Runtime** | `Python 3` | |
| **Build Command** | `pip install -r backend/requirements.txt` | Installs all FastAPI & ML packages |
| **Start Command** | `python -m uvicorn backend.app:app --host 0.0.0.0 --port $PORT` | Binds dynamically to Render's allocated port |
| **Instance Type** | `Free` | 512 MB RAM, 0.1 CPU |

### Step 3.3: Set Environment Variables
Scroll down to the **Environment Variables** section on Render and click **Add Environment Variable**:

1. **`PYTHON_VERSION`** = `3.11.9`
2. **`OMP_NUM_THREADS`** = `1` *(Prevents multi-threading hangs on Linux containers)*

### Step 3.4: Deploy
Click **Create Web Service**.

Render will now:
1. Clone your GitHub repository.
2. Build the Python environment and run `pip install -r backend/requirements.txt`.
3. Launch your service with `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.

Once the log displays `Application startup complete.`, your service status will turn **Live** (indicated with a green badge).

---

## 🔍 Phase 4: Verifying Your Live Deployment

Your web application will be accessible at:
```
https://<your-service-name>.onrender.com/
```

### Endpoints to Test:
- **Interactive Web UI**: `https://<your-service-name>.onrender.com/`
  - Verify the vehicle accident showcase banner is clearly visible.
  - Test the **"Detect & Use My Current Location"** button (uses network IP fallback if GPS is restricted).
  - Explore the Google Map and click on accident hotspots (Severity 1 to 4).
  - Submit a severity prediction test.
- **Swagger API Docs**: `https://<your-service-name>.onrender.com/docs`
- **Health Check**: `https://<your-service-name>.onrender.com/health`
- **Prediction POST API**: `POST https://<your-service-name>.onrender.com/predict`

---

## 💡 Phase 5: Troubleshooting & FAQ

### 1. Render Free Tier Cold Starts
- On Render's free tier, the service automatically spins down after **15 minutes of inactivity** to conserve resources.
- When a user visits after inactivity, the service takes **30–50 seconds** to wake up. This is expected behavior on free cloud plans.
- Subsequent requests will respond instantly (< 100ms).

### 2. Automatic Updates (Continuous Deployment)
- Whenever you make changes locally and push them to GitHub (`git push origin main`), Render will **automatically detect the push, rebuild, and redeploy** your application with zero downtime.

### 3. Dynamic API URL Resolution
- In `frontend/index.html`, the prediction request automatically detects the host:
  ```javascript
  const apiUrl = window.location.origin.includes("http") && !window.location.origin.startsWith("file")
    ? `${window.location.origin}/predict`
    : "http://127.0.0.1:8000/predict";
  ```
  This ensures that the app works seamlessly **both locally on your computer** and **live on Render** without needing manual URL adjustments.

### 4. GitHub 100 MB Limit Error
- If Git warns about `SDC_DATASET.xls` being too large:
  Ensure `data/` is present in `.gitignore`. If it was already staged, run:
  ```bash
  git rm -r --cached data/
  git commit -m "Untrack large dataset file"
  git push origin main
  ```
