# Quick Netlify Deployment Guide

## What you need to deploy:

### ✅ Files Created (Ready!)
- `netlify.toml` - Netlify configuration
- `frontend/.env.production` - Production environment variables
- `frontend/.env.development` - Development environment variables
- `DEPLOYMENT.md` - Complete deployment guide

### ⚠️ What's Missing

Your app has **two parts** that need separate hosting:

#### 1. **Frontend** (can go on Netlify) ✅
- React + Vite application
- Static files only
- Hosted on Netlify

#### 2. **Backend** (CANNOT go on Netlify) ❌
- FastAPI Python server
- Needs to run Python code
- Must be hosted elsewhere (Render.com, Railway, Heroku, etc.)

---

## 🚀 Quick Deployment Steps

### Step 1: Deploy Backend First
1. Go to **Render.com** (free tier available)
2. Connect your GitHub repo
3. Create a new "Web Service"
4. Configure:
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Deploy and get your URL (e.g., `https://your-app.onrender.com`)

### Step 2: Update Frontend Configuration
1. Edit `frontend/.env.production`
2. Replace `your-backend-url-here.com` with your actual Render URL
3. Commit and push to GitHub

### Step 3: Deploy Frontend on Netlify
1. Go to **Netlify.com**
2. Click "Add new site" → "Import from Git"
3. Select your repository
4. Configure:
   - Base directory: `frontend`
   - Build command: `npm install && npm run build`
   - Publish directory: `frontend/dist`
5. Add Environment Variable:
   - Key: `VITE_API_URL`
   - Value: Your Render backend URL
6. Deploy!

### Step 4: Update Backend CORS
1. Edit `backend/config.py`
2. Add your Netlify URL to `CORS_ORIGINS`
3. Redeploy backend

---

## 📖 Full Instructions

See `DEPLOYMENT.md` for complete step-by-step instructions with troubleshooting tips.

---

## ⚠️ Important Notes

1. **Netlify ONLY hosts the frontend** - it cannot run your Python backend
2. **Backend must be hosted separately** on a service that supports Python
3. **Free tier limitations**:
   - Render free tier: Backend sleeps after 15 min of inactivity
   - First request takes ~30 seconds to wake up
4. **CORS must be configured** to allow your Netlify domain to access the backend

---

## 💡 Why This Architecture?

```
┌─────────────────┐
│   User Browser  │
└────────┬────────┘
         │
         │ (loads HTML/CSS/JS)
         ▼
┌─────────────────┐
│  Netlify.app    │  ← Frontend Only (Static Files)
│  (Frontend)     │
└────────┬────────┘
         │
         │ (API calls via fetch)
         ▼
┌─────────────────┐
│  Render.com     │  ← Backend (Python/FastAPI)
│  (Backend API)  │
└─────────────────┘
```

Netlify serves your React app, which then makes API calls to your backend hosted on Render.

---

## 🆘 Need Help?

Read the full `DEPLOYMENT.md` guide or check:
- Netlify Docs: https://docs.netlify.com
- Render Docs: https://render.com/docs
