# 🚀 Quick Deployment Commands

## Prerequisites Check

```powershell
# Check if Node.js is installed
node --version
# Should show: v18.x.x or higher

# Check if npm is installed
npm --version
# Should show: 9.x.x or higher

# Check if git is installed
git --version
# Should show: git version x.x.x
```

---

## 📦 Prepare for Deployment

### 1. Update Backend URL in Frontend

Edit `frontend/.env.production`:
```env
VITE_API_URL=https://your-actual-backend-url.onrender.com
```

### 2. Commit All Changes

```powershell
# Check what files changed
git status

# Stage all changes
git add .

# Commit
git commit -m "Add Netlify deployment configuration"

# Push to GitHub
git push origin master
```

---

## 🔙 Backend Deployment (Render.com)

### Via Render Dashboard (Recommended)
1. Go to: https://render.com
2. Click: "New +" → "Web Service"
3. Connect: Your GitHub repository
4. Configure:
   - **Name**: `movie-recommender-backend`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variables:
   - `API_HOST=0.0.0.0`
   - `API_PORT=$PORT`
6. Click: "Create Web Service"
7. Wait for deployment (5-10 minutes)
8. Copy your backend URL

### Test Backend
```powershell
# Replace with your actual backend URL
# Visit in browser:
https://your-backend-url.onrender.com/docs

# Or use curl:
curl https://your-backend-url.onrender.com/api/health
```

---

## 🎨 Frontend Deployment (Netlify)

### Via Netlify Dashboard (Recommended)
1. Go to: https://app.netlify.com
2. Click: "Add new site" → "Import an existing project"
3. Choose: "Deploy with GitHub"
4. Select: Your repository
5. Configure:
   - **Base directory**: `frontend`
   - **Build command**: `npm install && npm run build`
   - **Publish directory**: `frontend/dist`
6. Add Environment Variable:
   - **Key**: `VITE_API_URL`
   - **Value**: Your Render backend URL
7. Click: "Deploy site"
8. Wait for deployment (2-5 minutes)
9. Copy your Netlify URL

### Via Netlify CLI (Alternative)

```powershell
# Install Netlify CLI
npm install -g netlify-cli

# Login to Netlify
netlify login

# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Build the project
npm run build

# Deploy to production
netlify deploy --prod --dir=dist

# Follow the prompts to create a new site or link to existing
```

### Test Frontend
```powershell
# Visit your Netlify URL in browser:
https://your-app-name.netlify.app
```

---

## 🔄 Update CORS After Frontend Deployment

### Method 1: Via Render Dashboard
1. Go to Render dashboard
2. Select your backend service
3. Click "Environment"
4. Edit `CORS_ORIGINS`:
   ```
   CORS_ORIGINS=http://localhost:5173,http://localhost:3000,https://your-app.netlify.app,https://*.netlify.app
   ```
5. Save and redeploy

### Method 2: Via .env file
1. Create `backend/.env` file:
   ```env
   CORS_ORIGINS=http://localhost:5173,http://localhost:3000,https://your-app.netlify.app,https://*.netlify.app
   ```
2. Commit and push:
   ```powershell
   git add backend/.env
   git commit -m "Update CORS for production"
   git push origin master
   ```

---

## 🧪 Testing Commands

### Test Backend Health
```powershell
# Replace with your backend URL
curl https://your-backend-url.onrender.com/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {...}
}
```

### Test Backend Recommendations
```powershell
# Get recommendations for user 1
curl https://your-backend-url.onrender.com/api/recommend/1?n=5
```

### Test Search Endpoint
```powershell
# Search for movies
curl -X POST https://your-backend-url.onrender.com/api/search/description \
  -H "Content-Type: application/json" \
  -d '{"query":"sci-fi thriller","n_items":5}'
```

### Check Frontend in Browser
1. Open: `https://your-app.netlify.app`
2. Open DevTools: Press `F12`
3. Check Console for errors
4. Check Network tab for API calls

---

## 🔧 Local Development Commands

### Run Backend Locally
```powershell
# From project root
cd backend

# Install dependencies (first time only)
pip install -r requirements.txt

# Run server
python -m uvicorn api.main:app --reload --port 8000

# Or with explicit path
uvicorn backend.api.main:app --reload --port 8000
```

Backend will be available at: http://localhost:8000

### Run Frontend Locally
```powershell
# From project root
cd frontend

# Install dependencies (first time only)
npm install

# Run dev server
npm run dev
```

Frontend will be available at: http://localhost:5173

### Run Both with Docker
```powershell
# From project root
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 🔄 Redeployment Commands

### Redeploy Backend (Render)
```powershell
# Commit and push changes
git add .
git commit -m "Update backend"
git push origin master

# Render automatically redeploys on git push
```

### Redeploy Frontend (Netlify)
```powershell
# Commit and push changes
git add .
git commit -m "Update frontend"
git push origin master

# Netlify automatically redeploys on git push
```

### Manual Trigger
- **Render**: Dashboard → "Manual Deploy" → "Deploy latest commit"
- **Netlify**: Dashboard → "Trigger deploy" → "Deploy site"

---

## 🗑️ Rollback Commands

### Rollback on Render
1. Go to Render dashboard
2. Select your service
3. Click "Events" tab
4. Find previous successful deployment
5. Click "Redeploy"

### Rollback on Netlify
1. Go to Netlify dashboard
2. Select your site
3. Click "Deploys" tab
4. Find previous successful deployment
5. Click "..." → "Publish deploy"

### Rollback via Git
```powershell
# View commit history
git log --oneline

# Revert to specific commit
git revert <commit-hash>

# Or reset (careful! this rewrites history)
git reset --hard <commit-hash>
git push --force origin master
```

---

## 📊 Monitoring Commands

### Check Deployment Status

#### Render
```powershell
# Visit dashboard or check logs
https://dashboard.render.com
```

#### Netlify
```powershell
# Via CLI
netlify status

# Check build logs
netlify open --site
```

### View Logs

#### Render Logs
1. Go to Render dashboard
2. Select your service
3. Click "Logs" tab
4. View real-time logs

#### Netlify Logs
```powershell
# Via CLI
netlify watch

# Via browser
netlify open --site
# Then click "Deploy log"
```

---

## 🆘 Troubleshooting Commands

### Clear Build Cache (Netlify)
```powershell
# Via dashboard
# Site Settings → Build & deploy → Clear cache and deploy site
```

### Restart Service (Render)
1. Go to Render dashboard
2. Select your service
3. Click "Manual Deploy" → "Clear build cache & deploy"

### Check Environment Variables

#### Backend (Render)
```powershell
# Check in dashboard: Environment tab
```

#### Frontend (Netlify)
```powershell
# Via CLI
netlify env:list

# Via dashboard
# Site settings → Environment variables
```

### Debug Frontend Build
```powershell
cd frontend
npm install
npm run build
# Check for errors
```

### Debug Backend Startup
```powershell
cd backend
pip install -r requirements.txt
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
# Check for errors
```

---

## 📝 Quick Reference

### Important URLs
- **Netlify Dashboard**: https://app.netlify.com
- **Render Dashboard**: https://dashboard.render.com
- **Your Frontend**: https://your-app.netlify.app
- **Your Backend**: https://your-backend.onrender.com
- **Backend API Docs**: https://your-backend.onrender.com/docs

### Environment Variable Names
- Frontend: `VITE_API_URL` (must start with VITE_)
- Backend: `CORS_ORIGINS`, `API_HOST`, `API_PORT`

### Build Commands
- Frontend: `npm install && npm run build`
- Backend: `pip install -r requirements.txt`

### Start Commands
- Frontend: Served as static files (no start command)
- Backend: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

---

## ✅ Verification Checklist

Run these commands to verify deployment:

```powershell
# 1. Check backend health
curl https://your-backend-url.onrender.com/api/health

# 2. Check backend API docs (open in browser)
start https://your-backend-url.onrender.com/docs

# 3. Check frontend (open in browser)
start https://your-app.netlify.app

# 4. Test API integration (in browser console)
# Open your frontend, then:
# F12 → Console → Run:
# fetch('/api/health').then(r => r.json()).then(console.log)
```

All checks passed? 🎉 Your app is live!

---

## 🎓 Learn More

- **Vite Build**: https://vitejs.dev/guide/build.html
- **FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/
- **Netlify Docs**: https://docs.netlify.com
- **Render Docs**: https://render.com/docs

---

**Remember**: Always deploy backend FIRST, then frontend! 🎯
