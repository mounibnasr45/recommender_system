# 🚀 Deploying Movie Recommender System to Netlify

This guide will help you deploy your Movie Recommender System with the frontend on Netlify and backend on a separate service.

## 📋 Prerequisites

- [ ] GitHub account
- [ ] Netlify account (free tier available)
- [ ] Backend hosting account (Render.com recommended for free tier)

---

## Part 1️⃣: Deploy Backend to Render.com (or similar)

### Why can't the backend be on Netlify?
Netlify only hosts **static sites** (HTML, CSS, JavaScript). Your FastAPI backend needs a service that can run Python server applications.

### Recommended Option: Render.com (Free Tier)

1. **Create Account**: Go to https://render.com and sign up

2. **Connect Repository**: 
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect your backend

3. **Configure Service**:
   ```
   Name: movie-recommender-backend
   Region: Choose closest to your users
   Branch: master (or main)
   Root Directory: backend
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn api.main:app --host 0.0.0.0 --port $PORT
   ```

4. **Environment Variables** (Add in Render dashboard):
   ```
   API_HOST=0.0.0.0
   API_PORT=$PORT
   ```

5. **Deploy**: Click "Create Web Service"
   - Wait 5-10 minutes for first deployment
   - Note your backend URL: `https://movie-recommender-backend.onrender.com`

6. **Important Notes**:
   - Free tier sleeps after 15 minutes of inactivity
   - First request after sleep takes ~30 seconds to wake up
   - Consider upgrading for production use

---

## Part 2️⃣: Deploy Frontend to Netlify

### Step 1: Prepare Your Repository

1. **Update `.env.production`** with your backend URL:
   ```env
   VITE_API_URL=https://movie-recommender-backend.onrender.com
   ```

2. **Commit your changes**:
   ```powershell
   git add .
   git commit -m "Add Netlify deployment configuration"
   git push origin master
   ```

### Step 2: Deploy on Netlify

#### Option A: Deploy via Git (Recommended)

1. **Login to Netlify**: https://app.netlify.com

2. **Import Project**:
   - Click "Add new site" → "Import an existing project"
   - Choose "Deploy with GitHub"
   - Authorize Netlify to access your repository
   - Select your repository

3. **Configure Build Settings**:
   ```
   Base directory: frontend
   Build command: npm install && npm run build
   Publish directory: frontend/dist
   ```

4. **Add Environment Variables**:
   - Go to Site settings → Environment variables
   - Add variable:
     - Key: `VITE_API_URL`
     - Value: `https://movie-recommender-backend.onrender.com`
     - (Replace with your actual backend URL)

5. **Deploy**:
   - Click "Deploy site"
   - Wait 2-3 minutes for build
   - Your site will be live at: `https://random-name-12345.netlify.app`

6. **Custom Domain** (Optional):
   - Go to Site settings → Domain management
   - Add custom domain or change Netlify subdomain

#### Option B: Deploy via Netlify CLI

```powershell
# Install Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Navigate to frontend
cd frontend

# Build the project
npm install
npm run build

# Deploy
netlify deploy --prod --dir=dist
```

---

## Part 3️⃣: Update Backend CORS Settings

Your backend needs to allow requests from your Netlify domain.

**Edit `backend/config.py`**:
```python
# Add your Netlify URL to CORS_ORIGINS
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://your-app-name.netlify.app",  # Add this
    "https://*.netlify.app",  # Allow all Netlify preview URLs
]
```

**Redeploy backend** after making this change.

---

## 🔍 Testing Your Deployment

1. **Test Backend**:
   - Visit: `https://your-backend-url.onrender.com/docs`
   - Try the `/api/health` endpoint
   - Should return: `{"status": "healthy", ...}`

2. **Test Frontend**:
   - Visit: `https://your-app-name.netlify.app`
   - Check browser console for errors
   - Try getting recommendations for a user

3. **Test API Connection**:
   - Open browser DevTools → Network tab
   - Try an action that calls the API
   - Check if requests go to correct backend URL

---

## 🐛 Troubleshooting

### Frontend shows "Failed to fetch" errors
- **Check**: Backend URL in Netlify environment variables
- **Check**: Backend CORS settings include your Netlify URL
- **Check**: Backend is awake (visit `/docs` endpoint)

### Backend returns 404 for all API calls
- **Check**: API routes start with `/api` (configured in backend)
- **Check**: Frontend API calls include `/api` prefix

### Environment variables not working
- **Solution**: Rebuild site after adding variables
- **Check**: Variables start with `VITE_` prefix (required for Vite)
- **Access in code**: `import.meta.env.VITE_API_URL`

### Build fails on Netlify
- **Check**: `package.json` has correct dependencies
- **Check**: Node version compatibility (specified in `netlify.toml`)
- **Check**: Build command is correct
- **View**: Build logs in Netlify dashboard

### Backend sleeps on free tier (Render)
- **Expected**: Free tier sleeps after 15 minutes inactivity
- **First request**: Takes ~30 seconds to wake up
- **Solution**: 
  - Upgrade to paid tier for 24/7 uptime
  - Or use a ping service to keep it awake
  - Or accept the wake-up delay

---

## 💰 Cost Summary

| Service | Tier | Cost | Limits |
|---------|------|------|--------|
| **Netlify** | Free | $0 | 100GB bandwidth/month |
| **Render** | Free | $0 | Sleeps after 15 min inactivity |
| **Total** | Free | **$0/month** | Good for demo/portfolio |

### Recommended Upgrades for Production:
- **Netlify Pro**: $19/month (custom domains, more bandwidth)
- **Render Starter**: $7/month (24/7 uptime, no sleeping)

---

## 📚 Alternative Backend Hosting Options

### Railway.app (Free Tier)
- Similar to Render
- 500 hours/month free
- $5 credit per month
- Setup similar to Render

### Heroku (Paid Only - since Nov 2022)
- Starts at $7/month
- More mature platform
- Better documentation

### AWS/Azure/GCP
- More complex setup
- Pay-as-you-go pricing
- Better for large-scale production

### DigitalOcean App Platform
- Starts at $5/month
- Good performance
- Simple deployment

---

## ✅ Deployment Checklist

- [ ] Backend deployed to Render/Railway/other
- [ ] Backend URL noted
- [ ] Backend CORS configured with Netlify domain
- [ ] Frontend `.env.production` updated with backend URL
- [ ] Repository pushed to GitHub
- [ ] Netlify connected to repository
- [ ] Netlify build settings configured
- [ ] Netlify environment variable added (VITE_API_URL)
- [ ] Site deployed successfully
- [ ] Backend `/docs` endpoint accessible
- [ ] Frontend loads without errors
- [ ] API calls work from frontend to backend
- [ ] Test recommendations for a user

---

## 🎉 Next Steps

1. **Custom Domain**: Add a custom domain in Netlify settings
2. **SSL Certificate**: Netlify provides free SSL automatically
3. **Analytics**: Enable Netlify Analytics or add Google Analytics
4. **Monitoring**: Set up uptime monitoring for backend
5. **CI/CD**: Configure automatic deployments on git push

---

## 📞 Need Help?

- **Netlify Docs**: https://docs.netlify.com
- **Render Docs**: https://render.com/docs
- **FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/

Good luck with your deployment! 🚀
