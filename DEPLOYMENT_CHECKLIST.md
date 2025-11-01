# 📋 Netlify Deployment Checklist

## ✅ Files Created & Ready

### Configuration Files
- [x] `netlify.toml` - Netlify build configuration
- [x] `backend/render.yaml` - Backend deployment configuration for Render
- [x] `frontend/.env.development` - Development environment variables
- [x] `frontend/.env.production` - Production environment variables (needs backend URL)
- [x] `backend/.env.example` - Backend environment variables template
- [x] `.gitignore` - Ignore unnecessary files

### Documentation
- [x] `DEPLOYMENT.md` - Complete deployment guide
- [x] `NETLIFY_README.md` - Quick start guide
- [x] `DEPLOYMENT_CHECKLIST.md` - This file!

### Code Updates
- [x] `frontend/src/utils/api.js` - Updated to use environment variables
- [x] `backend/config.py` - Updated CORS to support environment variables
- [x] `backend/api/main.py` - Updated to use new CORS configuration

---

## 🚀 Deployment Steps

### Phase 1: Prepare Backend ⚠️ (Required First)

#### 1.1 Choose Backend Hosting Platform
- [ ] Create account on one of these:
  - [ ] **Render.com** (Recommended - Free tier, easy setup)
  - [ ] **Railway.app** (Alternative - $5 free credit)
  - [ ] **Heroku** (Paid only)
  - [ ] **AWS/Azure/GCP** (Advanced)

#### 1.2 Deploy Backend to Render.com
- [ ] Sign up at https://render.com
- [ ] Click "New +" → "Web Service"
- [ ] Connect GitHub repository
- [ ] Configure service:
  ```
  Name: movie-recommender-backend
  Root Directory: backend
  Runtime: Python 3
  Build Command: pip install -r requirements.txt
  Start Command: uvicorn api.main:app --host 0.0.0.0 --port $PORT
  ```
- [ ] Set environment variables in Render dashboard:
  - `API_HOST=0.0.0.0`
  - `API_PORT=$PORT`
- [ ] Click "Create Web Service" and wait for deployment
- [ ] **IMPORTANT**: Note your backend URL (e.g., `https://movie-recommender-backend.onrender.com`)
- [ ] Test backend by visiting: `https://your-backend-url.onrender.com/docs`

---

### Phase 2: Configure Frontend

#### 2.1 Update Production Environment
- [ ] Edit `frontend/.env.production`
- [ ] Replace `https://your-backend-url-here.com` with your actual Render URL
  ```env
  VITE_API_URL=https://movie-recommender-backend.onrender.com
  ```

#### 2.2 Update Backend CORS
- [ ] Edit `backend/config.py` or create `backend/.env`
- [ ] Add your Netlify URL to CORS_ORIGINS (you'll update this again after Netlify deployment)
  ```
  CORS_ORIGINS=http://localhost:5173,http://localhost:3000,https://*.netlify.app
  ```
- [ ] Commit and push changes to trigger backend redeployment

#### 2.3 Commit Changes
- [ ] Stage all changes:
  ```powershell
  git add .
  ```
- [ ] Commit:
  ```powershell
  git commit -m "Add Netlify deployment configuration"
  ```
- [ ] Push to GitHub:
  ```powershell
  git push origin master
  ```

---

### Phase 3: Deploy Frontend to Netlify

#### 3.1 Setup Netlify Account
- [ ] Sign up at https://app.netlify.com
- [ ] Authorize Netlify to access your GitHub account

#### 3.2 Import Project
- [ ] Click "Add new site" → "Import an existing project"
- [ ] Choose "Deploy with GitHub"
- [ ] Select your repository

#### 3.3 Configure Build Settings
- [ ] Base directory: `frontend`
- [ ] Build command: `npm install && npm run build`
- [ ] Publish directory: `frontend/dist`

#### 3.4 Add Environment Variables
- [ ] Go to Site settings → Build & deploy → Environment variables
- [ ] Click "Add a variable"
- [ ] Add:
  - Key: `VITE_API_URL`
  - Value: Your backend URL from Phase 1 (e.g., `https://movie-recommender-backend.onrender.com`)

#### 3.5 Deploy
- [ ] Click "Deploy site"
- [ ] Wait 2-5 minutes for build to complete
- [ ] **IMPORTANT**: Note your Netlify URL (e.g., `https://random-name-12345.netlify.app`)

#### 3.6 Custom Domain (Optional)
- [ ] Go to Site settings → Domain management
- [ ] Click "Options" → "Edit site name"
- [ ] Change to something memorable (e.g., `movie-recommender-app`)
- [ ] Or add a custom domain if you own one

---

### Phase 4: Update Backend CORS (Again)

#### 4.1 Add Netlify URL to Backend CORS
- [ ] Go back to your Render.com dashboard
- [ ] Find your backend service
- [ ] Go to Environment → Environment Variables
- [ ] Update `CORS_ORIGINS` to include your Netlify URL:
  ```
  CORS_ORIGINS=http://localhost:5173,http://localhost:3000,https://your-app.netlify.app,https://*.netlify.app
  ```
- [ ] Save and redeploy backend

---

### Phase 5: Testing

#### 5.1 Test Backend
- [ ] Visit: `https://your-backend-url.onrender.com/docs`
- [ ] Should see FastAPI documentation page
- [ ] Click "Try it out" on `/api/health` endpoint
- [ ] Should return: `{"status": "healthy", "model_loaded": true, ...}`

#### 5.2 Test Frontend
- [ ] Visit: `https://your-app.netlify.app`
- [ ] Page should load without errors
- [ ] Open browser DevTools (F12) → Console tab
- [ ] Check for any errors (red text)

#### 5.3 Test API Connection
- [ ] On frontend, try entering a user ID
- [ ] Click "Get Recommendations"
- [ ] Should see movie recommendations
- [ ] Check DevTools → Network tab
- [ ] API calls should go to your backend URL (green status 200)

#### 5.4 Test CORS
- [ ] If you see errors like "CORS policy blocked"
- [ ] Double-check backend CORS settings include Netlify URL
- [ ] Redeploy backend after CORS changes

---

## 🐛 Common Issues & Solutions

### Issue: "Failed to fetch" errors
**Solution:**
- Check backend URL in Netlify environment variables
- Ensure backend is running (visit `/docs` endpoint)
- Check browser console for exact error

### Issue: CORS errors
**Solution:**
- Add Netlify URL to backend CORS_ORIGINS
- Redeploy backend
- Clear browser cache and try again

### Issue: Build fails on Netlify
**Solution:**
- Check build logs in Netlify dashboard
- Ensure `package.json` has all dependencies
- Verify build command is correct
- Check Node version compatibility

### Issue: Environment variables not working
**Solution:**
- Variables must start with `VITE_` for Vite
- Rebuild site after adding variables
- Check spelling of variable names

### Issue: Backend sleeps (Render free tier)
**Solution:**
- Expected behavior on free tier
- First request takes ~30 seconds
- Consider upgrading to paid tier for 24/7 uptime
- Or use a ping service to keep awake

### Issue: 404 on API routes
**Solution:**
- Ensure API routes start with `/api`
- Check backend is deployed correctly
- Verify frontend API calls include correct path

---

## 📊 Deployment Status

### Backend Status
- [ ] Deployed to: __________________ (e.g., Render.com)
- [ ] Backend URL: __________________ 
- [ ] `/docs` endpoint accessible: Yes / No
- [ ] `/api/health` returns healthy: Yes / No

### Frontend Status
- [ ] Deployed to Netlify: Yes / No
- [ ] Netlify URL: __________________
- [ ] Page loads without errors: Yes / No
- [ ] API calls successful: Yes / No

### Configuration Status
- [ ] Backend URL added to frontend .env.production: Yes / No
- [ ] Netlify URL added to backend CORS: Yes / No
- [ ] Environment variables set in Netlify: Yes / No
- [ ] All changes committed to Git: Yes / No

---

## 💰 Cost Estimate

| Service | Tier | Monthly Cost | Notes |
|---------|------|--------------|-------|
| Netlify | Free | $0 | 100GB bandwidth |
| Render | Free | $0 | Sleeps after 15 min inactivity |
| **Total** | **Free** | **$0** | Perfect for portfolio/demo |

### Upgrade Options:
- Netlify Pro: $19/month (more bandwidth, analytics)
- Render Starter: $7/month (24/7 uptime, no sleeping)

---

## ✅ Final Checklist

- [ ] Backend deployed and accessible
- [ ] Frontend deployed and accessible
- [ ] Environment variables configured
- [ ] CORS configured correctly
- [ ] All tests passed
- [ ] No console errors
- [ ] Recommendations work
- [ ] Search functionality works
- [ ] Custom domain configured (optional)
- [ ] SSL certificate active (automatic on Netlify)

---

## 🎉 Success!

Once all checkboxes are complete, your Movie Recommender System is live!

**Share your links:**
- Frontend: `https://your-app.netlify.app`
- Backend API: `https://your-backend.onrender.com/docs`

**Next Steps:**
- Add to portfolio
- Share on LinkedIn/Twitter
- Add Google Analytics
- Set up monitoring
- Consider upgrading for production use

---

## 📞 Support Resources

- **Netlify Docs**: https://docs.netlify.com
- **Render Docs**: https://render.com/docs
- **FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/
- **Vite Env Variables**: https://vitejs.dev/guide/env-and-mode.html

Good luck! 🚀
