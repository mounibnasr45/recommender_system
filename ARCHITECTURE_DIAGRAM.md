# 🏗️ Deployment Architecture Diagram

## Current Local Setup (Before Deployment)

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Computer (Local)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐              ┌─────────────────┐       │
│  │   Frontend      │              │   Backend       │       │
│  │   (React App)   │◄────────────►│   (FastAPI)     │       │
│  │                 │   API Calls  │                 │       │
│  │ localhost:5173  │              │ localhost:8000  │       │
│  └─────────────────┘              └─────────────────┘       │
│         │                                  │                │
│         │                                  │                │
│         ▼                                  ▼                │
│  User's Browser                    ML Models + Data         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Production Setup (After Deployment)

```
┌──────────────────────────────────────────────────────────────────┐
│                         PRODUCTION                               │
└──────────────────────────────────────────────────────────────────┘

        User's Browser (Anywhere in the world)
               │
               │ (1) Visits website
               │
               ▼
┌────────────────────────────────────────┐
│         Netlify.app (CDN)              │  ← Frontend Hosting
│  ┌──────────────────────────────────┐  │     (Global CDN)
│  │  Your React App (Static Files)   │  │
│  │  - HTML, CSS, JavaScript         │  │
│  │  - Served from closest location  │  │
│  │  - https://your-app.netlify.app  │  │
│  └──────────────────────────────────┘  │
└────────────────┬───────────────────────┘
                 │
                 │ (2) API Requests
                 │     (fetch calls to backend)
                 │
                 ▼
┌────────────────────────────────────────┐
│       Render.com (Server)              │  ← Backend Hosting
│  ┌──────────────────────────────────┐  │     (Single Server)
│  │  FastAPI Server (Python)         │  │
│  │  - Runs Python code              │  │
│  │  - Processes requests            │  │
│  │  - Executes ML models            │  │
│  │  - Returns JSON responses        │  │
│  │  - https://backend.onrender.com  │  │
│  └──────────┬───────────────────────┘  │
│             │                           │
│             ▼                           │
│  ┌──────────────────────────────────┐  │
│  │  Data & Models                   │  │
│  │  - movies.csv                    │  │
│  │  - ratings.csv                   │  │
│  │  - trained_models/               │  │
│  │  - embeddings.pkl                │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

---

## Request Flow

```
┌─────────────┐
│    User     │
│  (Browser)  │
└──────┬──────┘
       │
       │ ① Types URL: https://your-app.netlify.app
       │
       ▼
┌─────────────────────────┐
│    Netlify CDN          │
│                         │
│  - Serves index.html    │
│  - Serves app.js        │
│  - Serves styles.css    │
└───────────┬─────────────┘
            │
            │ ② React app loads in browser
            │
            ▼
┌─────────────────────────┐
│   React App Running     │
│   in User's Browser     │
│                         │
│  User clicks:           │
│  "Get Recommendations"  │
└───────────┬─────────────┘
            │
            │ ③ JavaScript makes API call:
            │    fetch('https://backend.onrender.com/api/recommend/1')
            │
            ▼
┌─────────────────────────┐
│   Backend on Render     │
│                         │
│  - Receives request     │
│  - Loads ML model       │
│  - Calculates recs      │
│  - Returns JSON         │
└───────────┬─────────────┘
            │
            │ ④ Response: JSON with movies
            │    { recommendations: [...] }
            │
            ▼
┌─────────────────────────┐
│   React App             │
│                         │
│  - Receives response    │
│  - Renders movies       │
│  - Shows to user        │
└───────────┬─────────────┘
            │
            │ ⑤ User sees recommendations!
            │
            ▼
┌─────────────────────────┐
│   Happy User! 🎉        │
└─────────────────────────┘
```

---

## Why Two Separate Services?

```
┌──────────────────────────────────────────────────────────┐
│  Why can't everything be on Netlify?                     │
└──────────────────────────────────────────────────────────┘

Netlify:
✅ Excellent for: Static files (HTML, CSS, JS)
✅ Fast: Global CDN, instant loading
✅ Free: Generous free tier
✅ Simple: Just upload files
❌ Cannot: Run Python code
❌ Cannot: Execute ML models
❌ Cannot: Keep server running
❌ Cannot: Access databases in real-time

Render:
✅ Can run: Python/Node/Ruby/Go servers
✅ Can execute: ML models, computations
✅ Can connect: To databases
✅ Can keep: Server running 24/7
✅ Has: Free tier (with limitations)
⚠️  Slower: Single server location
⚠️  Not global: No CDN for backend

Perfect Together:
🎯 Netlify serves frontend fast (global CDN)
🎯 Render runs backend logic (server processing)
🎯 They communicate via API calls
🎯 Best of both worlds!
```

---

## File Distribution

```
┌─────────────────────────────────────────────────────────┐
│              Your GitHub Repository                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  frontend/                                               │
│  ├── src/                                                │
│  ├── package.json       ───────┐                        │
│  ├── vite.config.js             │                        │
│  └── .env.production            │                        │
│                                  │                        │
│  backend/                        │                        │
│  ├── api/                        │                        │
│  ├── models/                     │                        │
│  ├── services/                   │                        │
│  ├── requirements.txt   ─────┐  │                        │
│  └── config.py               │  │                        │
│                              │  │                        │
│  data/                       │  │                        │
│  ml-latest-small/            │  │                        │
│  netlify.toml       ─────────┼──┘                        │
│  render.yaml        ─────────┘                           │
│                                                          │
└─────┬────────────────────────┬───────────────────────────┘
      │                        │
      │                        │
      ▼                        ▼
┌──────────────┐        ┌──────────────┐
│   Netlify    │        │    Render    │
│              │        │              │
│ Builds:      │        │ Builds:      │
│ - frontend/  │        │ - backend/   │
│              │        │ - data/      │
│ Uses:        │        │              │
│ netlify.toml │        │ Uses:        │
│              │        │ render.yaml  │
│ Serves:      │        │              │
│ Static files │        │ Runs:        │
│              │        │ Python server│
└──────────────┘        └──────────────┘
```

---

## Environment Variables Flow

```
┌────────────────────────────────────────────────┐
│         Development (Local)                    │
├────────────────────────────────────────────────┤
│                                                │
│  Frontend uses:                                │
│  .env.development                              │
│  ├── VITE_API_URL=http://localhost:8000       │
│  │                                             │
│  └── Points to local backend                  │
│                                                │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│         Production (Deployed)                  │
├────────────────────────────────────────────────┤
│                                                │
│  Netlify uses:                                 │
│  Environment Variables in Dashboard            │
│  ├── VITE_API_URL=https://backend.onrender.com│
│  │                                             │
│  └── Points to production backend             │
│                                                │
│  Render uses:                                  │
│  Environment Variables in Dashboard            │
│  ├── API_HOST=0.0.0.0                         │
│  ├── API_PORT=$PORT (auto-provided)           │
│  └── CORS_ORIGINS=https://your-app.netlify.app│
│                                                │
└────────────────────────────────────────────────┘
```

---

## CORS Explained

```
┌──────────────────────────────────────────────────────────┐
│  Without CORS: Browser blocks the request ❌             │
└──────────────────────────────────────────────────────────┘

User visits: https://your-app.netlify.app
    │
    │ JavaScript tries to call:
    │ https://backend.onrender.com/api/recommend/1
    │
    ▼
┌─────────────────┐
│     Browser     │  🚫 "Blocked by CORS policy!"
│   (Security)    │  🚫 Different domains!
└─────────────────┘

┌──────────────────────────────────────────────────────────┐
│  With CORS: Browser allows the request ✅                │
└──────────────────────────────────────────────────────────┘

Backend config.py has:
CORS_ORIGINS = [
    "https://your-app.netlify.app"  ← Frontend domain
]

User visits: https://your-app.netlify.app
    │
    │ JavaScript calls:
    │ https://backend.onrender.com/api/recommend/1
    │
    ▼
┌─────────────────┐
│     Browser     │  📋 Checks CORS headers
└────────┬────────┘
         │
         │ "Is https://your-app.netlify.app allowed?"
         │
         ▼
┌─────────────────┐
│    Backend      │  ✅ "Yes! It's in CORS_ORIGINS"
│  (Render.com)   │  ✅ Sends response
└────────┬────────┘
         │
         │ Response sent back
         │
         ▼
┌─────────────────┐
│   Frontend      │  ✅ Receives data
│  (React App)    │  ✅ Shows movies
└─────────────────┘
```

---

## Deployment Timeline

```
Time: 0 min                      10 min                20 min
│                                │                     │
│  ① Deploy Backend              │  ③ Deploy Frontend  │  ⑤ Done!
│     to Render.com              │     to Netlify      │
│                                │                     │
├─────────────────────────────►  ├──────────────────►  ├────────►
│                                │                     │
│  - Upload code                 │  - Upload code      │  ✅ Live!
│  - Install dependencies        │  - Install deps     │
│  - Start server                │  - Build React app  │
│  - Get URL                     │  - Deploy to CDN    │
│                                │                     │
│  ② Update Frontend .env        │  ④ Update CORS      │
│     with backend URL           │     with Netlify URL│
│                                │                     │

Total time: ~15-20 minutes
```

---

## Scaling Comparison

```
┌────────────────────────────────────────────────┐
│           With Netlify + Render                │
├────────────────────────────────────────────────┤
│                                                │
│  1 user:     Fast ✅                           │
│  10 users:   Fast ✅                           │
│  100 users:  Frontend fast ✅                  │
│              Backend may slow ⚠️               │
│  1000 users: Frontend fast ✅                  │
│              Backend overloaded 🔥             │
│                                                │
│  Solution: Upgrade Render tier                 │
│            Add caching                         │
│            Use database                        │
│            Add load balancer                   │
│                                                │
└────────────────────────────────────────────────┘
```

---

## Cost Breakdown Over Time

```
Month 1-12: Development & Portfolio
├── Netlify: $0 (Free tier)
├── Render:  $0 (Free tier, with sleeping)
└── Total:   $0/month 🎉

Month 13+: Growing User Base
├── Netlify: $0 (Free tier still fine)
├── Render:  $7 (Starter tier, 24/7 uptime)
└── Total:   $7/month

Scale Up: Production App
├── Netlify: $19 (Pro tier)
├── Render:  $25-85 (Standard tier + database)
└── Total:   $44-104/month
```

---

## Summary Diagram

```
┌───────────────────────────────────────────────────┐
│           Your Movie Recommender System           │
├───────────────────────────────────────────────────┤
│                                                   │
│  Frontend (Netlify)           Backend (Render)    │
│  ┌────────────────┐           ┌────────────────┐ │
│  │ React App      │ ◄────────► │ FastAPI Server│ │
│  │ - HTML/CSS/JS  │  API Calls │ - Python Code │ │
│  │ - Static Files │            │ - ML Models   │ │
│  │ - Global CDN   │            │ - Database    │ │
│  └────────────────┘            └────────────────┘ │
│         │                             │           │
│         │                             │           │
│         ▼                             ▼           │
│  Users worldwide              Processing Power    │
│  (Fast loading)               (Heavy computation) │
│                                                   │
└───────────────────────────────────────────────────┘

Key Insight: Separate concerns for optimal performance!
- Static content → CDN (Netlify)
- Dynamic logic → Server (Render)
```

---

This architecture ensures:
- ⚡ Fast frontend loading (CDN)
- 🚀 Scalable backend (server-side)
- 💰 Cost-effective (free tiers available)
- 🔧 Easy maintenance (separate deployments)
- 🌍 Global reach (Netlify CDN)

---

**Remember**: Frontend serves the interface, Backend does the thinking! 🧠
