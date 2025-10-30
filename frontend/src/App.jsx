import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [userId, setUserId] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState('');
  const [modelStatus, setModelStatus] = useState(null);
  const [checkingHealth, setCheckingHealth] = useState(true);
  const [activeTab, setActiveTab] = useState('recommendations');
  const [mobileOpen, setMobileOpen] = useState(false);
  const tabsRef = useRef(null);

  // UI: show/hide ratings & history after getting recommendations
  const [showDetails, setShowDetails] = useState(false);
  const [userRatings, setUserRatings] = useState([]);
  const [userHistory, setUserHistory] = useState([]);
  const [loadingDetails, setLoadingDetails] = useState(false);

  // User authentication state
  const [currentUser, setCurrentUser] = useState(null);
  const [showAuth, setShowAuth] = useState(false);
  const [authMode, setAuthMode] = useState('login'); // 'login' or 'register'
  const [authForm, setAuthForm] = useState({
    username: '',
    email: '',
    password: ''
  });

  // Rating state
  const [ratingMovie, setRatingMovie] = useState(null);
  const [userRating, setUserRating] = useState(0);

  useEffect(() => {
    checkHealth();
    checkCurrentUser();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await fetch('/api/health');
      const data = await response.json();
      setModelStatus(data);
    } catch (err) {
      setError('Failed to connect to backend');
    } finally {
      setCheckingHealth(false);
    }
  };

  const checkCurrentUser = () => {
    const savedUser = localStorage.getItem('currentUser');
    if (savedUser) {
      try {
        setCurrentUser(JSON.parse(savedUser));
      } catch (err) {
        localStorage.removeItem('currentUser');
      }
    }
  };

  const handleAuthSubmit = async (e) => {
    e.preventDefault();
    setError('');

    try {
      const endpoint = authMode === 'login' ? '/api/users/login' : '/api/users/register';
      const body = authMode === 'login'
        ? { username: authForm.username, password: authForm.password }
        : authForm;

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Authentication failed');
      }

      setCurrentUser(data.user);
      localStorage.setItem('currentUser', JSON.stringify(data.user));
      setShowAuth(false);
      setAuthForm({ username: '', email: '', password: '' });
      setError('Welcome! You can now rate movies and get personalized recommendations.');

    } catch (err) {
      setError(err.message);
    }
  };

  const handleLogout = () => {
    setCurrentUser(null);
    localStorage.removeItem('currentUser');
    setRecommendations([]);
    setError('Logged out successfully');
  };

  // Switch tab helper that closes mobile menu
  const switchTab = async (tab) => {
    setActiveTab(tab);
    setMobileOpen(false);
    // if opening recommendations, fetch details lazily
    if (tab === 'recommendations' && !recommendations.length) {
      // nothing automatic — user must request recommendations
    }
    // when opening activity tab, load user's ratings & history
    if (tab === 'activity' && (currentUser || userId)) {
      try {
        await fetchDetails();
      } catch (e) {
        console.error('Failed to load activity on tab switch', e);
      }
    }
  };

  const handleRateMovie = async (movieId, rating) => {
    if (!currentUser) {
      setError('Please login to rate movies');
      return;
    }

    try {
      const response = await fetch(`/api/users/${currentUser.user_id}/ratings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ movie_id: movieId, rating: rating }),
      });

      if (!response.ok) {
        throw new Error('Failed to save rating');
      }

      setError('Rating saved! The model will be updated with your preferences.');
      setRatingMovie(null);
      setUserRating(0);

    } catch (err) {
      setError(err.message);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const targetUserId = currentUser ? currentUser.user_id : userId;

    if (!targetUserId && !currentUser) {
      setError('Please enter a user ID or login');
      return;
    }

    setLoading(true);
    setError('');
    setRecommendations([]);

    try {
      const response = await fetch(`/api/recommend/${targetUserId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setRecommendations(data.recommendations || []);
      // Reset details panel visibility when new recommendations arrive
      setShowDetails(false);
      setUserRatings([]);
      setUserHistory([]);
      // If we have a logged-in user, also fetch their details so activity shows up
      if (currentUser) {
        try {
          await fetchDetails();
        } catch (err) {
          console.warn('Failed to fetch details after recommendations', err);
        }
      }
    } catch (err) {
      setError(`Failed to get recommendations: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchDetails = async () => {
    // Load user's ratings and history when toggling the details panel open
    const targetUserId = currentUser ? currentUser.user_id : userId;
    if (!targetUserId) {
      setError('No user selected to fetch details');
      return;
    }

    setLoadingDetails(true);
    setError('');
    try {
      // Ratings endpoint
      const ratingsResp = await fetch(`/api/users/${targetUserId}/ratings`);
      if (ratingsResp.ok) {
        const ratingsData = await ratingsResp.json();
        console.log('[Frontend] fetched user ratings:', ratingsData);
        setUserRatings(ratingsData.ratings || []);
      } else {
        setUserRatings([]);
      }

      // History endpoint (may require model to be loaded)
      const historyResp = await fetch(`/api/history/${targetUserId}`);
      if (historyResp.ok) {
        const historyData = await historyResp.json();
        console.log('[Frontend] fetched user history:', historyData);
        setUserHistory(historyData.history || []);
      } else {
        setUserHistory([]);
      }
    } catch (err) {
      setError(`Failed to load details: ${err.message}`);
    } finally {
      setLoadingDetails(false);
    }
  };

  const handleTrain = async () => {
    setTraining(true);
    setError('');

    try {
      const response = await fetch('/api/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ force_retrain: true }),
      });

      const data = await response.json();

      if (data.status === 'success') {
        setError('Model trained successfully! You can now get recommendations.');
        await checkHealth(); // Refresh model status
      } else {
        setError(`Training failed: ${data.message}`);
      }
    } catch (err) {
      setError(`Training failed: ${err.message}`);
    } finally {
      setTraining(false);
    }
  };

  const handleDescriptionSearch = async () => {
    if (!searchQuery.trim()) return;

    console.log('[Frontend] Starting description search for:', searchQuery);
    setSearchLoading(true);
    setSearchResults([]);
    try {
      const response = await fetch('/api/search/description', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery.trim(),
          limit: 10
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        console.error('[Frontend] Search failed - status:', response.status, 'body:', text);
        throw new Error('Search failed');
      }

  const data = await response.json();
  console.log('[Frontend] Search response:', data);
  // Accept either 'results' (frontend convention) or 'movies' (backend key)
  const results = data.results || data.movies || [];
      if (results.length === 0) {
        console.warn('[Frontend] No search results returned for query:', searchQuery);
      }
      setSearchResults(results);
    } catch (error) {
      console.error('Error searching by description:', error);
      setSearchResults([]);
    } finally {
      setSearchLoading(false);
    }
  };

  if (checkingHealth) {
    return (
      <div className="app-root">
        <div className="splash">Checking system status...</div>
      </div>
    );
  }

  return (
    <div className="app-root">
      <header className="site-header">
        <div className="header-inner">
          <div className="brand" onClick={() => switchTab('recommendations')}>
            <div className="logo">🎬</div>
            <div className="brand-title"><span className="gradient-text">CineMatch</span></div>
          </div>

          <nav className={`main-nav ${mobileOpen ? 'open' : ''}`} ref={tabsRef}>
            <button className={`tab-btn ${activeTab === 'recommendations' ? 'active' : ''}`} onClick={() => switchTab('recommendations')}>Recommendations</button>
            <button className={`tab-btn ${activeTab === 'search' ? 'active' : ''}`} onClick={() => switchTab('search')}>Search</button>
            <button className={`tab-btn ${activeTab === 'activity' ? 'active' : ''}`} onClick={() => switchTab('activity')}>My Activity</button>
          </nav>

          <div className="header-actions">
            {modelStatus && (
              <div className={`status-badge ${modelStatus.model_loaded ? 'ready' : 'loading'}`}>
                <span className={`dot ${modelStatus.model_loaded ? 'green' : 'yellow'}`}></span>
                {modelStatus.model_loaded ? 'Ready' : 'Training'}
              </div>
            )}

            <div className="user-area">
              {currentUser ? (
                <div className="profile">
                  <div className="avatar">{currentUser.username?.charAt(0).toUpperCase()}</div>
                  <span className="username">{currentUser.username}</span>
                  <button onClick={handleLogout} className="icon-btn">Logout</button>
                </div>
              ) : (
                <button onClick={() => setShowAuth(true)} className="primary-btn">Sign In</button>
              )}
            </div>

            <button className="hamburger" onClick={() => setMobileOpen(!mobileOpen)} aria-label="Toggle navigation">☰</button>
          </div>
        </div>
      </header>

      <main className="main-content">
        {/* Authentication Modal */}
        {showAuth && (
          <div className="auth-modal">
            <div className="auth-form">
              <h3>{authMode === 'login' ? 'Login' : 'Register'}</h3>
              <form onSubmit={handleAuthSubmit}>
                <div className="form-group">
                  <label>Username:</label>
                  <input
                    type="text"
                    value={authForm.username}
                    onChange={(e) => setAuthForm({...authForm, username: e.target.value})}
                    required
                  />
                </div>
                {authMode === 'register' && (
                  <div className="form-group">
                    <label>Email:</label>
                    <input
                      type="email"
                      value={authForm.email}
                      onChange={(e) => setAuthForm({...authForm, email: e.target.value})}
                      required
                    />
                  </div>
                )}
                <div className="form-group">
                  <label>Password:</label>
                  <input
                    type="password"
                    value={authForm.password}
                    onChange={(e) => setAuthForm({...authForm, password: e.target.value})}
                    required
                  />
                </div>
                <div className="auth-buttons">
                  <button type="submit" className="submit-btn">
                    {authMode === 'login' ? 'Login' : 'Register'}
                  </button>
                  <button
                    type="button"
                    onClick={() => setAuthMode(authMode === 'login' ? 'register' : 'login')}
                    className="switch-btn"
                  >
                    {authMode === 'login' ? 'Need an account?' : 'Already have an account?'}
                  </button>
                  <button type="button" onClick={() => setShowAuth(false)} className="cancel-btn">
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

  {/* Rating Modal */}
        {ratingMovie && (
          <div className="rating-modal">
            <div className="rating-form">
              <h3>Rate: {ratingMovie.movie_title}</h3>
              <div className="rating-stars">
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    onClick={() => setUserRating(star)}
                    className={`star ${userRating >= star ? 'active' : ''}`}
                  >
                    ★
                  </button>
                ))}
              </div>
              <div className="rating-buttons">
                <button
                  onClick={() => handleRateMovie(ratingMovie.movie_id, userRating)}
                  disabled={userRating === 0}
                  className="submit-btn"
                >
                  Submit Rating
                </button>
                <button onClick={() => { setRatingMovie(null); setUserRating(0); }} className="cancel-btn">
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Training Notice (card shown inside tabs when model not loaded) */}
        {activeTab === 'recommendations' && (
          <section className="hero">
            <div className="hero-inner">
              <h1 className="hero-title"><span className="gradient-text">Discover Your Next Favorite Movie</span></h1>
              <p className="hero-sub">AI-powered hybrid recommendations — personalized and explainable.</p>

              <div className="hero-cta">
                <form onSubmit={handleSubmit} className="inline-form">
                  {!currentUser && (
                    <input
                      type="number"
                      id="userId"
                      value={userId}
                      onChange={(e) => setUserId(e.target.value)}
                      min="1"
                      max="943"
                      placeholder="User ID (1-943)"
                      className="input-field"
                    />
                  )}
                  <button type="submit" disabled={loading} className="primary-btn">
                    {loading ? 'Loading...' : 'Get Recommendations'}
                  </button>
                </form>

                {(!modelStatus?.model_loaded) && (
                  <div className="training-banner">
                    <div className="banner-text">Model not trained yet — Train to enable personalized recommendations</div>
                    <button onClick={handleTrain} disabled={training} className="secondary-btn">
                      {training ? 'Training...' : 'Train Model'}
                    </button>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {/* Tabs content */}
        <section className={`tab-panel ${activeTab === 'recommendations' ? 'visible' : 'hidden'}`}>
          {error && <div className="toast error">{error}</div>}

          {recommendations.length > 0 ? (
            <div className="grid-wrap">
              {recommendations.map((movie, index) => (
                <article key={index} className="card glass">
                  <div className={`thumb ${movie.reason?.includes('Popular') ? 'thumb-pop' : 'thumb-rec'}`}>
                    <div className="thumb-icon">🎞️</div>
                  </div>
                  <div className="card-body">
                    <h3 className="card-title">{movie.title || movie.movie_title}</h3>
                    <div className="card-meta">{Array.isArray(movie.genres) ? movie.genres.join(', ') : movie.genres}</div>
                    <div className="card-rating">★ {(movie.predicted_rating || movie.rating).toFixed(1)}</div>
                    <div className="card-actions">
                      {currentUser && (
                        <button onClick={() => setRatingMovie({ movie_id: movie.movieId || movie.movie_id, movie_title: movie.title || movie.movie_title })} className="ghost-btn">Rate Movie</button>
                      )}
                    </div>
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-emoji">🍿</div>
              <div className="empty-text">No recommendations yet — try getting recommendations or train the model.</div>
            </div>
          )}
        </section>

        <section className={`tab-panel ${activeTab === 'search' ? 'visible' : 'hidden'}`}>
          <div className="search-card glass">
            <textarea value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} placeholder="e.g., 'mind-bending sci-fi thriller with time travel'" className="input-field large" />
            <div className="search-actions">
              <button onClick={handleDescriptionSearch} disabled={!searchQuery.trim() || searchLoading} className="search-primary">
                {searchLoading ? 'Searching...' : 'Search Movies'}
              </button>
            </div>
          </div>

          {searchLoading && <div className="skeleton-grid" />}

          {searchResults.length > 0 ? (
            <div className="grid-wrap">
              {searchResults.map((movie, i) => (
                <article key={i} className="card glass search-result">
                  <div className="thumb thumb-search"><div className="thumb-icon">🎞️</div></div>
                  <div className="card-body">
                    <h3 className="card-title">{movie.title || movie.movie_title}</h3>
                    <div className="card-meta">{Array.isArray(movie.genres) ? movie.genres.join(', ') : movie.genres}</div>
                    <div className="card-extra">Similarity: {(movie.similarity || movie.similarity_score || 0).toFixed(2)}</div>
                    <div className="card-actions">
                      {currentUser && <button onClick={() => setRatingMovie({ movie_id: movie.movie_id || movie.movieId, movie_title: movie.title || movie.movie_title })} className="ghost-btn">Rate Movie</button>}
                    </div>
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <div className="empty-state small">Describe a movie above to start searching.</div>
          )}
        </section>

        <section className={`tab-panel ${activeTab === 'activity' ? 'visible' : 'hidden'}`}>
          <div className="activity-grid">
            <div className="panel glass">
              <h4>My Ratings</h4>
              {userRatings.length === 0 ? <div className="empty-small">No ratings yet</div> : (
                <ul>
                  {userRatings.map((r, idx) => <li key={idx}><strong>{r.title || r.movie_title}</strong> — {r.rating}</li>)}
                </ul>
              )}
            </div>
            <div className="panel glass">
              <h4>History</h4>
              {userHistory.length === 0 ? <div className="empty-small">No history</div> : (
                <ul>
                  {userHistory.map((h, idx) => <li key={idx}><strong>{h.title || h.movie_title}</strong> — {h.rating}</li>)}
                </ul>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
