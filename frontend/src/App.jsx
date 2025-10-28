import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [userId, setUserId] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState('');
  const [modelStatus, setModelStatus] = useState(null);
  const [checkingHealth, setCheckingHealth] = useState(true);

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
        setUserRatings(ratingsData.ratings || []);
      } else {
        setUserRatings([]);
      }

      // History endpoint (may require model to be loaded)
      const historyResp = await fetch(`/api/history/${targetUserId}`);
      if (historyResp.ok) {
        const historyData = await historyResp.json();
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

  if (checkingHealth) {
    return (
      <div className="App">
        <div className="loading">Checking system status...</div>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>Movie Recommendation System</h1>
        <p>Get personalized movie recommendations using hybrid matrix factorization</p>

        {/* User Authentication Section */}
        <div className="auth-section">
          {currentUser ? (
            <div className="user-info">
              <span>Welcome, {currentUser.username}!</span>
              <button onClick={handleLogout} className="logout-btn">Logout</button>
            </div>
          ) : (
            <button onClick={() => setShowAuth(true)} className="auth-btn">
              Login / Register
            </button>
          )}
        </div>

        {modelStatus && (
          <div className={`model-status ${modelStatus.model_loaded ? 'loaded' : 'not-loaded'}`}>
            Model Status: {modelStatus.model_loaded ? '✓ Loaded' : '⚠️ Not Loaded'}
          </div>
        )}
      </header>

      <main className="App-main">
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

        {!modelStatus?.model_loaded && (
          <div className="training-section">
            <div className="training-notice">
              <h3>Model Not Available</h3>
              <p>The recommendation model needs to be trained before you can get personalized recommendations.</p>
              <p>You can still get popular movie suggestions as a fallback.</p>
              <button
                onClick={handleTrain}
                disabled={training}
                className="train-btn"
              >
                {training ? 'Training Model...' : 'Train Model'}
              </button>
            </div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="recommendation-form">
          {!currentUser && (
            <div className="form-group">
              <label htmlFor="userId">Enter User ID (1-943) or Login for personalized experience:</label>
              <input
                type="number"
                id="userId"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                min="1"
                max="943"
                placeholder="e.g., 1"
              />
            </div>
          )}
          <button type="submit" disabled={loading} className="submit-btn">
            {loading ? 'Getting Recommendations...' : 'Get Recommendations'}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        {recommendations.length > 0 && (
          <div className="recommendations">
            <h2>Recommended Movies</h2>
            <div className="details-toggle">
              <button
                onClick={async () => {
                  // Toggle and fetch details when opening
                  if (!showDetails) await fetchDetails();
                  setShowDetails(!showDetails);
                }}
                className="toggle-btn"
              >
                {showDetails ? 'Hide my ratings & history' : 'Show my ratings & history'}
              </button>
              {loadingDetails && <span className="loading-inline"> Loading details...</span>}
            </div>
            <p className="recommendation-type">
              {recommendations[0]?.reason?.includes('Popular movie') ?
                'Showing popular movies (model not available)' :
                currentUser ?
                  'Personalized recommendations based on your ratings!' :
                  'Personalized recommendations based on your taste'}
            </p>
            <div className="movie-grid">
              {recommendations.map((movie, index) => (
                <div key={index} className="movie-card">
                  <h3>{movie.title || movie.movie_title}</h3>
                  <p className="movie-genres">
                    {Array.isArray(movie.genres) ? movie.genres.join(', ') : movie.genres}
                  </p>
                  <p className="movie-rating">
                    Predicted Rating: {(movie.predicted_rating || movie.rating).toFixed(2)}
                  </p>
                  {currentUser && (
                    <button
                      onClick={() => setRatingMovie({
                        movie_id: movie.movieId || movie.movie_id,
                        movie_title: movie.title || movie.movie_title
                      })}
                      className="rate-btn"
                    >
                      Rate This Movie
                    </button>
                  )}
                  {movie.reason && (
                    <p className="movie-reason">{movie.reason}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Ratings & History Panel */}
        {showDetails && (
          <div className="details-panel">
            <div className="ratings-panel">
              <h3>Your Ratings</h3>
              {userRatings.length === 0 ? (
                <p>No ratings found.</p>
              ) : (
                <ul>
                  {userRatings.map((r, i) => (
                    <li key={i}>
                      <strong>{r.title || r.movie_title || `Movie ID: ${r.movie_id}`}</strong>
                      {r.genres ? <span> — {Array.isArray(r.genres) ? r.genres.join(', ') : r.genres}</span> : null}
                      <span> — Rating: {r.rating}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="history-panel">
              <h3>Your History</h3>
              {userHistory.length === 0 ? (
                <p>No history available.</p>
              ) : (
                <ul>
                  {userHistory.map((h, i) => (
                    <li key={i}>
                      <strong>{h.title || h.movie_title || 'Unknown Title'}</strong>
                      {h.genres ? <span> — {Array.isArray(h.genres) ? h.genres.join(', ') : h.genres}</span> : null}
                      <span> — Rated: {h.rating}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;