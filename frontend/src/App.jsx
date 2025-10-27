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

  useEffect(() => {
    checkHealth();
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

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userId.trim()) {
      setError('Please enter a user ID');
      return;
    }

    setLoading(true);
    setError('');
    setRecommendations([]);

    try {
      const response = await fetch(`/api/recommend/${userId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setRecommendations(data.recommendations || []);
    } catch (err) {
      setError(`Failed to get recommendations: ${err.message}`);
    } finally {
      setLoading(false);
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
        {modelStatus && (
          <div className={`model-status ${modelStatus.model_loaded ? 'loaded' : 'not-loaded'}`}>
            Model Status: {modelStatus.model_loaded ? '✓ Loaded' : '⚠️ Not Loaded'}
          </div>
        )}
      </header>

      <main className="App-main">
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
          <div className="form-group">
            <label htmlFor="userId">Enter User ID (1-943):</label>
            <input
              type="number"
              id="userId"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              min="1"
              max="943"
              placeholder="e.g., 1"
              required
            />
          </div>
          <button type="submit" disabled={loading} className="submit-btn">
            {loading ? 'Getting Recommendations...' : 'Get Recommendations'}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        {recommendations.length > 0 && (
          <div className="recommendations">
            <h2>Recommended Movies</h2>
            <p className="recommendation-type">
              {recommendations[0]?.reason?.includes('Popular movie') ?
                'Showing popular movies (model not available)' :
                'Personalized recommendations based on your taste'}
            </p>
            <div className="movie-grid">
              {recommendations.map((movie, index) => (
                <div key={index} className="movie-card">
                  <h3>{movie.movie_title || movie.title}</h3>
                  <p className="movie-genres">
                    {Array.isArray(movie.genres) ? movie.genres.join(', ') : movie.genres}
                  </p>
                  <p className="movie-rating">
                    Predicted Rating: {(movie.predicted_rating || movie.rating).toFixed(2)}
                  </p>
                  {movie.reason && (
                    <p className="movie-reason">{movie.reason}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;