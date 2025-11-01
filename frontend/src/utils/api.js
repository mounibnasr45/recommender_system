// API utility functions for the movie recommender frontend

// Use environment variable for API URL in production, fallback to /api for development
const API_BASE_URL = import.meta.env.VITE_API_URL 
  ? `${import.meta.env.VITE_API_URL}/api` 
  : '/api';

/**
 * Get movie recommendations for a specific user
 * @param {number} userId - The user ID to get recommendations for
 * @returns {Promise<Object>} - Promise resolving to recommendations data
 */
export const getRecommendations = async (userId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/recommend/${userId}`);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching recommendations:', error);
    throw error;
  }
};

/**
 * Check the health status of the API
 * @returns {Promise<Object>} - Promise resolving to health status
 */
export const checkApiHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error checking API health:', error);
    throw error;
  }
};

/**
 * Validate user ID format
 * @param {string|number} userId - The user ID to validate
 * @returns {boolean} - True if valid, false otherwise
 */
export const isValidUserId = (userId) => {
  const id = parseInt(userId, 10);
  return !isNaN(id) && id >= 1 && id <= 943;
};