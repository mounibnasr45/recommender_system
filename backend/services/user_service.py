"""
User Management Service
Handles user creation, authentication, and rating storage for new users
"""

import os
import json
import hashlib
import secrets
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from pathlib import Path


class UserService:
    """Service for managing users and their ratings"""

    def __init__(self, users_file: str = None, ratings_file: str = None):
        if users_file is None:
            # Default to backend directory
            backend_dir = Path(__file__).parent.parent
            users_file = backend_dir / "users.json"
            ratings_file = backend_dir / "user_ratings.json"

        self.users_file = Path(users_file)
        self.ratings_file = Path(ratings_file)

        # Ensure files exist
        self._ensure_files_exist()

        # Load data
        self.users = self._load_users()
        self.ratings = self._load_ratings()

        # Track next user ID (starting from 1000 to avoid conflict with MovieLens users)
        self.next_user_id = self._get_next_user_id()

    def _ensure_files_exist(self):
        """Ensure user and rating files exist"""
        if not self.users_file.exists():
            self.users_file.write_text("{}")

        if not self.ratings_file.exists():
            self.ratings_file.write_text("[]")

    def _load_users(self) -> Dict:
        """Load users from file"""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _load_ratings(self) -> List[Dict]:
        """Load ratings from file"""
        try:
            with open(self.ratings_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_users(self):
        """Save users to file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)

    def _save_ratings(self):
        """Save ratings to file"""
        with open(self.ratings_file, 'w') as f:
            json.dump(self.ratings, f, indent=2)

    def _get_next_user_id(self) -> int:
        """Get next available user ID"""
        if not self.users:
            return 1000  # Start from 1000 to avoid MovieLens user ID conflicts

        max_id = max(int(uid) for uid in self.users.keys())
        return max_id + 1

    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
        return f"{salt}:{hashed}"

    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hashed = hashed_password.split(':')
            expected = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
            return expected == hashed
        except:
            return False

    def create_user(self, username: str, email: str, password: str) -> Tuple[bool, str, Optional[int]]:
        """
        Create a new user

        Returns:
            (success, message, user_id)
        """
        # Check if username or email already exists
        for user_data in self.users.values():
            if user_data['username'] == username:
                return False, "Username already exists", None
            if user_data['email'] == email:
                return False, "Email already exists", None

        # Create user
        user_id = self.next_user_id
        self.next_user_id += 1

        self.users[str(user_id)] = {
            'user_id': user_id,
            'username': username,
            'email': email,
            'password_hash': self._hash_password(password),
            'created_at': datetime.now().isoformat(),
            'last_login': None
        }

        self._save_users()
        return True, "User created successfully", user_id

    def authenticate_user(self, username: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Authenticate user

        Returns:
            (success, message, user_data)
        """
        # Find user by username
        for user_id, user_data in self.users.items():
            if user_data['username'] == username:
                if self._verify_password(password, user_data['password_hash']):
                    # Update last login
                    user_data['last_login'] = datetime.now().isoformat()
                    self._save_users()
                    return True, "Login successful", user_data
                else:
                    return False, "Invalid password", None

        return False, "User not found", None

    def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user data by ID"""
        return self.users.get(str(user_id))

    def add_rating(self, user_id: int, movie_id: int, rating: float, timestamp: Optional[str] = None) -> Dict:
        """
        Add or update a rating for a user

        Args:
            user_id: User ID
            movie_id: Movie ID
            rating: Rating value (0.5-5.0)
            timestamp: Optional timestamp, defaults to now

        Returns:
            The rating data dictionary
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        # Check if rating already exists
        existing_rating_idx = None
        for i, r in enumerate(self.ratings):
            if r['user_id'] == user_id and r['movie_id'] == movie_id:
                existing_rating_idx = i
                break

        rating_data = {
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': timestamp
        }

        if existing_rating_idx is not None:
            # Update existing rating
            self.ratings[existing_rating_idx] = rating_data
        else:
            # Add new rating
            self.ratings.append(rating_data)

        self._save_ratings()
        return rating_data

    def get_user_ratings(self, user_id: int) -> List[Dict]:
        """Get all ratings for a user"""
        return [r for r in self.ratings if r['user_id'] == user_id]

    def get_user_rating_for_movie(self, user_id: int, movie_id: int) -> Optional[float]:
        """Get user's rating for a specific movie"""
        for r in self.ratings:
            if r['user_id'] == user_id and r['movie_id'] == movie_id:
                return r['rating']
        return None

    def get_all_ratings_df(self) -> pd.DataFrame:
        """Get all user ratings as DataFrame"""
        if not self.ratings:
            return pd.DataFrame(columns=['user_id', 'movie_id', 'rating', 'timestamp'])

        df = pd.DataFrame(self.ratings)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def get_stats(self) -> Dict:
        """Get user statistics"""
        return {
            'total_users': len(self.users),
            'total_ratings': len(self.ratings),
            'avg_ratings_per_user': len(self.ratings) / len(self.users) if self.users else 0
        }