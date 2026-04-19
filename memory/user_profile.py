"""
LifeOS Agent - User Profile Manager
=====================================
Manages user profiles stored as local JSON files.
Handles name, goals, preferences, topics of interest,
and goal progress tracking.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Default storage directory — uses persistent volume on Railway
PROFILES_DIR = os.getenv("PROFILES_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "profiles"))

# ═══════════════════════════════════════════
# USER PROFILE MANAGER
# ═══════════════════════════════════════════


class UserProfile:
    """Manages persistent user profile data stored as JSON files."""

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or PROFILES_DIR
        os.makedirs(self.storage_dir, exist_ok=True)
        logger.info(f"UserProfile manager initialized. Storage: {self.storage_dir}")

    def _get_profile_path(self, user_id: str) -> str:
        """Get the file path for a user's profile JSON."""
        safe_id = str(user_id).replace("/", "_").replace("\\", "_")
        return os.path.join(self.storage_dir, f"user_{safe_id}.json")

    def _default_profile(self, user_id: str) -> dict:
        """Create a default empty profile."""
        return {
            "user_id": str(user_id),
            "name": "",
            "goals": [],
            "preferences": {},
            "topics_of_interest": [],
            "history": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

    def load(self, user_id: str) -> dict:
        """
        Load full user profile by user_id.

        Args:
            user_id: Unique user identifier.

        Returns:
            Dict with user profile data.
        """
        try:
            filepath = self._get_profile_path(user_id)
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    profile = json.load(f)
                logger.info(f"Loaded profile for user {user_id}")
                return profile
            else:
                logger.info(f"No profile found for user {user_id}, creating default")
                profile = self._default_profile(user_id)
                self.save(user_id, profile)
                return profile
        except Exception as e:
            logger.error(f"Failed to load profile for user {user_id}: {e}")
            return self._default_profile(user_id)

    def save(self, user_id: str, profile: dict) -> bool:
        """
        Save user profile to disk.

        Args:
            user_id: Unique user identifier.
            profile: Profile dict to save.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            filepath = self._get_profile_path(user_id)
            profile["updated_at"] = datetime.now().isoformat()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved profile for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save profile for user {user_id}: {e}")
            return False

    def update_name(self, user_id: str, name: str) -> bool:
        """Update user's display name."""
        try:
            profile = self.load(user_id)
            profile["name"] = name
            return self.save(user_id, profile)
        except Exception as e:
            logger.error(f"Failed to update name for user {user_id}: {e}")
            return False

    def add_goal(self, user_id: str, goal: str, category: str = "general") -> bool:
        """
        Add a new goal to the user's profile.

        Args:
            user_id: Unique user identifier.
            goal: Goal description text.
            category: Goal category (e.g., career, health, learning).

        Returns:
            True if added successfully.
        """
        try:
            profile = self.load(user_id)
            goal_entry = {
                "goal": goal,
                "category": category,
                "status": "active",
                "progress": 0,
                "created_at": datetime.now().isoformat(),
                "milestones": [],
            }
            profile["goals"].append(goal_entry)
            return self.save(user_id, profile)
        except Exception as e:
            logger.error(f"Failed to add goal for user {user_id}: {e}")
            return False

    def update_goal_progress(
        self, user_id: str, goal_index: int, progress: int, milestone: str = ""
    ) -> bool:
        """
        Update progress on a specific goal.

        Args:
            user_id: Unique user identifier.
            goal_index: Index of the goal in the goals list.
            progress: New progress percentage (0-100).
            milestone: Optional milestone description.

        Returns:
            True if updated successfully.
        """
        try:
            profile = self.load(user_id)
            if 0 <= goal_index < len(profile["goals"]):
                profile["goals"][goal_index]["progress"] = min(100, max(0, progress))
                if milestone:
                    profile["goals"][goal_index]["milestones"].append(
                        {
                            "description": milestone,
                            "date": datetime.now().isoformat(),
                        }
                    )
                if progress >= 100:
                    profile["goals"][goal_index]["status"] = "completed"
                return self.save(user_id, profile)
            else:
                logger.warning(f"Invalid goal index {goal_index} for user {user_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to update goal progress for user {user_id}: {e}")
            return False

    def add_interest(self, user_id: str, topic: str) -> bool:
        """Add a topic of interest to user's profile."""
        try:
            profile = self.load(user_id)
            if topic not in profile["topics_of_interest"]:
                profile["topics_of_interest"].append(topic)
                return self.save(user_id, profile)
            return True  # Already exists
        except Exception as e:
            logger.error(f"Failed to add interest for user {user_id}: {e}")
            return False

    def update_preferences(self, user_id: str, key: str, value) -> bool:
        """Update a specific user preference."""
        try:
            profile = self.load(user_id)
            profile["preferences"][key] = value
            return self.save(user_id, profile)
        except Exception as e:
            logger.error(f"Failed to update preference for user {user_id}: {e}")
            return False

    def add_history_entry(self, user_id: str, query: str, summary: str) -> bool:
        """
        Add a conversation history entry.

        Args:
            user_id: Unique user identifier.
            query: The user's input query.
            summary: Brief summary of the response.

        Returns:
            True if added successfully.
        """
        try:
            profile = self.load(user_id)
            entry = {
                "query": query[:200],  # Truncate long queries
                "summary": summary[:500],  # Truncate long summaries
                "timestamp": datetime.now().isoformat(),
            }
            profile["history"].append(entry)

            # Keep only last 50 history entries
            if len(profile["history"]) > 50:
                profile["history"] = profile["history"][-50:]

            return self.save(user_id, profile)
        except Exception as e:
            logger.error(f"Failed to add history for user {user_id}: {e}")
            return False

    def get_goals_summary(self, user_id: str) -> str:
        """Get a formatted summary of user's goals."""
        try:
            profile = self.load(user_id)
            goals = profile.get("goals", [])

            if not goals:
                return "No goals set yet. Tell me your goals and I'll track them!"

            lines = ["📎 **Your Goals:**\n"]
            for idx, goal in enumerate(goals, 1):
                status_emoji = "✅" if goal["status"] == "completed" else "🔄"
                progress = goal.get("progress", 0)
                bar = "█" * (progress // 10) + "░" * (10 - progress // 10)
                lines.append(
                    f"{status_emoji} **{idx}. {goal['goal']}**\n"
                    f"   Category: {goal.get('category', 'general')} | "
                    f"Progress: [{bar}] {progress}%"
                )

            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Failed to get goals summary for user {user_id}: {e}")
            return "Could not retrieve goals."

    def clear_profile(self, user_id: str) -> bool:
        """Clear user profile and start fresh."""
        try:
            filepath = self._get_profile_path(user_id)
            if os.path.exists(filepath):
                os.remove(filepath)
            logger.info(f"Cleared profile for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear profile for user {user_id}: {e}")
            return False

    def get_profile_summary(self, user_id: str) -> str:
        """Get a human-readable summary of the user's profile."""
        try:
            profile = self.load(user_id)
            name = profile.get("name", "Unknown")
            goals_count = len(profile.get("goals", []))
            active_goals = len(
                [g for g in profile.get("goals", []) if g.get("status") == "active"]
            )
            interests = ", ".join(profile.get("topics_of_interest", [])) or "None set"
            history_count = len(profile.get("history", []))

            return (
                f"👤 **User Profile: {name}**\n"
                f"Goals: {goals_count} total ({active_goals} active)\n"
                f"Interests: {interests}\n"
                f"Conversations: {history_count}\n"
                f"Member since: {profile.get('created_at', 'N/A')[:10]}"
            )
        except Exception as e:
            logger.error(f"Failed to get profile summary for user {user_id}: {e}")
            return "Profile unavailable."


# ═══════════════════════════════════════════
# GLOBAL SINGLETON INSTANCE
# ═══════════════════════════════════════════

_profile_instance: Optional[UserProfile] = None


def get_profile_manager() -> UserProfile:
    """Get or create the global UserProfile singleton."""
    global _profile_instance
    if _profile_instance is None:
        _profile_instance = UserProfile()
    return _profile_instance
