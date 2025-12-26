"""Dataset watcher for detecting changes in datos.gob.es datasets."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class DatasetWatcher:
    """Watches datasets for changes by comparing snapshots."""

    CACHE_DIR = Path.home() / ".cache" / "datos-gob-es"
    SNAPSHOTS_FILE = CACHE_DIR / "dataset_snapshots.json"

    def __init__(self):
        self._ensure_cache_dir()
        self.snapshots: dict[str, dict[str, Any]] = self._load_snapshots()

    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_snapshots(self) -> dict[str, dict[str, Any]]:
        """Load snapshots from disk."""
        if self.SNAPSHOTS_FILE.exists():
            try:
                with open(self.SNAPSHOTS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_snapshots(self):
        """Save snapshots to disk."""
        with open(self.SNAPSHOTS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.snapshots, f, ensure_ascii=False, indent=2)

    def take_snapshot(self, dataset_id: str, dataset_data: dict[str, Any]) -> dict[str, Any]:
        """Take a snapshot of a dataset's current state.

        Args:
            dataset_id: The dataset identifier
            dataset_data: The current dataset metadata from the API

        Returns:
            The snapshot that was created
        """
        snapshot = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().isoformat(),
            "modified": dataset_data.get("modified"),
            "title": self._extract_text(dataset_data.get("title")),
            "description_hash": hash(str(dataset_data.get("description"))),
            "distributions_count": len(dataset_data.get("distribution", [])),
            "distribution_urls": [
                d.get("accessURL") for d in dataset_data.get("distribution", [])
                if isinstance(d, dict)
            ],
        }

        self.snapshots[dataset_id] = snapshot
        self._save_snapshots()
        return snapshot

    def get_snapshot(self, dataset_id: str) -> dict[str, Any] | None:
        """Get the last snapshot for a dataset."""
        return self.snapshots.get(dataset_id)

    def check_changes(
        self,
        dataset_id: str,
        current_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Check if a dataset has changed since the last snapshot.

        Args:
            dataset_id: The dataset identifier
            current_data: The current dataset metadata from the API

        Returns:
            Dict with change information
        """
        previous = self.get_snapshot(dataset_id)

        if not previous:
            # First time seeing this dataset
            self.take_snapshot(dataset_id, current_data)
            return {
                "dataset_id": dataset_id,
                "status": "new",
                "message": "First snapshot taken, no previous data to compare",
                "timestamp": datetime.now().isoformat(),
            }

        # Compare with current data
        changes = []
        current_modified = current_data.get("modified")
        current_title = self._extract_text(current_data.get("title"))
        current_desc_hash = hash(str(current_data.get("description")))
        current_dists = current_data.get("distribution", [])
        current_dist_count = len(current_dists)
        current_dist_urls = [
            d.get("accessURL") for d in current_dists
            if isinstance(d, dict)
        ]

        # Check modified date
        if current_modified != previous.get("modified"):
            changes.append({
                "field": "modified",
                "previous": previous.get("modified"),
                "current": current_modified,
            })

        # Check title
        if current_title != previous.get("title"):
            changes.append({
                "field": "title",
                "previous": previous.get("title"),
                "current": current_title,
            })

        # Check description (by hash)
        if current_desc_hash != previous.get("description_hash"):
            changes.append({
                "field": "description",
                "message": "Description has changed",
            })

        # Check distributions count
        if current_dist_count != previous.get("distributions_count"):
            changes.append({
                "field": "distributions_count",
                "previous": previous.get("distributions_count"),
                "current": current_dist_count,
            })

        # Check distribution URLs
        prev_urls = set(previous.get("distribution_urls", []))
        curr_urls = set(current_dist_urls)

        added_urls = curr_urls - prev_urls
        removed_urls = prev_urls - curr_urls

        if added_urls:
            changes.append({
                "field": "distributions_added",
                "urls": list(added_urls),
            })

        if removed_urls:
            changes.append({
                "field": "distributions_removed",
                "urls": list(removed_urls),
            })

        # Update snapshot if there are changes
        if changes:
            self.take_snapshot(dataset_id, current_data)

        return {
            "dataset_id": dataset_id,
            "status": "changed" if changes else "unchanged",
            "changes": changes,
            "previous_check": previous.get("timestamp"),
            "current_check": datetime.now().isoformat(),
        }

    def list_watched(self) -> list[dict[str, Any]]:
        """List all datasets being watched."""
        return [
            {
                "dataset_id": dataset_id,
                "last_check": snapshot.get("timestamp"),
                "modified": snapshot.get("modified"),
                "title": snapshot.get("title"),
            }
            for dataset_id, snapshot in self.snapshots.items()
        ]

    def remove_watch(self, dataset_id: str) -> bool:
        """Remove a dataset from the watch list.

        Returns:
            True if the dataset was removed, False if it wasn't being watched
        """
        if dataset_id in self.snapshots:
            del self.snapshots[dataset_id]
            self._save_snapshots()
            return True
        return False

    def clear_all(self):
        """Clear all snapshots."""
        self.snapshots = {}
        self._save_snapshots()

    @staticmethod
    def _extract_text(value: Any) -> str | None:
        """Extract text from a potentially multilingual field."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return value.get("_value")
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, dict):
                return first.get("_value")
            return str(first)
        return str(value)


# Global watcher instance
dataset_watcher = DatasetWatcher()
