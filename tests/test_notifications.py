"""Tests for notifications and webhook system."""

import json
import pytest
import respx
from pathlib import Path
from unittest.mock import patch, MagicMock

from notifications.watcher import DatasetWatcher
from notifications.webhook import WebhookManager, WebhookError


# =============================================================================
# Dataset Watcher Tests
# =============================================================================


class TestDatasetWatcher:
    """Tests for the dataset change watcher."""

    @pytest.fixture
    def watcher(self, tmp_path):
        """Create a watcher with temporary cache directory."""
        with patch.object(DatasetWatcher, 'CACHE_DIR', tmp_path):
            with patch.object(DatasetWatcher, 'SNAPSHOTS_FILE', tmp_path / "snapshots.json"):
                yield DatasetWatcher()

    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset data for testing."""
        return {
            "_about": "https://datos.gob.es/catalog/dataset/test",
            "title": {"_value": "Test Dataset", "_lang": "es"},
            "description": {"_value": "A test dataset"},
            "modified": "2024-12-26T10:00:00Z",
            "distribution": [
                {"accessURL": "https://example.com/data.csv"},
                {"accessURL": "https://example.com/data.json"},
            ]
        }

    def test_take_snapshot(self, watcher, sample_dataset):
        """Test taking a dataset snapshot."""
        snapshot = watcher.take_snapshot("test-dataset", sample_dataset)

        assert snapshot["dataset_id"] == "test-dataset"
        assert snapshot["modified"] == "2024-12-26T10:00:00Z"
        assert snapshot["title"] == "Test Dataset"
        assert snapshot["distributions_count"] == 2
        assert len(snapshot["distribution_urls"]) == 2

    def test_get_snapshot(self, watcher, sample_dataset):
        """Test retrieving a snapshot."""
        watcher.take_snapshot("test-dataset", sample_dataset)

        retrieved = watcher.get_snapshot("test-dataset")

        assert retrieved is not None
        assert retrieved["dataset_id"] == "test-dataset"

    def test_get_snapshot_not_found(self, watcher):
        """Test retrieving non-existent snapshot."""
        snapshot = watcher.get_snapshot("nonexistent")
        assert snapshot is None

    def test_check_changes_new_dataset(self, watcher, sample_dataset):
        """Test checking changes for a new dataset."""
        result = watcher.check_changes("new-dataset", sample_dataset)

        assert result["status"] == "new"
        assert "First snapshot taken" in result["message"]

    def test_check_changes_no_changes(self, watcher, sample_dataset):
        """Test checking changes when nothing changed."""
        watcher.take_snapshot("test-dataset", sample_dataset)

        result = watcher.check_changes("test-dataset", sample_dataset)

        assert result["status"] == "unchanged"
        assert len(result["changes"]) == 0

    def test_check_changes_modified_date(self, watcher, sample_dataset):
        """Test detecting modified date change."""
        watcher.take_snapshot("test-dataset", sample_dataset)

        # Modify the dataset
        modified_dataset = sample_dataset.copy()
        modified_dataset["modified"] = "2024-12-26T15:00:00Z"

        result = watcher.check_changes("test-dataset", modified_dataset)

        assert result["status"] == "changed"
        assert any(c["field"] == "modified" for c in result["changes"])

    def test_check_changes_added_distribution(self, watcher, sample_dataset):
        """Test detecting added distribution."""
        watcher.take_snapshot("test-dataset", sample_dataset)

        # Add a distribution
        modified_dataset = sample_dataset.copy()
        modified_dataset["distribution"] = sample_dataset["distribution"] + [
            {"accessURL": "https://example.com/data.xlsx"}
        ]

        result = watcher.check_changes("test-dataset", modified_dataset)

        assert result["status"] == "changed"
        assert any(c["field"] == "distributions_count" for c in result["changes"])
        assert any(c["field"] == "distributions_added" for c in result["changes"])

    def test_list_watched(self, watcher, sample_dataset):
        """Test listing watched datasets."""
        watcher.take_snapshot("dataset-1", sample_dataset)
        watcher.take_snapshot("dataset-2", sample_dataset)

        watched = watcher.list_watched()

        assert len(watched) == 2
        assert any(w["dataset_id"] == "dataset-1" for w in watched)
        assert any(w["dataset_id"] == "dataset-2" for w in watched)

    def test_remove_watch(self, watcher, sample_dataset):
        """Test removing a dataset from watch list."""
        watcher.take_snapshot("test-dataset", sample_dataset)

        removed = watcher.remove_watch("test-dataset")

        assert removed is True
        assert watcher.get_snapshot("test-dataset") is None

    def test_remove_watch_not_found(self, watcher):
        """Test removing non-existent dataset."""
        removed = watcher.remove_watch("nonexistent")
        assert removed is False

    def test_clear_all(self, watcher, sample_dataset):
        """Test clearing all snapshots."""
        watcher.take_snapshot("dataset-1", sample_dataset)
        watcher.take_snapshot("dataset-2", sample_dataset)

        watcher.clear_all()

        assert len(watcher.list_watched()) == 0


# =============================================================================
# Webhook Manager Tests
# =============================================================================


class TestWebhookManager:
    """Tests for the webhook manager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a webhook manager with temporary cache directory."""
        with patch.object(WebhookManager, 'CACHE_DIR', tmp_path):
            with patch.object(WebhookManager, 'WEBHOOKS_FILE', tmp_path / "webhooks.json"):
                yield WebhookManager()

    def test_register_webhook(self, manager):
        """Test registering a webhook."""
        webhook = manager.register(
            webhook_url="https://example.com/webhook",
            dataset_id="test-dataset",
            event_types=["created", "updated"]
        )

        assert webhook["url"] == "https://example.com/webhook"
        assert webhook["dataset_id"] == "test-dataset"
        assert webhook["event_types"] == ["created", "updated"]
        assert webhook["active"] is True

    def test_register_webhook_invalid_url(self, manager):
        """Test registering webhook with invalid URL."""
        with pytest.raises(WebhookError) as exc_info:
            manager.register(webhook_url="not-a-url")

        assert "valid HTTP" in str(exc_info.value)

    def test_register_webhook_empty_url(self, manager):
        """Test registering webhook with empty URL."""
        with pytest.raises(WebhookError) as exc_info:
            manager.register(webhook_url="")

        assert "required" in str(exc_info.value)

    def test_unregister_webhook(self, manager):
        """Test unregistering a webhook."""
        webhook = manager.register(webhook_url="https://example.com/webhook")

        removed = manager.unregister(webhook["id"])

        assert removed is True
        assert manager.get(webhook["id"]) is None

    def test_unregister_webhook_not_found(self, manager):
        """Test unregistering non-existent webhook."""
        removed = manager.unregister("nonexistent")
        assert removed is False

    def test_get_webhook(self, manager):
        """Test getting a webhook by ID."""
        webhook = manager.register(webhook_url="https://example.com/webhook")

        retrieved = manager.get(webhook["id"])

        assert retrieved is not None
        assert retrieved["id"] == webhook["id"]

    def test_list_all(self, manager):
        """Test listing all webhooks."""
        manager.register(webhook_url="https://example.com/webhook1")
        manager.register(webhook_url="https://example.com/webhook2")

        webhooks = manager.list_all()

        assert len(webhooks) == 2

    def test_list_for_dataset(self, manager):
        """Test filtering webhooks by dataset."""
        manager.register(
            webhook_url="https://example.com/webhook1",
            dataset_id="dataset-1"
        )
        manager.register(
            webhook_url="https://example.com/webhook2",
            dataset_id="dataset-2"
        )
        manager.register(
            webhook_url="https://example.com/webhook3"  # All datasets
        )

        webhooks = manager.list_for_dataset("dataset-1")

        assert len(webhooks) == 2  # Specific + catch-all

    def test_set_active(self, manager):
        """Test enabling/disabling webhook."""
        webhook = manager.register(webhook_url="https://example.com/webhook")

        manager.set_active(webhook["id"], False)

        updated = manager.get(webhook["id"])
        assert updated["active"] is False

    @pytest.mark.asyncio
    async def test_trigger_webhook(self, manager):
        """Test triggering a webhook."""
        webhook = manager.register(
            webhook_url="https://example.com/webhook",
            event_types=["updated"]
        )

        with respx.mock:
            respx.post("https://example.com/webhook").respond(
                status_code=200,
                json={"status": "received"}
            )

            result = await manager.trigger(
                webhook_id=webhook["id"],
                event_type="updated",
                payload={"test": True}
            )

            assert result["status"] == "success"
            assert result["response_status"] == 200

    @pytest.mark.asyncio
    async def test_trigger_webhook_wrong_event(self, manager):
        """Test triggering webhook with wrong event type."""
        webhook = manager.register(
            webhook_url="https://example.com/webhook",
            event_types=["created"]
        )

        result = await manager.trigger(
            webhook_id=webhook["id"],
            event_type="updated",  # Not in event_types
            payload={"test": True}
        )

        assert result["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_trigger_webhook_not_found(self, manager):
        """Test triggering non-existent webhook."""
        with pytest.raises(WebhookError):
            await manager.trigger(
                webhook_id="nonexistent",
                event_type="updated",
                payload={}
            )

    @pytest.mark.asyncio
    async def test_trigger_webhook_inactive(self, manager):
        """Test triggering inactive webhook."""
        webhook = manager.register(webhook_url="https://example.com/webhook")
        manager.set_active(webhook["id"], False)

        with pytest.raises(WebhookError) as exc_info:
            await manager.trigger(
                webhook_id=webhook["id"],
                event_type="updated",
                payload={}
            )

        assert "not active" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_notify_change(self, manager):
        """Test notifying all relevant webhooks."""
        manager.register(
            webhook_url="https://example.com/webhook1",
            dataset_id="test-dataset"
        )
        manager.register(
            webhook_url="https://example.com/webhook2"  # All datasets
        )

        with respx.mock:
            respx.post("https://example.com/webhook1").respond(status_code=200)
            respx.post("https://example.com/webhook2").respond(status_code=200)

            results = await manager.notify_change(
                dataset_id="test-dataset",
                event_type="updated",
                changes={"field": "modified"}
            )

            assert len(results) == 2
            assert all(r["status"] == "success" for r in results)

    @pytest.mark.asyncio
    async def test_trigger_webhook_timeout(self, manager):
        """Test webhook timeout handling."""
        webhook = manager.register(webhook_url="https://example.com/webhook")

        with respx.mock:
            import httpx
            respx.post("https://example.com/webhook").mock(
                side_effect=httpx.TimeoutException("Timeout")
            )

            result = await manager.trigger(
                webhook_id=webhook["id"],
                event_type="updated",
                payload={}
            )

            assert result["status"] == "error"
            assert "timed out" in result["error"]
