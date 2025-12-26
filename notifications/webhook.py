"""Webhook system for notifying about dataset changes."""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from .watcher import dataset_watcher

# Load environment variables
load_dotenv()


class WebhookError(Exception):
    """Exception raised for webhook errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class WebhookManager:
    """Manages webhook registrations and notifications."""

    CACHE_DIR = Path.home() / ".cache" / "datos-gob-es"
    WEBHOOKS_FILE = CACHE_DIR / "webhooks.json"
    DEFAULT_TIMEOUT = 10.0

    def __init__(self):
        self._ensure_cache_dir()
        self.webhooks: dict[str, dict[str, Any]] = self._load_webhooks()
        self.secret = os.getenv("WEBHOOK_SECRET")

    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_webhooks(self) -> dict[str, dict[str, Any]]:
        """Load webhooks from disk."""
        if self.WEBHOOKS_FILE.exists():
            try:
                with open(self.WEBHOOKS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_webhooks(self):
        """Save webhooks to disk."""
        with open(self.WEBHOOKS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.webhooks, f, ensure_ascii=False, indent=2)

    def register(
        self,
        webhook_url: str,
        dataset_id: str | None = None,
        theme: str | None = None,
        event_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Register a new webhook.

        Args:
            webhook_url: URL to send notifications to
            dataset_id: Optional dataset ID to watch (specific dataset)
            theme: Optional theme to watch (all datasets in theme)
            event_types: List of event types to notify on
                         Options: 'created', 'updated', 'deleted'

        Returns:
            The webhook registration info including ID
        """
        if not webhook_url:
            raise WebhookError("webhook_url is required")

        if not webhook_url.startswith(("http://", "https://")):
            raise WebhookError("webhook_url must be a valid HTTP(S) URL")

        webhook_id = str(uuid.uuid4())[:8]

        webhook = {
            "id": webhook_id,
            "url": webhook_url,
            "dataset_id": dataset_id,
            "theme": theme,
            "event_types": event_types or ["created", "updated", "deleted"],
            "created_at": datetime.now().isoformat(),
            "last_triggered": None,
            "trigger_count": 0,
            "active": True,
        }

        self.webhooks[webhook_id] = webhook
        self._save_webhooks()

        return webhook

    def unregister(self, webhook_id: str) -> bool:
        """Unregister a webhook.

        Args:
            webhook_id: The webhook ID to remove

        Returns:
            True if removed, False if not found
        """
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            self._save_webhooks()
            return True
        return False

    def get(self, webhook_id: str) -> dict[str, Any] | None:
        """Get a webhook by ID."""
        return self.webhooks.get(webhook_id)

    def list_all(self) -> list[dict[str, Any]]:
        """List all registered webhooks."""
        return list(self.webhooks.values())

    def list_for_dataset(self, dataset_id: str) -> list[dict[str, Any]]:
        """Get webhooks that apply to a specific dataset."""
        return [
            wh for wh in self.webhooks.values()
            if wh.get("active") and (
                wh.get("dataset_id") == dataset_id or
                wh.get("dataset_id") is None  # Catches all datasets
            )
        ]

    def list_for_theme(self, theme: str) -> list[dict[str, Any]]:
        """Get webhooks that apply to a specific theme."""
        return [
            wh for wh in self.webhooks.values()
            if wh.get("active") and (
                wh.get("theme") == theme or
                wh.get("theme") is None  # Catches all themes
            )
        ]

    async def trigger(
        self,
        webhook_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Trigger a specific webhook with a payload.

        Args:
            webhook_id: The webhook to trigger
            event_type: The type of event ('created', 'updated', 'deleted')
            payload: The data to send

        Returns:
            Result of the webhook call
        """
        webhook = self.webhooks.get(webhook_id)
        if not webhook:
            raise WebhookError(f"Webhook {webhook_id} not found")

        if not webhook.get("active"):
            raise WebhookError(f"Webhook {webhook_id} is not active")

        if event_type not in webhook.get("event_types", []):
            return {
                "status": "skipped",
                "reason": f"Event type '{event_type}' not in webhook event_types",
            }

        # Build the notification payload
        notification = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "webhook_id": webhook_id,
            "data": payload,
        }

        # Add signature if secret is configured
        if self.secret:
            import hashlib
            import hmac
            signature = hmac.new(
                self.secret.encode(),
                json.dumps(notification, sort_keys=True).encode(),
                hashlib.sha256,
            ).hexdigest()
            notification["signature"] = signature

        # Send the webhook
        async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
            try:
                response = await client.post(
                    webhook["url"],
                    json=notification,
                    headers={"Content-Type": "application/json"},
                )

                # Update webhook stats
                webhook["last_triggered"] = datetime.now().isoformat()
                webhook["trigger_count"] = webhook.get("trigger_count", 0) + 1
                self._save_webhooks()

                return {
                    "status": "success",
                    "webhook_id": webhook_id,
                    "response_status": response.status_code,
                    "response_body": response.text[:500],  # Truncate response
                }

            except httpx.TimeoutException:
                return {
                    "status": "error",
                    "webhook_id": webhook_id,
                    "error": "Request timed out",
                }
            except httpx.RequestError as e:
                return {
                    "status": "error",
                    "webhook_id": webhook_id,
                    "error": str(e),
                }

    async def notify_change(
        self,
        dataset_id: str,
        event_type: str,
        changes: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Notify all relevant webhooks about a dataset change.

        Args:
            dataset_id: The dataset that changed
            event_type: Type of change ('created', 'updated', 'deleted')
            changes: The change details

        Returns:
            List of notification results
        """
        webhooks = self.list_for_dataset(dataset_id)
        results = []

        for webhook in webhooks:
            payload = {
                "dataset_id": dataset_id,
                "changes": changes,
            }
            result = await self.trigger(webhook["id"], event_type, payload)
            results.append(result)

        return results

    def set_active(self, webhook_id: str, active: bool) -> bool:
        """Enable or disable a webhook.

        Args:
            webhook_id: The webhook ID
            active: Whether to activate (True) or deactivate (False)

        Returns:
            True if updated, False if not found
        """
        if webhook_id in self.webhooks:
            self.webhooks[webhook_id]["active"] = active
            self._save_webhooks()
            return True
        return False


# Global webhook manager instance
webhook_manager = WebhookManager()


def _handle_error(e: Exception) -> str:
    """Format error message."""
    if isinstance(e, WebhookError):
        return json.dumps({"error": e.message}, ensure_ascii=False)
    return json.dumps({"error": str(e)}, ensure_ascii=False)


def register_webhook_tools(mcp):
    """Register webhook tools with the MCP server."""

    @mcp.tool()
    async def webhook_register(
        webhook_url: str,
        dataset_id: str | None = None,
        theme: str | None = None,
        event_types: str = "created,updated,deleted",
    ) -> str:
        """Register a webhook to receive notifications about dataset changes.

        Set up a webhook endpoint to be notified when datasets are created,
        updated, or deleted. You can watch specific datasets, themes, or all changes.

        Args:
            webhook_url: Your HTTP(S) endpoint URL to receive POST notifications.
            dataset_id: Optional specific dataset ID to watch.
            theme: Optional theme to watch (e.g., 'economia', 'salud').
            event_types: Comma-separated event types: 'created', 'updated', 'deleted'.
                         Default is all three.

        Returns:
            JSON with the webhook registration details including the webhook ID.
        """
        try:
            events = [e.strip() for e in event_types.split(",")]
            webhook = webhook_manager.register(
                webhook_url=webhook_url,
                dataset_id=dataset_id,
                theme=theme,
                event_types=events,
            )
            return json.dumps({
                "status": "registered",
                "webhook": webhook,
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return _handle_error(e)

    @mcp.tool()
    async def webhook_list() -> str:
        """List all registered webhooks.

        View all webhook endpoints you have registered for dataset change
        notifications. Shows URL, filters, and statistics.

        Returns:
            JSON with list of all registered webhooks.
        """
        try:
            webhooks = webhook_manager.list_all()
            return json.dumps({
                "total_webhooks": len(webhooks),
                "webhooks": webhooks,
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return _handle_error(e)

    @mcp.tool()
    async def webhook_delete(webhook_id: str) -> str:
        """Delete a registered webhook.

        Remove a webhook registration to stop receiving notifications.

        Args:
            webhook_id: The webhook ID to delete (from webhook_register or webhook_list).

        Returns:
            JSON confirmation of deletion.
        """
        try:
            removed = webhook_manager.unregister(webhook_id)
            if removed:
                return json.dumps({
                    "status": "deleted",
                    "webhook_id": webhook_id,
                }, ensure_ascii=False, indent=2)
            return json.dumps({
                "status": "not_found",
                "webhook_id": webhook_id,
                "message": "Webhook not found",
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return _handle_error(e)

    @mcp.tool()
    async def check_dataset_changes(dataset_id: str, current_data: str | None = None) -> str:
        """Check if a dataset has changed since the last check.

        Compare the current state of a dataset with its previous snapshot
        to detect changes. Useful for monitoring datasets you care about.

        Note: You need to fetch the current dataset data first using get_dataset,
        then pass it here. If no current_data is provided, this will show the
        last known state.

        Args:
            dataset_id: The dataset ID to check.
            current_data: Optional JSON string with current dataset data from API.

        Returns:
            JSON with change detection results.
        """
        try:
            if current_data:
                data = json.loads(current_data)
                # Handle both direct data and API response format
                if "result" in data:
                    items = data.get("result", {}).get("items", [])
                    if items:
                        data = items[0]

                result = dataset_watcher.check_changes(dataset_id, data)
            else:
                # Just return the current snapshot
                snapshot = dataset_watcher.get_snapshot(dataset_id)
                if snapshot:
                    result = {
                        "dataset_id": dataset_id,
                        "status": "snapshot_found",
                        "snapshot": snapshot,
                    }
                else:
                    result = {
                        "dataset_id": dataset_id,
                        "status": "not_watched",
                        "message": "No previous snapshot. Fetch dataset with get_dataset and pass the data to start watching.",
                    }

            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return _handle_error(e)

    @mcp.tool()
    async def list_watched_datasets() -> str:
        """List all datasets being watched for changes.

        View all datasets that have snapshots and are being monitored
        for changes.

        Returns:
            JSON with list of watched datasets and their last check time.
        """
        try:
            watched = dataset_watcher.list_watched()
            return json.dumps({
                "total_watched": len(watched),
                "datasets": watched,
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return _handle_error(e)

    @mcp.tool()
    async def webhook_test(webhook_id: str) -> str:
        """Send a test notification to a webhook.

        Verify that your webhook endpoint is working correctly by
        sending a test notification.

        Args:
            webhook_id: The webhook ID to test.

        Returns:
            JSON with the test result including response from your endpoint.
        """
        try:
            result = await webhook_manager.trigger(
                webhook_id=webhook_id,
                event_type="updated",
                payload={
                    "test": True,
                    "message": "This is a test notification from datos-gob-es-mcp",
                    "timestamp": datetime.now().isoformat(),
                },
            )
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return _handle_error(e)
