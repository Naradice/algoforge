"""
Google Drive uploader for dataset snapshots (see snapshot_service.py).

Auth: OAuth2 as a real Google account (installed-app flow), NOT a service account. A service
account has no Drive storage quota of its own, so uploading to a regular "My Drive" folder from
one fails with `storageQuotaExceeded` even if the folder is shared with it (confirmed live,
2026-09-01) — Google's own fix for that is a Shared Drive, which requires a Google Workspace
account and isn't available on a plain personal Gmail account. OAuth as a real account avoids
the whole issue: the uploaded file is owned by that account, using its own quota, in a folder it
already owns.

Required env:
    GDRIVE_OAUTH_CLIENT_JSON     — path to an OAuth client JSON (Desktop app type)
    GDRIVE_SNAPSHOT_FOLDER_ID    — Drive folder ID to upload into (must be owned by, or already
                                   shared as editable to, the account that completes the login)
Optional env:
    GDRIVE_OAUTH_TOKEN_PATH      — where the resulting refresh token is cached
                                   (default: secrets/gdrive-token.json, relative to backend/)
    GDRIVE_SNAPSHOT_PUBLIC       — "true" (default) to grant "anyone with the link: viewer" so a
                                   public Colab notebook can fetch the file without any auth.
                                   Set "false" to rely on the folder's own sharing instead.

One-time setup (by a human):
    1. In Google Cloud Console, create/choose a project and enable the Drive API.
    2. Create an OAuth client ID of type "Desktop app", download its client JSON.
    3. Put its path in backend/.env as GDRIVE_OAUTH_CLIENT_JSON.
    4. Pick (or create) a Drive folder under the Google account you intend to upload as, put its
       ID (from the folder's URL) into backend/.env as GDRIVE_SNAPSHOT_FOLDER_ID.
    5. Run any command that uploads (e.g. `scripts/export_dataset_snapshot.py --upload-gdrive`)
       from an interactive terminal — the first call opens a browser for you to log into that
       Google account and approve access; the resulting token is cached at
       GDRIVE_OAUTH_TOKEN_PATH, so every call after that is non-interactive.

Requires the `google-api-python-client` and `google-auth-oauthlib` packages (see
requirements.txt) — not installed by default, since this is the one export path most dev setups
won't use.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("gdrive_export")

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


class GDriveNotConfigured(RuntimeError):
    """Raised when required env vars or packages for Drive upload are missing."""


def _get_credentials() -> Any:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise GDriveNotConfigured(
            "google-api-python-client / google-auth-oauthlib are not installed — "
            "pip install -r requirements.txt (see the google-api-python-client entry)"
        ) from exc

    client_json = os.getenv("GDRIVE_OAUTH_CLIENT_JSON", "")
    if not client_json:
        raise GDriveNotConfigured("GDRIVE_OAUTH_CLIENT_JSON is not set")
    if not Path(client_json).is_file():
        raise GDriveNotConfigured(f"GDRIVE_OAUTH_CLIENT_JSON points to a missing file: {client_json}")

    token_path = Path(os.getenv("GDRIVE_OAUTH_TOKEN_PATH", "secrets/gdrive-token.json"))

    creds = None
    if token_path.is_file():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Blocks and opens a browser -- only works from an interactive terminal. Once this
            # succeeds, the cached token below makes every later call non-interactive.
            flow = InstalledAppFlow.from_client_secrets_file(client_json, SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
        log.info(f"cached refresh token at {token_path}")

    return creds


def upload(local_path: Path, name: str | None = None) -> dict[str, str]:
    """Upload *local_path* into the configured Drive folder.

    Returns {"file_id": ..., "url": ...}. Raises GDriveNotConfigured if setup is incomplete, or
    the underlying Drive API's own exception otherwise — unlike artifact_store.upload's
    best-effort backup semantics, a snapshot export failing silently would be actively
    misleading (the caller believes a shareable link exists when it doesn't).
    """
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    folder_id = os.getenv("GDRIVE_SNAPSHOT_FOLDER_ID", "")
    if not folder_id:
        raise GDriveNotConfigured("GDRIVE_SNAPSHOT_FOLDER_ID is not set")

    creds = _get_credentials()
    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    file_metadata = {"name": name or local_path.name, "parents": [folder_id]}
    media = MediaFileUpload(str(local_path), resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields="id, webViewLink").execute()
    file_id = file["id"]

    public = os.getenv("GDRIVE_SNAPSHOT_PUBLIC", "true").lower() in ("1", "true", "yes")
    if public:
        service.permissions().create(
            fileId=file_id, body={"type": "anyone", "role": "reader"},
        ).execute()

    # A direct-download URL rather than webViewLink (which opens Drive's HTML preview) — this
    # form works with gdown/wget/curl once the file is shared publicly, no auth round-trip.
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    log.info(f"uploaded {local_path} -> gdrive file {file_id}")
    return {"file_id": file_id, "url": url}
