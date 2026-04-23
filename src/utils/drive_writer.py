import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def get_drive_service():
    creds_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")

    if not creds_json:
        raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_JSON")

    creds_dict = json.loads(creds_json)

    credentials = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive"]
    )

    service = build("drive", "v3", credentials=credentials)
    return service


def verify_folder_access(service, drive_folder_id):
    """
    Check that the folder exists and is visible to the service account.
    """
    try:
        folder = service.files().get(
            fileId=drive_folder_id,
            fields="id,name,mimeType",
            supportsAllDrives=True
        ).execute()

        print("Verified Drive folder access:")
        print(folder)

        if folder.get("mimeType") != "application/vnd.google-apps.folder":
            raise ValueError(f"ID is not a folder: {drive_folder_id}")

    except Exception as e:
        raise ValueError(
            f"Cannot access Google Drive folder ID '{drive_folder_id}'. "
            f"Check that the folder ID is correct and the folder is shared with the service account. "
            f"Original error: {e}"
        )


def upload_file_to_drive(local_path, drive_folder_id):
    service = get_drive_service()

    verify_folder_access(service, drive_folder_id)

    file_name = os.path.basename(local_path)

    file_metadata = {
        "name": file_name,
        "parents": [drive_folder_id]
    }

    media = MediaFileUpload(local_path, resumable=False)

    created = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id,name,parents",
        supportsAllDrives=True
    ).execute()

    print(f"Uploaded to Drive: {created['name']} | id={created['id']}")
