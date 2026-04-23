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


def upload_file_to_drive(local_path, drive_folder_id):
    service = get_drive_service()

    file_name = os.path.basename(local_path)

    file_metadata = {
        "name": file_name,
        "parents": [drive_folder_id]
    }

    media = MediaFileUpload(local_path, mimetype="text/csv")

    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()

    print(f"Uploaded to Drive: {file_name}")
