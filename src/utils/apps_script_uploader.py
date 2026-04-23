import os
import requests


def guess_mime_type(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".csv"):
        return "text/csv"
    if lower.endswith(".json"):
        return "application/json"
    if lower.endswith(".txt"):
        return "text/plain"
    return "application/octet-stream"


def upload_file_via_apps_script(local_path: str, webhook_url: str, secret: str) -> None:
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File not found: {local_path}")

    filename = os.path.basename(local_path)
    mime_type = guess_mime_type(local_path)

    with open(local_path, "r", encoding="utf-8") as f:
        content = f.read()

    response = requests.post(
        webhook_url,
        params={
            "secret": secret,
            "filename": filename,
            "mimetype": mime_type,
        },
        data=content.encode("utf-8"),
        timeout=120,
    )
    response.raise_for_status()
    print(f"Uploaded via Apps Script: {filename}")
    print(response.text)
