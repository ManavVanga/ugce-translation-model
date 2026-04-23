import os
import json
import hashlib
from datetime import datetime


def sha256_file(file_path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_manifest(file_paths, run_name, output_path):
    manifest = {
        "run_name": run_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "files": []
    }

    for path in file_paths:
        if os.path.exists(path):
            manifest["files"].append({
                "path": path,
                "size_bytes": os.path.getsize(path),
                "sha256": sha256_file(path)
            })

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
