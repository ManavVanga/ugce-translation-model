import json
import os

def load_drive_paths():
    """
    Load Google Drive paths from config file
    """
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(base_path, "configs", "drive_paths.json")

    with open(config_path, "r") as f:
        paths = json.load(f)

    return paths


if __name__ == "__main__":
    paths = load_drive_paths()
    print("Loaded Drive paths:")
    for k, v in paths.items():
        print(f"{k}: {v}")
