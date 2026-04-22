import json
import os


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def load_json_config(relative_path):
    repo_root = _repo_root()
    config_path = os.path.join(repo_root, relative_path)

    with open(config_path, "r") as f:
        return json.load(f)


def load_drive_paths():
    return load_json_config(os.path.join("configs", "drive_paths.json"))


def load_runtime_config():
    return load_json_config(os.path.join("configs", "runtime_config.json"))


def get_base_output_dir():
    runtime = load_runtime_config()

    storage_mode = runtime.get("storage_mode", "github").strip().lower()

    if storage_mode == "colab":
        return runtime["colab_output_dir"]

    return runtime["github_output_dir"]


if __name__ == "__main__":
    print("Drive paths:")
    print(load_drive_paths())

    print("\nRuntime config:")
    print(load_runtime_config())

    print("\nBase output dir:")
    print(get_base_output_dir())
