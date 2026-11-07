"""Utility to download the quantized local GGUF model from Hugging Face."""

import os
from pathlib import Path


def download_local_model():
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    target_file = models_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

    if target_file.exists() and target_file.stat().st_size > 10_000_000:
        print(f"Model already exists at {target_file} ({target_file.stat().st_size / (1024*1024):.1f} MB). Skipping download.")
        return str(target_file)

    print("Downloading Qwen2.5-0.5B-Instruct Q4_K_M GGUF from Hugging Face...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
            local_dir="models",
            local_dir_use_symlinks=False,
        )
        print(f"Model successfully saved to: {downloaded}")
        return downloaded
    except ImportError:
        print("huggingface_hub is not installed. Installing or running urllib fallback...")
        import urllib.request
        url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        urllib.request.urlretrieve(url, str(target_file))
        print(f"Downloaded model to {target_file}")
        return str(target_file)


if __name__ == "__main__":
    download_local_model()
