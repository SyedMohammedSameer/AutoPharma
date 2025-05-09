#!/usr/bin/env python3
"""
Script to download a HuggingFace model and convert it to GGUF format for use with llama.cpp
"""
import argparse
import subprocess
import sys
import shutil
from pathlib import Path

def install_packages():
    """Ensure required packages are installed."""
    try:
        print("Installing required packages…")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "huggingface_hub", "--upgrade"],
            check=True
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "sentencepiece"],
            check=True
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", "llama-cpp-python[server]"],
            check=True
        )
        print("✅ Required packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install required packages: {e}")
        sys.exit(1)

def download_model(repo_id: str, temp_dir: Path) -> str:
    """Download the model snapshot into temp_dir."""
    print(f"📥 Downloading model {repo_id}…")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_str = str(temp_dir.resolve())
    try:
        subprocess.run([
            sys.executable, "-c",
            f"from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id='{repo_id}', local_dir=r'{temp_str}')"
        ], check=True)
        print("✅ Model downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download model: {e}")
        sys.exit(1)
    return temp_str

def convert_to_gguf(temp_str: str, output_file: Path, quantize: str):
    """Convert the downloaded model to GGUF using llama.cpp's converter script."""
    print(f"🔄 Converting to GGUF format with {quantize} quantization…")

    # Clone llama.cpp if missing
    repo_dir = Path("llama.cpp")
    if not repo_dir.exists():
        print("📥 Cloning llama.cpp for conversion script…")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(repo_dir)],
            check=True
        )

    # Locate converter script (top-level or fallback)
    converter = repo_dir / "convert_hf_to_gguf.py"
    if not converter.exists():
        converter = repo_dir / "convert.py"
    if not converter.exists():
        print(f"❌ Converter script not found in {repo_dir}")
        sys.exit(1)

    # Map user quantize to supported outtype
    valid_outtypes = {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}
    scheme = quantize.lower()
    if scheme not in valid_outtypes:
        print(f"⚠️ Quantization scheme '{quantize}' not supported; defaulting to 'auto'.")
        scheme = "auto"

    cmd = [
        sys.executable,
        str(converter),
        temp_str,
        "--outfile", str(output_file),
        "--outtype", scheme
    ]
    print("Running conversion:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Model converted and saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download & convert a HuggingFace model to GGUF for llama.cpp")
    parser.add_argument("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct",
                        help="HuggingFace model ID to download")
    parser.add_argument("--quantize", type=str, default="Q4_K_M",
                        help="Quantization type (e.g., Q4_K_M, Q5_K_M, Q8_0)")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save converted GGUF file")
    args = parser.parse_args()

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model.split("/")[-1]
    output_file = output_dir / f"{model_name}.{args.quantize}.gguf"
    if output_file.exists():
        print(f"⚠️ {output_file} already exists, skipping conversion.")
        return

    # Run steps
    install_packages()
    temp_dir = Path.cwd() / "tmp_model"
    temp_str = download_model(args.model, temp_dir)
    convert_to_gguf(temp_str, output_file, args.quantize)

    # Cleanup
    print("🧹 Cleaning up temporary files…")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"🎉 All done! Your GGUF model is ready at {output_file}")

if __name__ == "__main__":
    main()
