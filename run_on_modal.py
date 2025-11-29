#!/usr/bin/env python3
"""
run_on_modal.py - Run BrainToText training on Modal.

This script sets up a Modal environment for training the BrainToText model,
handling dataset staging and proper import path configuration.

Usage:
    # Run locally (for testing the download logic)
    python run_on_modal.py

    # Run on Modal
    modal run run_on_modal.py
"""

import modal
import os
import sys
import zipfile
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Modal App Configuration
# ---------------------------------------------------------------------------

# Default dataset URL - users should replace this with their actual dataset URL
# The Dryad DOI link is: https://doi.org/10.5061/dryad.dncjsxm85
B2T_DATASET_URL_DEFAULT = os.environ.get(
    "B2T_DATASET_URL",
    ""  # Empty by default - must be provided by user or pre-staged in volume
)

# Create the Modal app
app = modal.App("braintotext-training")

# Define the Modal image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pandas>=2.0.0",
        "h5py>=3.0.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "tqdm>=4.60.0",
        "seaborn>=0.12.0",
        "matplotlib>=3.5.0",
    )
    .add_local_dir(".", "/app")
)

# Create a volume for persisting the dataset
dataset_volume = modal.Volume.from_name("braintotext-data", create_if_missing=True)

# Path where dataset will be mounted in Modal
MODAL_DATA_PATH = "/data/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final/"


def download_and_extract_dataset(url: str, extract_to: str) -> bool:
    """
    Download and extract dataset from a URL.

    Args:
        url: URL to the dataset archive (zip file)
        extract_to: Directory to extract the dataset to

    Returns:
        True if successful, False otherwise
    """
    if not url:
        print("No dataset URL provided. Skipping download.")
        return False

    extract_path = Path(extract_to)
    extract_path.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    if any(extract_path.iterdir()):
        print(f"Data already exists at {extract_to}. Skipping download.")
        return True

    print(f"Downloading dataset from {url}...")

    try:
        # Download the file
        zip_path = extract_path / "dataset.zip"
        urllib.request.urlretrieve(url, zip_path)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        # Clean up the zip file
        zip_path.unlink()
        print("Dataset download and extraction complete.")
        return True

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


@app.function(
    image=image,
    volumes={"/data": dataset_volume},
    gpu="any",
    timeout=3600 * 4,  # 4 hour timeout for training
)
def run_b2t_job(debug: bool = True, train: bool = True, dataset_url: str = "") -> dict:
    """
    Modal function to run BrainToText training.

    This function is named `run_b2t_job` to avoid collision with the `run_b2t` module.

    Args:
        debug: If True, use a smaller subset of data for faster training
        train: If True, train the model; otherwise, just evaluate
        dataset_url: URL to download the dataset from (if not already staged)

    Returns:
        dict with training status and any results
    """
    import sys
    import os

    # Add the app directory to the Python path
    sys.path.insert(0, "/app")

    # Set the data path environment variable
    os.environ["B2T_DATA_PATH"] = MODAL_DATA_PATH

    # Try to download dataset if URL provided and data doesn't exist
    data_path = Path(MODAL_DATA_PATH)
    if not data_path.exists() or not any(data_path.iterdir()):
        if dataset_url or B2T_DATASET_URL_DEFAULT:
            url = dataset_url or B2T_DATASET_URL_DEFAULT
            # Download to parent of the expected path structure
            download_and_extract_dataset(url, "/data")
        else:
            print(
                "WARNING: Dataset not found and no URL provided. "
                "Please provide a dataset_url or pre-stage data in the Modal volume."
            )

    # Import and run the training
    try:
        import run_b2t

        run_b2t.main(debug=debug, train=train)
        return {"status": "success", "message": "Training completed successfully"}
    except Exception as e:
        import traceback

        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


@app.local_entrypoint()
def main(
    debug: bool = True,
    train: bool = True,
    dataset_url: str = "",
):
    """
    Local entrypoint to trigger the Modal training job.

    Usage:
        modal run run_on_modal.py
        modal run run_on_modal.py --debug=False --train=True
        modal run run_on_modal.py --dataset-url="https://example.com/dataset.zip"
    """
    print("Starting BrainToText training on Modal...")
    result = run_b2t_job.remote(debug=debug, train=train, dataset_url=dataset_url)
    print(f"Training result: {result}")
    return result


if __name__ == "__main__":
    # For local testing without Modal, run directly
    print("Running locally (not on Modal)...")
    print("To run on Modal, use: modal run run_on_modal.py")

    # Set up local environment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    # For local testing, use the default local path
    # (don't override B2T_DATA_PATH if already set)
    if "B2T_DATA_PATH" not in os.environ:
        os.environ["B2T_DATA_PATH"] = os.path.join(
            script_dir, "brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final/"
        )

    import run_b2t

    run_b2t.main(debug=True, train=True)
