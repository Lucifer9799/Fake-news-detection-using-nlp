"""Download the dataset into data/raw/ (run once).

    python -m src.download_data
"""
import urllib.request

from src.config import DATASET_URL, RAW_DATA_PATH


def download(force: bool = False) -> None:
    if RAW_DATA_PATH.exists() and not force:
        print(f"Dataset already present: {RAW_DATA_PATH}")
        return
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_URL}\n  -> {RAW_DATA_PATH}")
    try:
        urllib.request.urlretrieve(DATASET_URL, RAW_DATA_PATH)
    except Exception as exc:  # network errors, 404, etc.
        raise SystemExit(
            f"Download failed: {exc}\n"
            "Manual fallback: open the URL above in a browser, save the file as\n"
            f"{RAW_DATA_PATH}"
        )
    print("Done.")


if __name__ == "__main__":
    download()
