"""Utility functions for the styletts2 plugin."""
import os

import phonemizer
import requests
from nendo import NendoError
from tqdm import tqdm

global_phonemizer = phonemizer.backend.EspeakBackend(
    language="en-us", preserve_punctuation=True, with_stress=True
)


def download_model(download_url: str, model_path: str) -> None:
    """Load the model from the given download url to the given path.

    Args:
        download_url (str): The url to download the model from.
        model_path (str): The path to save the model to.
    """
    with requests.get(download_url, stream=True, timeout=2000) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(model_path, "wb") as f:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            os.remove(model_path)
            raise NendoError(f"Error while downloading {model_path}.")
