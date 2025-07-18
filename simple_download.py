#!/usr/bin/env python3
"""
Download TLDR dataset from Hugging Face and save in a format
compatible with `datasets.load_from_disk`.
"""

import pandas as pd
from pathlib import Path
import requests
from datasets import Dataset, DatasetDict

def download_tldr_hf():
    """Download and save TLDR dataset in HuggingFace format."""

    data_dir = Path("data_hf")
    data_dir.mkdir(exist_ok=True)

    print("Downloading TLDR dataset...")

    urls = {
        "train": "https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4/resolve/main/data/train-00000-of-00001.parquet",
        # Add more splits if available
    }

    dataset_splits = {}

    for split_name, url in urls.items():
        print(f"Downloading {split_name} split...")

        try:
            response = requests.get(url)
            response.raise_for_status()

            tmp_file = data_dir / f"tmp_{split_name}.parquet"
            with open(tmp_file, 'wb') as f:
                f.write(response.content)

            df = pd.read_parquet(tmp_file)
            print(f"Saved {len(df)} examples to {tmp_file}")
            print(f"  Columns: {list(df.columns)}")

            dataset_splits[split_name] = Dataset.from_pandas(df)

            tmp_file.unlink()  # delete temp parquet

        except Exception as e:
            print(f"Error downloading {split_name}: {e}")

    # Save in HuggingFace `DatasetDict` format
    dataset = DatasetDict(dataset_splits)
    dataset.save_to_disk(str(data_dir / "countdown_dataset"))

    print("\nDone! You can now load it with:")
    print(f"from datasets import load_from_disk\n"
          f"dataset = load_from_disk('{data_dir}/countdown_dataset')")

if __name__ == "__main__":
    download_tldr_hf()
