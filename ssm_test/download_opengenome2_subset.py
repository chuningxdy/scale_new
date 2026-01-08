"""
Download a subset of opengenome2 dataset (~16GB) for local training.
Saves to /mfs1/datasets/pile/opengenome2_16gb/
"""

import os
from datasets import load_dataset
from tqdm import tqdm

# Configuration
OUTPUT_DIR = "/mfs1/datasets/pile/opengenome2_16gb"
TARGET_SIZE_GB = 16
BYTES_PER_GB = 1024 ** 3

# Estimate: each sequence is ~10KB on average (varies widely)
# 16GB / 10KB = ~1.6M sequences, but let's be conservative
# We'll track actual size as we go
TARGET_SIZE_BYTES = TARGET_SIZE_GB * BYTES_PER_GB

def main():
    print(f"Downloading ~{TARGET_SIZE_GB}GB of opengenome2 to {OUTPUT_DIR}")
    print("This may take a while...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load as streaming to avoid downloading everything
    print("\nLoading dataset in streaming mode...")
    print("Note: If you get auth errors, run: huggingface-cli login")
    dataset = load_dataset(
        "arcinstitute/opengenome2",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    # Collect sequences until we hit target size
    sequences = []
    total_bytes = 0

    print(f"\nCollecting sequences until ~{TARGET_SIZE_GB}GB...")
    pbar = tqdm(total=TARGET_SIZE_BYTES, unit='B', unit_scale=True, desc="Downloaded")

    try:
        for i, example in enumerate(dataset):
            # Get the sequence
            seq = example.get("sequence", "")
            if not seq:
                continue

            # Estimate size (sequence + overhead)
            seq_bytes = len(seq.encode('utf-8'))

            sequences.append({"sequence": seq})
            total_bytes += seq_bytes
            pbar.update(seq_bytes)

            # Progress update every 10k sequences
            if (i + 1) % 10000 == 0:
                print(f"\n  Collected {len(sequences):,} sequences ({total_bytes / BYTES_PER_GB:.2f} GB)")

            # Check if we've reached target
            if total_bytes >= TARGET_SIZE_BYTES:
                print(f"\n\nReached target size!")
                break

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Saving {len(sequences):,} sequences collected so far...")
    finally:
        pbar.close()

    if len(sequences) == 0:
        print("No sequences collected!")
        return

    print(f"\nCollected {len(sequences):,} sequences ({total_bytes / BYTES_PER_GB:.2f} GB)")

    # Convert to HuggingFace Dataset and save
    print(f"\nConverting to HuggingFace Dataset format...")
    from datasets import Dataset

    hf_dataset = Dataset.from_list(sequences)

    print(f"Dataset info:")
    print(f"  - Number of examples: {len(hf_dataset):,}")
    print(f"  - Features: {hf_dataset.features}")

    # Save to disk
    print(f"\nSaving to {OUTPUT_DIR}...")
    hf_dataset.save_to_disk(OUTPUT_DIR)

    print(f"\nDone! Dataset saved to {OUTPUT_DIR}")
    print(f"Total size: {total_bytes / BYTES_PER_GB:.2f} GB")
    print(f"Number of sequences: {len(sequences):,}")

    # Also save some metadata
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write(f"Source: arcinstitute/opengenome2\n")
        f.write(f"Number of sequences: {len(sequences)}\n")
        f.write(f"Total size (approx): {total_bytes / BYTES_PER_GB:.2f} GB\n")
        f.write(f"Subset: first {len(sequences)} sequences from train split\n")

    print(f"\nTo use this dataset, update nn_dict.json:")
    print('  {"data": "opengenome2_local", "model": "hyena"}')

if __name__ == "__main__":
    main()
