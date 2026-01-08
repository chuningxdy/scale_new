"""
Download opengenome2 dataset and convert to packed binary format.
Memory-efficient: streams directly to disk without loading into RAM.

Usage:
    pip install hf_transfer
    export HF_HUB_ENABLE_HF_TRANSFER=1
    python download_opengenome2_fast.py
"""

import os
import gzip
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

# Configuration
OUTPUT_DIR = "/mfs1/datasets/pile/opengenome2_16gb"
TARGET_SIZE_GB = 16
BYTES_PER_GB = 1024 ** 3
TARGET_TOKENS = TARGET_SIZE_GB * BYTES_PER_GB  # ~16 billion tokens (1 byte per token in final)
SEQ_LEN = 2048  # Sequence length for packed binary
CACHE_DIR = "/mfs1/datasets/pile/huggingface"

# Char-level vocab for DNA
VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
UNK = 4

# Lookup table for fast vectorized conversion (ASCII -> token)
CHAR_TO_TOKEN = np.zeros(256, dtype=np.uint16) + UNK  # default to UNK
CHAR_TO_TOKEN[ord('A')] = 0
CHAR_TO_TOKEN[ord('C')] = 1
CHAR_TO_TOKEN[ord('G')] = 2
CHAR_TO_TOKEN[ord('T')] = 3
CHAR_TO_TOKEN[ord('N')] = 4
CHAR_TO_TOKEN[ord('a')] = 0
CHAR_TO_TOKEN[ord('c')] = 1
CHAR_TO_TOKEN[ord('g')] = 2
CHAR_TO_TOKEN[ord('t')] = 3
CHAR_TO_TOKEN[ord('n')] = 4

def fasta_gz_to_packed_bin(fasta_gz_path, out_bin_path, seq_len=2048, max_tokens=None):
    """
    Convert gzipped FASTA to packed binary format.
    Memory-efficient: streams directly to disk.

    Returns:
        (tokens_written, num_sequences)
    """
    buf = np.empty(seq_len, dtype=np.uint16)
    i = 0
    written = 0

    with gzip.open(fasta_gz_path, "rt") as f, open(out_bin_path, "wb") as out:
        for line in tqdm(f, desc="Converting to binary", unit=" lines"):
            if not line or line[0] == ">":
                continue
            s = line.strip().upper()
            for ch in s:
                buf[i] = VOCAB.get(ch, UNK)
                i += 1
                written += 1
                if i == seq_len:
                    out.write(buf.tobytes())
                    i = 0
                if max_tokens is not None and written >= max_tokens:
                    return written, (written // seq_len)

    return written, (written // seq_len)


def fasta_gz_to_packed_bin_multi(fasta_paths, out_bin_path, seq_len=2048, max_tokens=None):
    """
    Convert multiple gzipped FASTA files to a single packed binary file.
    Memory-efficient: streams directly to disk.

    Returns:
        (tokens_written, num_sequences)
    """
    buf = np.empty(seq_len, dtype=np.uint16)
    i = 0
    written = 0

    with open(out_bin_path, "wb") as out:
        for fasta_gz_path in fasta_paths:
            print(f"\nProcessing: {os.path.basename(fasta_gz_path)}")

            with gzip.open(fasta_gz_path, "rt") as f:
                for line in tqdm(f, desc="Converting", unit=" lines"):
                    if not line or line[0] == ">":
                        continue
                    s = line.strip().upper()
                    for ch in s:
                        buf[i] = VOCAB.get(ch, UNK)
                        i += 1
                        written += 1
                        if i == seq_len:
                            out.write(buf.tobytes())
                            i = 0
                        if max_tokens is not None and written >= max_tokens:
                            print(f"\nReached target of {max_tokens:,} tokens")
                            return written, (written // seq_len)

            print(f"  Progress: {written:,} tokens ({written / 1e9:.2f}B), {written // seq_len:,} sequences")

    return written, (written // seq_len)


def main():
    print(f"Downloading ~{TARGET_SIZE_GB}GB of opengenome2 to {OUTPUT_DIR}")
    print(f"Converting to packed binary format (seq_len={SEQ_LEN})")
    print("Memory-efficient: streaming directly to disk")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # List available files
    print("\nListing available files in arcinstitute/opengenome2...")
    try:
        files = list_repo_files("arcinstitute/opengenome2", repo_type="dataset")
    except Exception as e:
        print(f"Error listing files: {e}")
        print("You may need to run: huggingface-cli login")
        return

    # Filter for FASTA files
    fasta_files = sorted([f for f in files if f.endswith('.fasta.gz')])

    # Prefer metagenome files
    metagenome_files = [f for f in fasta_files if 'metagenome' in f.lower()]
    if metagenome_files:
        fasta_files = metagenome_files

    print(f"Found {len(fasta_files)} FASTA files")
    for f in fasta_files[:5]:
        print(f"  {f}")
    if len(fasta_files) > 5:
        print(f"  ... and {len(fasta_files) - 5} more")

    # Download and process FASTA files one at a time (stop when target reached)
    out_bin_path = os.path.join(OUTPUT_DIR, f"opengenome2_{SEQ_LEN}_uint16.bin")
    print(f"\nConverting to packed binary: {out_bin_path}")
    print(f"Target: {TARGET_TOKENS:,} tokens ({TARGET_SIZE_GB}GB)")

    downloaded_files = []
    buf = np.empty(SEQ_LEN, dtype=np.uint16)
    buf_idx = 0
    tokens = 0

    with open(out_bin_path, "wb") as out:
        for fasta_file in fasta_files:
            # Check if we've reached target
            if tokens >= TARGET_TOKENS:
                print(f"\nReached target of {TARGET_TOKENS:,} tokens")
                break

            # Download this file
            print(f"\nDownloading: {fasta_file}")
            try:
                local_path = hf_hub_download(
                    repo_id="arcinstitute/opengenome2",
                    filename=fasta_file,
                    repo_type="dataset",
                    cache_dir=CACHE_DIR
                )
                downloaded_files.append(local_path)
                print(f"  Cached at: {local_path}")
            except Exception as e:
                print(f"  Error downloading: {e}")
                continue

            # Process this file (vectorized for speed)
            print(f"  Processing...")
            with gzip.open(local_path, "rt") as f:
                for line in tqdm(f, desc="Converting", unit=" lines"):
                    if not line or line[0] == ">":
                        continue
                    # Vectorized conversion: string -> bytes -> numpy -> lookup
                    line_bytes = np.frombuffer(line.strip().encode('ascii'), dtype=np.uint8)
                    line_tokens = CHAR_TO_TOKEN[line_bytes]

                    # Write in chunks
                    pos = 0
                    while pos < len(line_tokens):
                        space_left = SEQ_LEN - buf_idx
                        chunk_size = min(space_left, len(line_tokens) - pos)
                        buf[buf_idx:buf_idx + chunk_size] = line_tokens[pos:pos + chunk_size]
                        buf_idx += chunk_size
                        pos += chunk_size
                        tokens += chunk_size

                        if buf_idx == SEQ_LEN:
                            out.write(buf.tobytes())
                            buf_idx = 0

                        if tokens >= TARGET_TOKENS:
                            break

                    if tokens >= TARGET_TOKENS:
                        break

            print(f"  Progress: {tokens:,} tokens ({tokens / 1e9:.2f}B), {tokens // SEQ_LEN:,} sequences")

    num_seqs = tokens // SEQ_LEN

    if len(downloaded_files) == 0:
        print("No files downloaded!")
        return

    # Calculate actual file size
    file_size = os.path.getsize(out_bin_path)

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Output file: {out_bin_path}")
    print(f"  Total tokens: {tokens:,} ({tokens / 1e9:.2f}B)")
    print(f"  Total sequences: {num_seqs:,}")
    print(f"  File size: {file_size / BYTES_PER_GB:.2f} GB")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Format: uint16 packed binary")
    print(f"{'='*60}")

    # Save metadata
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write(f"Source: arcinstitute/opengenome2\n")
        f.write(f"Format: uint16 packed binary\n")
        f.write(f"Sequence length: {SEQ_LEN}\n")
        f.write(f"Total tokens: {tokens}\n")
        f.write(f"Total sequences: {num_seqs}\n")
        f.write(f"File size (bytes): {file_size}\n")
        f.write(f"Vocab: A=0, C=1, G=2, T=3, N=4\n")
        f.write(f"\nSource files:\n")
        for df in downloaded_files:
            f.write(f"  - {os.path.basename(df)}\n")

    # Also save vocab info
    vocab_path = os.path.join(OUTPUT_DIR, "vocab.txt")
    with open(vocab_path, "w") as f:
        for char, idx in sorted(VOCAB.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{char}\n")

    print(f"\nMetadata saved to {metadata_path}")
    print(f"Vocab saved to {vocab_path}")

    print(f"\nTo load this data in PyTorch:")
    print(f"""
import numpy as np
import torch

# Load binary file
data = np.memmap('{out_bin_path}', dtype=np.uint16, mode='r')
data = data.reshape(-1, {SEQ_LEN})

# Convert to torch tensor (lazy loading)
print(f"Total sequences: {{len(data):,}}")
print(f"Sequence shape: {{data.shape}}")

# Access a batch
batch = torch.from_numpy(data[0:32].copy()).long()
print(f"Batch shape: {{batch.shape}}")
""")

if __name__ == "__main__":
    main()
