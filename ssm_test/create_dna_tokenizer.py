"""
Create a single-nucleotide DNA tokenizer for genomics data (Evo2-style)

Vocabulary:
- A, C, G, T: Standard nucleotides
- N: Unknown/ambiguous nucleotide
- Special tokens: [PAD], [UNK], [BOS], [EOS]
"""

import os
import json
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast

def create_dna_tokenizer(save_path: str = "datasets/tokenizers/dna_single_nucleotide"):
    """
    Create a single-nucleotide DNA tokenizer compatible with HuggingFace.

    Vocabulary (8 tokens):
        0: [PAD] - Padding token
        1: [UNK] - Unknown token
        2: [BOS] - Beginning of sequence
        3: [EOS] - End of sequence
        4: A - Adenine
        5: C - Cytosine
        6: G - Guanine
        7: T - Thymine
        8: N - Unknown/ambiguous base
    """

    # Define vocabulary
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "A": 4,
        "C": 5,
        "G": 6,
        "T": 7,
        "N": 8,
        # Also add lowercase versions mapping to same IDs
        "a": 4,
        "c": 5,
        "g": 6,
        "t": 7,
        "n": 8,
    }

    # Create a WordLevel tokenizer (character-level for DNA)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))

    # Pre-tokenizer: split into individual characters
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")

    # Add special tokens handling
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", vocab["[BOS]"]),
            ("[EOS]", vocab["[EOS]"]),
        ],
    )

    # Create the save directory
    os.makedirs(save_path, exist_ok=True)

    # Save the base tokenizer
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))

    # Wrap in HuggingFace PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        clean_up_tokenization_spaces=False,
    )

    # Save in HuggingFace format
    hf_tokenizer.save_pretrained(save_path)

    print(f"DNA tokenizer saved to: {save_path}")
    print(f"Vocabulary size: {hf_tokenizer.vocab_size}")
    print(f"Vocabulary: {hf_tokenizer.get_vocab()}")

    # Test the tokenizer
    test_sequences = [
        "ACGTACGT",
        "NNACGT",
        "acgtACGT",  # Mixed case
        "ATCGATCGATCG",
    ]

    print("\nTest tokenization:")
    for seq in test_sequences:
        tokens = hf_tokenizer.tokenize(seq)
        ids = hf_tokenizer.encode(seq)
        decoded = hf_tokenizer.decode(ids)
        print(f"  '{seq}' -> tokens: {tokens}")
        print(f"           -> ids: {ids}")
        print(f"           -> decoded: '{decoded}'")

    return hf_tokenizer


def create_dna_tokenizer_simple(save_path: str = "datasets/tokenizers/dna_single_nucleotide"):
    """
    Create an even simpler DNA tokenizer without BOS/EOS added automatically.
    Better for language modeling where we handle special tokens separately.
    """

    # Define vocabulary (smaller, just what we need)
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "A": 4,
        "C": 5,
        "G": 6,
        "T": 7,
        "N": 8,
    }

    # Create a WordLevel tokenizer
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))

    # Pre-tokenizer: split into individual characters
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")

    # No post-processor (don't auto-add BOS/EOS)

    # Create the save directory
    os.makedirs(save_path, exist_ok=True)

    # Save the base tokenizer
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))

    # Wrap in HuggingFace PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=131072,  # Support long sequences
        clean_up_tokenization_spaces=False,
    )

    # Save in HuggingFace format
    hf_tokenizer.save_pretrained(save_path)

    print(f"DNA tokenizer (simple) saved to: {save_path}")
    print(f"Vocabulary size: {hf_tokenizer.vocab_size}")
    print(f"Vocabulary: {hf_tokenizer.get_vocab()}")

    # Test the tokenizer
    test_sequences = [
        "ACGTACGT",
        "NNACGT",
        "ATCGATCGATCG",
    ]

    print("\nTest tokenization:")
    for seq in test_sequences:
        tokens = hf_tokenizer.tokenize(seq)
        ids = hf_tokenizer.encode(seq, add_special_tokens=False)
        decoded = hf_tokenizer.decode(ids)
        print(f"  '{seq}'")
        print(f"    -> tokens: {tokens}")
        print(f"    -> ids: {ids}")
        print(f"    -> decoded: '{decoded}'")

    return hf_tokenizer


if __name__ == "__main__":
    # Create the simple tokenizer (recommended for LM training)
    save_path = "datasets/tokenizers/dna_single_nucleotide"
    tokenizer = create_dna_tokenizer_simple(save_path)

    print(f"\n{'='*60}")
    print(f"Tokenizer ready to use!")
    print(f"Path: {save_path}")
    print(f"Usage in nn_dict.json:")
    print(f'  {{"data": "opengenome2_stream", "model": "ssm"}}')
    print(f"{'='*60}")
