import argparse
import os
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode prompts with FLUX.2-klein-9B text encoder and push to HuggingFace Hub."
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace API token (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="black-forest-labs/FLUX.2-klein-9B",
        help="HuggingFace model ID for the text encoder.",
    )
    parser.add_argument(
        "--source-dataset",
        type=str,
        default="derekl35/alphonse-mucha-style",
        help="Source dataset to load from HuggingFace Hub.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for HuggingFace caches (HF_HOME / TRANSFORMERS_CACHE).",
    )
    return parser.parse_args()


def print_gpu_info() -> None:
    """Print GPU information if nvidia-smi is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        )
        print(result.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("⚠️  nvidia-smi not found or failed – running on CPU.")


# Text encoder + tokenizer


def load_text_encoder(model_id: str, device: str):
    """Load the 4-bit quantised text encoder."""
    quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    text_encoder = AutoModelForCausalLM.from_pretrained(
        model_id,
        subfolder="text_encoder",
        quantization_config=quantization_config,
        device_map="auto",
    )

    total_params = sum(p.numel() for p in text_encoder.parameters())
    print(f"Text encoder total parameters: {total_params:,}")

    return text_encoder


def load_tokenizer(model_id: str):
    """Load the tokenizer."""
    return AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")


# Prompt encoding


def make_encode_fn(text_encoder, tokenizer, device: str):
    """Return a map-function that encodes a single example's prompt."""

    def encode_prompts(inp):
        with torch.no_grad():
            x = text_encoder.model(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": inp["text"]}],
                    return_tensors="pt",
                    thinking=False,
                    tokenize=True,
                ).to(device),
                output_hidden_states=True,
            )
        x = torch.cat([x.hidden_states[i] for i in (9, 18, 27)], dim=-1)[0]
        return {"transformer_input": x}

    return encode_prompts


# Main


def main() -> None:
    args = parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print_gpu_info()

    # ── Load text encoder & tokenizer ──────────────────────────────────
    print(f"\nLoading text encoder from {args.model_id} …")
    text_encoder = load_text_encoder(args.model_id, device)

    print(f"Loading tokenizer from {args.model_id} …")
    tokenizer = load_tokenizer(args.model_id)

    # ── Load raw dataset ───────────────────────────────────────────────
    print(f"\nLoading dataset: {args.source_dataset}")
    dataset = load_dataset(args.source_dataset)

    # ── Encode prompts ─────────────────────────────────────────────────
    print("\n Encoding prompts with text encoder …")
    encode_fn = make_encode_fn(text_encoder, tokenizer, device)
    train_dataset = dataset["train"].map(encode_fn)

    # Verify encoded shape
    sample_shape = torch.tensor(train_dataset[0]["transformer_input"]).shape
    print(f"Encoded sample shape: {sample_shape}")

    # Drop the raw text column (no longer needed)
    train_dataset = train_dataset.remove_columns(["text"])

    train_dataset.save_to_disk("/opt/ml/processing/output")
    train_dataset.push_to_hub("Pankaj121212/alphonse_mucha_encoded_dataset")

    print("Dataset encoded and saved to disk!")


if __name__ == "__main__":
    main()
