import torch
import os
import json
import itertools
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# ============== CONFIGURATION ==============
MODEL_NAME = "roneneldan/TinyStories-8M"
OUTPUT_DIR = "./tuning_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameter search space
# Focus on batch size 512 with learning rates around the best (1e-4 for bs=256)
# Linear scaling rule suggests lr ~2e-4 for bs=512, but also try lower/higher
LEARNING_RATES = [5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4, 3e-4]
BATCH_SIZES = [512]  # Focus on batch size 512

# Training settings for each trial (shorter for efficiency)
STEPS_PER_TRIAL = 500  # Fewer steps for faster tuning
EVAL_STEPS = 100       # Evaluate every 100 steps
WARMUP_RATIO = 0.1     # 10% warmup
MAX_LENGTH = 256       # Sequence length

# Evaluation settings
EVAL_SAMPLES = 1000    # Subset of validation set for speed

# ============== SETUP ==============
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise ValueError("No GPU available")

print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============== LOAD TOKENIZER ==============
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============== LOAD AND TOKENIZE DATASETS ==============
print("Loading datasets...")
train_dataset = load_dataset("roneneldan/TinyStories", split="train")
val_dataset = load_dataset("roneneldan/TinyStories", split="validation")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

print("Tokenizing train dataset...")
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    num_proc=4,
)

print("Tokenizing validation dataset...")
tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    num_proc=4,
)
# Use subset for faster evaluation
tokenized_val = tokenized_val.select(range(min(EVAL_SAMPLES, len(tokenized_val))))
print(f"Using {len(tokenized_val)} samples for evaluation")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ============== TUNING FUNCTION ==============
def run_trial(lr, batch_size, trial_id):
    """Run a single training trial with given hyperparameters."""
    print(f"\n{'='*60}")
    print(f"Trial {trial_id}: lr={lr}, batch_size={batch_size}")
    print(f"{'='*60}")

    # Fresh model for each trial
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    trial_output_dir = os.path.join(OUTPUT_DIR, f"trial_{trial_id}")

    training_args = TrainingArguments(
        output_dir=trial_output_dir,
        max_steps=STEPS_PER_TRIAL,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="no",  # Don't save checkpoints during tuning
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    # Evaluate before training
    print("Evaluating at step 0...")
    initial_eval = trainer.evaluate()
    initial_loss = initial_eval["eval_loss"]

    # Train
    start_time = datetime.now()
    try:
        trainer.train()
        success = True
        error_msg = None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM Error with batch_size={batch_size}")
            success = False
            error_msg = "OOM"
            torch.cuda.empty_cache()
            return {
                "trial_id": trial_id,
                "lr": lr,
                "batch_size": batch_size,
                "success": False,
                "error": "OOM",
            }
        else:
            raise e

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Final evaluation
    final_eval = trainer.evaluate()
    final_loss = final_eval["eval_loss"]

    # Get training history
    train_losses = [
        log["loss"] for log in trainer.state.log_history if "loss" in log
    ]

    result = {
        "trial_id": trial_id,
        "lr": lr,
        "batch_size": batch_size,
        "success": True,
        "initial_eval_loss": initial_loss,
        "final_eval_loss": final_loss,
        "loss_reduction": initial_loss - final_loss,
        "loss_reduction_pct": (initial_loss - final_loss) / initial_loss * 100,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "duration_seconds": duration,
        "steps_per_second": STEPS_PER_TRIAL / duration,
    }

    print(f"Initial eval loss: {initial_loss:.4f}")
    print(f"Final eval loss: {final_loss:.4f}")
    print(f"Loss reduction: {result['loss_reduction']:.4f} ({result['loss_reduction_pct']:.1f}%)")
    print(f"Duration: {duration:.1f}s ({result['steps_per_second']:.2f} steps/s)")

    # Clean up
    del model
    del trainer
    torch.cuda.empty_cache()

    return result


# ============== RUN ALL TRIALS ==============
print("\n" + "="*60)
print("STARTING HYPERPARAMETER TUNING")
print("="*60)
print(f"Learning rates: {LEARNING_RATES}")
print(f"Batch sizes: {BATCH_SIZES}")
print(f"Total trials: {len(LEARNING_RATES) * len(BATCH_SIZES)}")
print(f"Steps per trial: {STEPS_PER_TRIAL}")

all_results = []
trial_id = 0

for lr, batch_size in itertools.product(LEARNING_RATES, BATCH_SIZES):
    trial_id += 1
    result = run_trial(lr, batch_size, trial_id)
    all_results.append(result)

    # Save intermediate results
    results_path = os.path.join(OUTPUT_DIR, "tuning_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

# ============== ANALYZE RESULTS ==============
print("\n" + "="*60)
print("TUNING RESULTS SUMMARY")
print("="*60)

# Filter successful trials
successful = [r for r in all_results if r["success"]]
failed = [r for r in all_results if not r["success"]]

if failed:
    print(f"\nFailed trials ({len(failed)}):")
    for r in failed:
        print(f"  lr={r['lr']}, batch_size={r['batch_size']}: {r.get('error', 'Unknown')}")

if successful:
    # Sort by final eval loss
    successful.sort(key=lambda x: x["final_eval_loss"])

    print(f"\nSuccessful trials ({len(successful)}), sorted by final eval loss:")
    print(f"{'Rank':<6}{'LR':<12}{'Batch':<8}{'Init Loss':<12}{'Final Loss':<12}{'Reduction':<12}{'Time (s)':<10}")
    print("-" * 72)

    for i, r in enumerate(successful):
        print(f"{i+1:<6}{r['lr']:<12.0e}{r['batch_size']:<8}{r['initial_eval_loss']:<12.4f}"
              f"{r['final_eval_loss']:<12.4f}{r['loss_reduction_pct']:<12.1f}%{r['duration_seconds']:<10.1f}")

    # Best result
    best = successful[0]
    print(f"\n{'='*60}")
    print("BEST HYPERPARAMETERS")
    print(f"{'='*60}")
    print(f"Learning Rate: {best['lr']}")
    print(f"Batch Size: {best['batch_size']}")
    print(f"Final Eval Loss: {best['final_eval_loss']:.4f}")
    print(f"Loss Reduction: {best['loss_reduction_pct']:.1f}%")

    # Save best config
    best_config = {
        "learning_rate": best["lr"],
        "batch_size": best["batch_size"],
        "final_eval_loss": best["final_eval_loss"],
    }
    best_path = os.path.join(OUTPUT_DIR, "best_config.json")
    with open(best_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nBest config saved to: {best_path}")

print(f"\nAll results saved to: {results_path}")
