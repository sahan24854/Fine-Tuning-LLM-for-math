"""Response-only data and loss contracts for the first LoRA experiment."""
import math

from math_project.data import completion, prompt, read_jsonl


def validate_config(config):
    if (config["objective"] != "causal_lm_response_only_with_eos" or
            config["checkpoint_selection"] != "lowest_validation_response_token_nll" or
            config["validation_prompt"] != "zero_shot_teacher_forcing_same_format_as_training"):
        raise ValueError("Unsupported training or selection contract")
    if "generated_validation" in config:
        from math_project.validation_analysis import load_config
        from pathlib import Path
        expected, _ = load_config(Path(__file__).resolve().parents[1]/"configs/validation_analysis.json")
        if config["generated_validation"] != expected:
            raise ValueError("Unsupported generated-validation contract")
    for name in ("epochs", "micro_batch_size", "gradient_accumulation_steps", "eval_batch_size",
                 "max_sequence_tokens", "checkpoint_every_updates"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 < config["learning_rate"] <= .01 or not 0 <= config["warmup_ratio"] < 1:
        raise ValueError("Invalid learning rate or warmup ratio")
    if config["weight_decay"] < 0 or config["max_grad_norm"] <= 0:
        raise ValueError("Invalid optimizer settings")
    if config["lora"]["bias"] != "none" or not config["lora"]["target_modules"]:
        raise ValueError("Only the declared LoRA modules may be trained")
    if config["lora"]["r"] <= 0 or config["lora"]["lora_alpha"] <= 0 or not 0 <= config["lora"]["lora_dropout"] < 1:
        raise ValueError("Invalid LoRA configuration")
    smoke = config["smoke"]
    if any(type(smoke[key]) is not int or smoke[key] <= 0 for key in ("train_rows", "validation_rows", "max_updates")):
        raise ValueError("Invalid smoke scope")


def encode_rows(rows, tokenizer, max_tokens, check_stored=False):
    if tokenizer.eos_token_id is None:
        raise ValueError("Training requires an EOS token")
    encoded = []
    for row in rows:
        text, target = prompt(row["question"]), completion(row)
        if check_stored and (row.get("prompt") != text or row.get("completion") != target):
            raise ValueError(f"Stored training text changed: {row['id']}")
        # Respect the explicit segmentation used by Notebook 01's token audit.
        prefix = tokenizer.encode(text, add_special_tokens=False)
        response = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
        if not prefix or len(response) < 2:
            raise ValueError(f"Empty training segment: {row['id']}")
        if len(prefix)+len(response) > max_tokens:
            raise ValueError(f"{row['id']} exceeds {max_tokens} tokens; refusing truncation or silent exclusions")
        if check_stored and (row.get("prompt_token_count") != len(prefix) or row.get("completion_token_count") != len(response)):
            raise ValueError(f"Training token audit changed: {row['id']}")
        encoded.append({"input_ids": prefix+response, "attention_mask": [1]*(len(prefix)+len(response)),
                        "labels": [-100]*len(prefix)+response})
    if not encoded:
        raise ValueError("Training and validation must each contain rows")
    return encoded


def datasets_for_training(bundle, tokenizer, config, stage):
    validate_config(config)
    if stage not in ("smoke", "full"):
        raise ValueError("Unknown training stage")
    if config["max_sequence_tokens"] != bundle["config"]["max_training_tokens"]:
        raise ValueError("Training sequence budget differs from the original bundle")
    train = read_jsonl(bundle["path"]/"train.jsonl")
    validation = bundle["splits"]["validation"]
    if {r["id"] for r in train} & {r["id"] for r in validation}:
        raise ValueError("Training/validation IDs overlap")
    if stage == "smoke":
        train = train[:config["smoke"]["train_rows"]]
        validation = validation[:config["smoke"]["validation_rows"]]
    # No test split or few-shot demonstration is used in the training objective.
    encoded_train = encode_rows(train, tokenizer, config["max_sequence_tokens"], check_stored=True)
    encoded_validation = encode_rows(validation, tokenizer, config["max_sequence_tokens"])
    audit = {"stage": stage, "train_rows": len(train), "validation_rows": len(validation),
             "train_ids": [r["id"] for r in train], "validation_ids": [r["id"] for r in validation],
             "max_train_tokens": max(len(r["input_ids"]) for r in encoded_train),
             "max_validation_tokens": max(len(r["input_ids"]) for r in encoded_validation),
             "response_tokens_train": sum(sum(v != -100 for v in r["labels"]) for r in encoded_train),
             "response_tokens_validation": sum(sum(v != -100 for v in r["labels"]) for r in encoded_validation),
             "effective_batch_size": config["micro_batch_size"]*config["gradient_accumulation_steps"],
             "planned_optimizer_updates": config["smoke"]["max_updates"] if stage == "smoke" else
                math.ceil(len(train)/(config["micro_batch_size"]*config["gradient_accumulation_steps"]))*config["epochs"],
             "label_policy": "Prompt and padding masked with -100; response and appended EOS supervised; no packing or truncation."}
    return encoded_train, encoded_validation, audit


class ResponseCollator:
    def __init__(self, pad_token_id):
        if pad_token_id is None:
            raise ValueError("Padding requires a token ID")
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        width = max(len(row["input_ids"]) for row in rows)
        return {name: torch.tensor([row[name]+[pad]*(width-len(row[name])) for row in rows], dtype=torch.long)
                for name, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100))}


def response_loss(outputs, labels, num_items_in_batch=None):
    """Sum causal next-token loss; normalize over actual supervised tokens."""
    import torch.nn.functional as F
    logits = outputs.logits[..., :-1, :].contiguous().float()
    targets = labels[..., 1:].contiguous()
    total = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100, reduction="sum")
    count = num_items_in_batch if num_items_in_batch is not None else (targets != -100).sum()
    return total / count


def trainer_class():
    from transformers import Trainer
    from transformers.trainer_utils import EvalLoopOutput

    class ResponseTrainer(Trainer):
        def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                            ignore_keys=None, metric_key_prefix="eval"):
            # Token-weighted NLL: short and long solutions contribute in proportion
            # to supervised target tokens. Do not retain validation logits.
            import torch
            model = self.model
            was_training = model.training
            model.eval()
            loss_sum, token_count, examples = 0., 0, 0
            try:
                for batch in dataloader:
                    batch = self._prepare_inputs(batch)
                    labels = batch.pop("labels")
                    with torch.inference_mode(), self.compute_loss_context_manager():
                        outputs = model(**batch)
                        count = int((labels[..., 1:] != -100).sum())
                        loss_sum += float(response_loss(outputs, labels, 1))
                    token_count += count
                    examples += labels.shape[0]
                    self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)
            finally:
                model.train(was_training)
            if not token_count or not math.isfinite(loss_sum):
                raise ValueError("Nonfinite or empty validation loss")
            return EvalLoopOutput(predictions=None, label_ids=None,
                metrics={f"{metric_key_prefix}_loss": loss_sum/token_count,
                         f"{metric_key_prefix}_response_tokens": token_count}, num_samples=examples)

    return ResponseTrainer
