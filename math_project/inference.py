"""Batched greedy inference, with no reference answers passed to the model."""
import os
import time


class ModelRunner:
    def __init__(self, model, tokenizer, config, device):
        import torch
        self.torch, self.model, self.tokenizer = torch, model, tokenizer
        self.config, self.device = config, device
        self.model.eval()
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer requires a pad or EOS token")
        eos = model.generation_config.eos_token_id
        self.eos_ids = eos if isinstance(eos, list) else [eos if eos is not None else tokenizer.eos_token_id]
        if any(value is None for value in self.eos_ids):
            raise ValueError("Model requires an EOS token")
        if config["context_window_tokens"] > model.config.max_position_embeddings:
            raise ValueError("Protocol exceeds model context window")

    def generate(self, prompts, max_new_tokens=None):
        torch = self.torch
        budget = max_new_tokens or self.config["generation"]["max_new_tokens"]
        batch = self.tokenizer(prompts, padding=True, truncation=False,
                               add_special_tokens=False, return_tensors="pt")
        # Only supported model inputs; reference solutions are never included.
        batch = {k: v.to(self.device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
        width = batch["input_ids"].shape[1]
        if width + budget > self.config["context_window_tokens"]:
            raise ValueError("Prompt plus generation budget exceeds context; refusing truncation")
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        from transformers import GenerationConfig
        settings = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=budget,
                                    use_cache=True, eos_token_id=self.eos_ids,
                                    pad_token_id=self.tokenizer.pad_token_id)
        with torch.inference_mode():
            output = self.model.generate(**batch, generation_config=settings)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        results = []
        for index, tokens in enumerate(output[:, width:].tolist()):
            stop = next((i for i, value in enumerate(tokens) if value in self.eos_ids), None)
            length = stop + 1 if stop is not None else len(tokens)
            results.append({"completion": self.tokenizer.decode(tokens[:length], skip_special_tokens=True),
                            "prompt_tokens": int(batch["attention_mask"][index].sum()),
                            "generated_tokens": length, "finish_reason": "eos" if stop is not None else "length",
                            "amortized_generation_seconds": elapsed / len(prompts)})
        return results


def load_runner(bundle, device):
    # Set before torch creates a CUDA context.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
    config = bundle["config"]
    if config["generation"]["do_sample"] or config["generation"]["num_beams"] != 1:
        raise ValueError("This evaluator implements the frozen greedy protocol only")
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Enable a Kaggle GPU and rerun environment setup.")
        dtype = torch.float16
        # An actual CUDA kernel catches driver/runtime incompatibility early.
        probe = torch.ones((32, 32), device="cuda", dtype=dtype)
        assert (probe @ probe).sum().item() == 32768
        hardware = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
    else:
        dtype, hardware = torch.float32, "CPU (development test only)"
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    print(f"Loading {config['model']['id']} at {config['model']['revision']} on {hardware}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(bundle["path"] / "tokenizer", local_files_only=True, trust_remote_code=False)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(config["model"]["id"], revision=config["model"]["revision"],
                                                torch_dtype=dtype, use_safetensors=True, trust_remote_code=False,
                                                attn_implementation="eager").to(device)
    runtime = {"torch": torch.__version__, "transformers": transformers_version, "cuda_runtime": torch.version.cuda, "device": device,
               "hardware": hardware, "dtype": str(dtype), "attention": "eager",
               "deterministic_algorithms": True, "tf32": False,
               "cuda_capability": list(torch.cuda.get_device_capability(0)) if device == "cuda" else None}
    print(f"Model loaded in {time.perf_counter()-started:.1f}s", flush=True)
    return ModelRunner(model, tokenizer, config, device), runtime
