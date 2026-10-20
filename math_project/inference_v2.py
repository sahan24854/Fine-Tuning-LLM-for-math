"""Per-sequence answer-boundary stopping layered over the frozen base-model runtime."""
import time

from math_project.protocol_v2 import boundary


def stopping_criteria(tokenizer, prompt_width, eos_ids):
    import torch
    from transformers import StoppingCriteria

    class TargetBoundary(StoppingCriteria):
        def __init__(self):
            self.finished = {}

        def __call__(self, input_ids, scores, **kwargs):
            for index, tokens in enumerate(input_ids[:, prompt_width:].tolist()):
                if index in self.finished:
                    continue
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                event = boundary(text)
                if event is None and tokens and tokens[-1] in eos_ids:
                    event = {"reason": "eos", "cut": len(text), "observed_at": len(text)}
                if event:
                    self.finished[index] = {**event, "generated_tokens": len(tokens), "raw_completion": text}
            return torch.tensor([i in self.finished for i in range(input_ids.shape[0])],
                                device=input_ids.device, dtype=torch.bool)

    return TargetBoundary()


class BoundedRunner:
    def __init__(self, base, protocol):
        self.base, self.protocol = base, protocol
        self.device, self.torch = base.device, base.torch

    def generate(self, prompts, max_new_tokens=None):
        from transformers import GenerationConfig, StoppingCriteriaList
        base, torch = self.base, self.torch
        budget = max_new_tokens or self.protocol["generation"]["max_new_tokens"]
        batch = base.tokenizer(prompts, padding=True, truncation=False,
                               add_special_tokens=False, return_tensors="pt")
        batch = {k: value.to(self.device) for k, value in batch.items() if k in ("input_ids", "attention_mask")}
        width = batch["input_ids"].shape[1]
        if width + budget > base.config["context_window_tokens"]:
            raise ValueError("Prompt plus diagnostic budget exceeds context; refusing truncation")
        criterion = stopping_criteria(base.tokenizer, width, base.eos_ids)
        settings = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=budget, use_cache=True,
                                    eos_token_id=base.eos_ids, pad_token_id=base.tokenizer.pad_token_id,
                                    bos_token_id=base.model.generation_config.bos_token_id)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = base.model.generate(**batch, generation_config=settings,
                                             stopping_criteria=StoppingCriteriaList([criterion]))
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        results = []
        for index, tokens in enumerate(generated[:, width:].tolist()):
            event = criterion.finished.get(index)
            if event is None:
                eos = next((i for i, value in enumerate(tokens) if value in base.eos_ids), None)
                length = eos+1 if eos is not None else len(tokens)
                raw = base.tokenizer.decode(tokens[:length], skip_special_tokens=True)
                event = boundary(raw) or {"reason": "eos" if eos is not None else "length", "cut": len(raw)}
                event = {**event, "generated_tokens": length, "raw_completion": raw}
            results.append({"completion": event["raw_completion"][:event["cut"]],
                            "raw_completion": event["raw_completion"],
                            "finish_reason": event["reason"], "generated_tokens": event["generated_tokens"],
                            "prompt_tokens": int(batch["attention_mask"][index].sum()),
                            "amortized_generation_seconds": elapsed/len(prompts)})
        return results
