import argparse
import json
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList


class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids, vocab_size):
        self.allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)
        self.allowed_mask[allowed_token_ids] = True

    def __call__(self, input_ids, scores):
        self.allowed_mask = self.allowed_mask.to(scores.device)
        scores[:, ~self.allowed_mask] = float("-inf")
        return scores


HEBREW_RANGES = [(0x0590, 0x05FF), (0xFB1D, 0xFB4F)]


def is_hebrew_char(ch: str) -> bool:
    o = ord(ch)
    for a, b in HEBREW_RANGES:
        if a <= o <= b:
            return True
    return False


def count_non_hebrew(text: str) -> int:
    # count non-space characters that are not Hebrew
    return sum(1 for c in text if (not c.isspace()) and (not is_hebrew_char(c)))


def load_prompts(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return [l.strip() for l in fh if l.strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model vs base on Hebrewness metric")
    parser.add_argument("--prompts", type=str, default="benchmark.txt", help="One-prompt-per-line input file")
    parser.add_argument("--output", type=str, default="evaluation.jsonl", help="JSONL output file")
    parser.add_argument("--constrain", action="store_true", help="Use constrained decoding (hebrew tokens)")
    args = parser.parse_args()

    base_model_id = "Qwen/Qwen2.5-7B-Instruct"
    lora_path = "./qwen-hebrew-lora-final"

    print("Loading tokenizer and base model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", torch_dtype=torch.bfloat16)

    print(f"Loading LoRA from {lora_path}...", flush=True)
    try:
        model = PeftModel.from_pretrained(base_model, lora_path)
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        return

    logits_processor = None
    if args.constrain:
        try:
            with open("hebrew_allowed_tokens_qwen.json", "r", encoding="utf-8") as fh:
                data = json.load(fh)
                allowed_token_ids = data["allowed_token_ids"]
                if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in allowed_token_ids:
                    allowed_token_ids.append(tokenizer.eos_token_id)
                logits_processor = LogitsProcessorList([AllowedTokensLogitsProcessor(allowed_token_ids, model.config.vocab_size)])
                print("Constrained decoding enabled.", flush=True)
        except FileNotFoundError:
            print("hebrew_allowed_tokens_qwen.json not found — continuing without constraint.", flush=True)

    prompts = load_prompts(args.prompts)
    if not prompts:
        print("No prompts found — exiting.", flush=True)
        return

    gen_kwargs = {"max_new_tokens": 500, "do_sample": True, "temperature": 0.7, "top_p": 0.9}
    if logits_processor is not None:
        gen_kwargs["logits_processor"] = logits_processor

    with open(args.output, "w", encoding="utf-8") as out:
        for i, prompt in enumerate(prompts, start=1):
            print(f"[{i}/{len(prompts)}] Prompt: {prompt}", flush=True)

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # base output (disable adapters)
            with model.disable_adapter():
                base_ids = model.generate(**model_inputs, **gen_kwargs)
                base_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, base_ids)]
                base_output = tokenizer.batch_decode(base_ids, skip_special_tokens=True)[0].strip()

            # finetuned output (with adapter)
            finetuned_ids = model.generate(**model_inputs, **gen_kwargs)
            finetuned_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, finetuned_ids)]
            finetuned_output = tokenizer.batch_decode(finetuned_ids, skip_special_tokens=True)[0].strip()

            base_non_hebrew = count_non_hebrew(base_output)
            finetuned_non_hebrew = count_non_hebrew(finetuned_output)

            note = ""
            if finetuned_non_hebrew < base_non_hebrew:
                note = f"Reduced non-Hebrew chars: {base_non_hebrew} -> {finetuned_non_hebrew} (improved)"
            elif finetuned_non_hebrew > base_non_hebrew:
                note = f"Increased non-Hebrew chars: {base_non_hebrew} -> {finetuned_non_hebrew} (worse)"
            else:
                note = f"No change in non-Hebrew chars: {base_non_hebrew}"

            record = {
                "prompt": prompt,
                "base_output": base_output,
                "finetuned_output": finetuned_output,
                "base_non_hebrew": base_non_hebrew,
                "finetuned_non_hebrew": finetuned_non_hebrew,
                "notes": note,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    print(f"Evaluation complete — results in {args.output}", flush=True)


if __name__ == "__main__":
    main()
