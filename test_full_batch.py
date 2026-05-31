import argparse
import json

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


def read_prompts(input_path):
    with open(input_path, "r", encoding="utf-8") as file_handle:
        return [line.strip() for line in file_handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Test Full Qwen LoRA Model on a batch of prompts")
    parser.add_argument("--input", type=str, default="benchmark.txt", help="Text file with one prompt per line")
    parser.add_argument("--output", type=str, default="responses.txt", help="Text file to write query/response pairs")
    parser.add_argument("--constrain", action="store_true", help="Use constrained decoding to force only allowed Hebrew tokens")
    args = parser.parse_args()

    base_model_id = "Qwen/Qwen2.5-7B-Instruct"
    lora_path = "./qwen-hebrew-lora-final"

    print("Loading base model and tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA weights from {lora_path}...", flush=True)
    try:
        model = PeftModel.from_pretrained(base_model, lora_path)
    except Exception as error:
        print(f"❌ Error loading LoRA from {lora_path}. Did the full training finish successfully?\nDetails: {error}")
        return

    logits_processor = None
    if args.constrain:
        try:
            with open("hebrew_allowed_tokens_qwen.json", "r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
                allowed_token_ids = data["allowed_token_ids"]
                if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in allowed_token_ids:
                    allowed_token_ids.append(tokenizer.eos_token_id)
                logits_processor = LogitsProcessorList(
                    [AllowedTokensLogitsProcessor(allowed_token_ids, model.config.vocab_size)]
                )
                print("✅ Constrained decoding enabled.", flush=True)
        except FileNotFoundError:
            print("⚠️ hebrew_allowed_tokens_qwen.json not found. Running without constrained decoding.", flush=True)

    prompts = read_prompts(args.input)
    print(f"Loaded {len(prompts)} prompts from {args.input}.", flush=True)

    generate_kwargs = {
        "max_new_tokens": 500,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    if logits_processor is not None:
        generate_kwargs["logits_processor"] = logits_processor

    with open(args.output, "w", encoding="utf-8") as output_handle:
        for index, prompt in enumerate(prompts, start=1):
            print(f"[{index}/{len(prompts)}] Processing prompt...", flush=True)
            output_handle.write(f"Query: {prompt}\n")

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            print(f"[{index}/{len(prompts)}] Generating response...", flush=True)
            generated_ids = model.generate(**model_inputs, **generate_kwargs)

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            output_handle.write(f"Response: {response}\n\n")
            output_handle.flush()

    print(f"Done. Wrote responses to {args.output}", flush=True)


if __name__ == "__main__":
    main()