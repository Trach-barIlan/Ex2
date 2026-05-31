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


EVAL_PROMPTS = [
    # 10 Provided Prompts
    "Explain why the sky looks blue during the day.",
    "Give two advantages and two disadvantages of public transportation.",
    "Write a short email asking a professor for an extension on an assignment.",
    "Describe how to make a simple omelette.",
    "What is the difference between supervised and unsupervised learning?",
    "Summarize the story of Cinderella in three sentences.",
    "Suggest three ways to reduce smartphone distraction while studying.",
    "Explain what happens when water boils.",
    "Give a polite refusal to an invitation to a party.",
    "Turn the idea “practice makes progress” into advice for a student.",
    
    # 10 Custom Prompts
    "Explain how a refrigerator keeps food cold.",
    "Write a short haiku about the ocean.",
    "What are the main differences between a crocodile and an alligator?",
    "Give a brief overview of the plot of The Lord of the Rings.",
    "How do you say 'thank you' in five different languages?",
    "What are the primary responsibilities of a software engineer?",
    "Describe the process of photosynthesis in plants.",
    "Provide three tips for improving one's sleep schedule.",
    "Write a 2-sentence description of a futuristic city.",
    "Explain the concept of gravity to a five-year-old."
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base vs Fine-Tuned Qwen LoRA Model")
    parser.add_argument("--output", type=str, default="eval_outputs.jsonl", help="JSONL output file")
    parser.add_argument("--constrain", action="store_true", help="Use constrained decoding for the fine-tuned model")
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
                print("✅ Constrained decoding enabled for fine-tuned outputs.", flush=True)
        except FileNotFoundError:
            print("⚠️ hebrew_allowed_tokens_qwen.json not found. Running without constrained decoding.", flush=True)

    generate_kwargs_base = {
        "max_new_tokens": 500,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    generate_kwargs_finetuned = generate_kwargs_base.copy()
    if logits_processor is not None:
        generate_kwargs_finetuned["logits_processor"] = logits_processor

    with open(args.output, "w", encoding="utf-8") as output_handle:
        for index, prompt in enumerate(EVAL_PROMPTS, start=1):
            print(f"\n[{index}/{len(EVAL_PROMPTS)}] Processing prompt: {prompt}", flush=True)

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # Generate BASE model output (temporarily disable LoRA adapter)
            print("   -> Generating BASE response...", flush=True)
            with model.disable_adapter():
                base_generated_ids = model.generate(**model_inputs, **generate_kwargs_base)
                base_generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, base_generated_ids)
                ]
                base_response = tokenizer.batch_decode(base_generated_ids, skip_special_tokens=True)[0]

            # Generate FINE-TUNED model output (LoRA adapter enabled automatically)
            print("   -> Generating FINE-TUNED response...", flush=True)
            finetuned_generated_ids = model.generate(**model_inputs, **generate_kwargs_finetuned)
            finetuned_generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, finetuned_generated_ids)
            ]
            finetuned_response = tokenizer.batch_decode(finetuned_generated_ids, skip_special_tokens=True)[0]

            # Save to JSONL
            result = {
                "prompt": prompt,
                "base_output": base_response.strip(),
                "finetuned_output": finetuned_response.strip(),
                "notes": ""
            }
            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_handle.flush()

    print(f"\n✅ Done. Wrote evaluation results to {args.output}", flush=True)


if __name__ == "__main__":
    main()