import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from peft import PeftModel

class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids, vocab_size):
        self.allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)
        self.allowed_mask[allowed_token_ids] = True

    def __call__(self, input_ids, scores):
        self.allowed_mask = self.allowed_mask.to(scores.device)
        scores[:, ~self.allowed_mask] = float('-inf')
        return scores

def main():
    parser = argparse.ArgumentParser(description="Test Full Qwen LoRA Model")
    parser.add_argument("--prompt", type=str, default="What are the three primary colors?", help="The English prompt to translate/answer in Hebrew")
    parser.add_argument("--constrain", action="store_true", help="Use constrained decoding to force only allowed Hebrew tokens")
    args = parser.parse_args()

    base_model_id = "Qwen/Qwen2.5-7B-Instruct"
    lora_path = "./qwen-hebrew-lora-final"

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    print(f"Loading LoRA weights from {lora_path}...")
    try:
        model = PeftModel.from_pretrained(base_model, lora_path)
    except Exception as e:
        print(f"❌ Error loading LoRA from {lora_path}. Did the full training finish successfully?\nDetails: {e}")
        return

    logits_processor = None
    if args.constrain:
        try:
            with open('hebrew_allowed_tokens_qwen.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                allowed_token_ids = data['allowed_token_ids']
                if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in allowed_token_ids:
                    allowed_token_ids.append(tokenizer.eos_token_id)
                logits_processor = LogitsProcessorList([
                    AllowedTokensLogitsProcessor(allowed_token_ids, model.config.vocab_size)
                ])
                print("✅ Constrained decoding enabled.")
        except FileNotFoundError:
            print("⚠️ hebrew_allowed_tokens_qwen.json not found. Running without constrained decoding.")

    print(f"\n--- Prompt ---\n{args.prompt}\n")

    messages = [
        {"role": "user", "content": args.prompt}
    ]
    
    # Apply chat template and add the generation prompt (which sets up the assistant role)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("Generating response...")
    generate_kwargs = {
        "max_new_tokens": 500,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9
    }
    if logits_processor is not None:
        generate_kwargs["logits_processor"] = logits_processor

    generated_ids = model.generate(
        **model_inputs,
        **generate_kwargs
    )
    
    # Slice the generated ids to remove the prompt tokens
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"--- Response ---\n{response}\n")

if __name__ == "__main__":
    main()
