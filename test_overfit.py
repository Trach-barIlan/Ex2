import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    base_model_id = "Qwen/Qwen2.5-7B-Instruct"
    lora_path = "./qwen-hebrew-lora-overfit-final"

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    print(f"Loading LoRA weights from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    # We ask the exact question from the first line of the JSONL file.
    # Since it's overfit, it should output the exact memorized Hebrew response.
    prompt = "Give three tips for a good sleep."
    
    print(f"\n--- Prompt ---\n{prompt}\n")

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template and add the generation prompt (which sets up the assistant role)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("Generating response...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=200,
        do_sample=False # Greedy decoding for exact memorized match
    )
    
    # Slice the generated ids to remove the prompt tokens
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"--- Response ---\n{response}\n")

if __name__ == "__main__":
    main()
