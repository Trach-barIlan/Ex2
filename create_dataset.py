import json
from datasets import load_dataset
from deep_translator import GoogleTranslator

# 1. Load the original 100% English dataset
print("Loading the original English Alpaca dataset...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Shuffle and select a subset. 200 is plenty for a basic Colab fine-tune.
dataset = dataset.shuffle(seed=42)
num_examples = 200
sample_data = dataset.select(range(min(num_examples, len(dataset))))

# Initialize the free translator ('iw' or 'he' is the code for Hebrew)
translator = GoogleTranslator(source='en', target='iw')

output_filename = "qwen_hebrew_sft_data.jsonl"
print(f"Translating {len(sample_data)} responses to Hebrew. This will take a minute or two...")

with open(output_filename, "w", encoding="utf-8") as f:
    for i, row in enumerate(sample_data):
        
        # --- ENGLISH PROMPT ---
        user_prompt = row.get("instruction", "").strip()
        
        # Safely handle the input field and ignore "nan" strings
        input_data = str(row.get("input", "")).strip()
        if input_data and input_data.lower() != "nan":
             user_prompt += f"\n\n{input_data}"
             
        # --- HEBREW RESPONSE ---
        english_response = row.get("output", "").strip()
        
        try:
            # Translate only the output
            hebrew_response = translator.translate(english_response)
        except Exception as e:
            print(f"Skipping row {i} due to translation error: {e}")
            continue
            
        # Ensure we don't save empty rows
        if not user_prompt or not hebrew_response:
            continue
            
        # Format into Qwen's ChatML
        chat_format = {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": hebrew_response}
            ]
        }
        
        f.write(json.dumps(chat_format, ensure_ascii=False) + '\n')
        
        # Print progress so you know it hasn't frozen
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{num_examples}...")

print(f"Success! Bilingual data saved to {output_filename}.")