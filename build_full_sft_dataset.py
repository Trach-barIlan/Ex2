import json
from datasets import load_dataset
from deep_translator import GoogleTranslator
from tqdm import tqdm

def main():
    print("Downloading the English Alpaca dataset...")
    # We load the English dataset
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    # We only need about 1,000 to 2,000 high-quality examples for fine-tuning
    # (Translating all 52k would take a long time without a paid API)
    num_examples_to_translate = 1000
    subset = dataset.select(range(num_examples_to_translate))
    
    output_file = "/home/dsi/halamim1/Ex2/full_hebrew_sft_data.jsonl"
    print(f"Translating {num_examples_to_translate} answers to Hebrew and saving to {output_file}...")
    
    translator = GoogleTranslator(source='en', target='iw') # 'iw' is the code for Hebrew
    
    with open(output_file, "w", encoding="utf-8") as f:
        for row in tqdm(subset):
            instruction = row.get("instruction", "").strip()
            inp = row.get("input", "").strip()
            english_output = row.get("output", "").strip()
            
            user_content = f"{instruction}\n\n{inp}".strip()
            
            try:
                # Translate ONLY the output to Hebrew
                # (We slice the string if it's too long for the free translator limit)
                hebrew_output = translator.translate(english_output[:4999])
                
                # Format for ChatML (English User -> Hebrew Assistant)
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": hebrew_output}
                ]
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                
            except Exception as e:
                print(f"Skipping a row due to translation error: {e}")
                continue
            
    print(f"✅ Done! Saved {num_examples_to_translate} EN->HE pairs to {output_file}.")

if __name__ == "__main__":
    main()