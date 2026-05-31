import json

def main():
    output_file = "/home/dsi/halamim1/Ex2/overfit_sft_data.jsonl"
    print("Creating a tiny overfit dataset (English IN -> Hebrew OUT)...")
    
    # 3 simple examples to overfit the model
    overfit_examples = [
        {"en_prompt": "Explain why leaves are green.", "he_answer": "עלים נראים ירוקים בגלל שהם מכילים כלורופיל, פיגמנט שקולט אור שמש ומשתמש בו לתהליך הפוטוסינתזה."},
        {"en_prompt": "What is the capital of France?", "he_answer": "עיר הבירה של צרפת היא פריז."},
        {"en_prompt": "Write a short poem about a cat.", "he_answer": "חתול קטן, שחור ולבן, יושב על החלון כל הזמן וישן."}
    ]
    
    with open(output_file, "w", encoding="utf-8") as f:
        for ex in overfit_examples:
            messages = [
                {"role": "user", "content": ex["en_prompt"]},
                {"role": "assistant", "content": ex["he_answer"]}
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            
    print(f"✅ Done! Saved {len(overfit_examples)} overfit examples to {output_file}.")

if __name__ == "__main__":
    main()
