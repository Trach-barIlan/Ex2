import json
import re

files_to_process = {
    "qwen": "raw_tokens_qwen.json",
    "mistral": "raw_tokens_mistral.json"
}

# Strict whitelist: Hebrew blocks, digits, and standard punctuation/whitespace
ALLOWED_PATTERN = re.compile(r'^[\u0590-\u05FF\uFB1D-\uFB4F0-9\s!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]+$')

def clean_raw_token(token_str):
    """Replaces raw tokenizer space markers with standard spaces."""
    return token_str.replace('\u2581', ' ').replace('Ġ', ' ')

for name, filepath in files_to_process.items():
    print(f"Loading {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}. Ensure it is in the same directory.")
        continue
        
    # --- THE FIX IS HERE ---
    # We must explicitly iterate over the items inside the "raw_tokens" dictionary
    if "raw_tokens" in vocab_data:
        vocab = vocab_data["raw_tokens"]
    else:
        print(f"Error: 'raw_tokens' key not found in {filepath}. Check file format.")
        continue
        
    allowed_token_ids = []
    
    for token_str, token_id in vocab.items():
        # Clean the special space characters
        cleaned_str = clean_raw_token(token_str)
        
        # Skip tokens that are empty after cleaning
        if not cleaned_str:
            continue
            
        # Evaluate the cleaned string against the whitelist
        if ALLOWED_PATTERN.match(cleaned_str):
            allowed_token_ids.append(token_id)
            
    # Sort IDs sequentially
    allowed_token_ids.sort()
    
    # Grab the model_id dynamically from your JSON file
    model_id = vocab_data.get("model_id", "Unknown_Model")
    
    output_data = {
        "model_id": model_id,
        "allowed_token_ids": allowed_token_ids
    }
    
    # Save the results
    output_filename = f"hebrew_allowed_tokens_{name}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Success! Saved {len(allowed_token_ids)} tokens to {output_filename}\n")