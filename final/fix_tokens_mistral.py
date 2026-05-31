import json
import re
from transformers import AutoTokenizer

ALLOWED_PATTERN = re.compile(r'^[\u0590-\u05FF\uFB1D-\uFB4F0-9\s!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]+$')

tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')

allowed_ids = []
for token_id in range(tokenizer.vocab_size):
    try:
        # Decode the token ID to get its string representation
        # It's important to not skip special tokens or we can if we want,
        # but the safest is to decode standard.
        token_str = tokenizer.decode([token_id])
    except Exception:
        continue
    
    # Check if the string matches the pattern
    if ALLOWED_PATTERN.match(token_str):
        allowed_ids.append(token_id)

print(f"Found {len(allowed_ids)} allowed tokens for Mistral.")

# Always include EOS and other special tokens just in case
for special_id in tokenizer.all_special_ids:
    if special_id not in allowed_ids:
        allowed_ids.append(special_id)

allowed_ids.sort()

with open('hebrew_allowed_tokens_mistral.json', 'w') as f:
    json.dump({
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "allowed_token_ids": allowed_ids
    }, f, indent=4)
