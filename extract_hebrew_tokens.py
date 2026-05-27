import json
import re
import unicodedata
from transformers import AutoTokenizer

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

UNICODE_TO_BYTE = {v: k for k, v in bytes_to_unicode().items()}

def get_token_bytes(token_string, model_id):
    if "Qwen" in model_id:
        # Qwen uses ByteLevel BPE, translating back to explicit bytes
        return bytes([UNICODE_TO_BYTE.get(c, ord(c)) for c in token_string])
    else:
        # Mistral uses SentencePiece/TikToken raw fallback and U+2581 for space
        m = re.fullmatch(r'<0x([0-9A-Fa-f]{2})>', token_string)
        if m:
            return bytes([int(m.group(1), 16)])
        s = token_string.replace('\u2581', ' ')
        return s.encode('utf-8')

ALLOWED_LEADING_BYTES = {0xC2, 0xD6, 0xD7, 0xE2, 0xF0, 0xF1, 0xF2, 0xF3, 0xF4}
# שנה ל-False אם תרצה להשאיר גם סימני פיסוק, רווחים ומספרים טהורים שאינם מכילים עברית
REQUIRE_HEBREW_CHAR = True

def is_hebrew_allowed(token_string, model_id):
    token_bytes = get_token_bytes(token_string, model_id)
    
    is_partial_byte = False
    # 1. Handle explicit byte fallback tokens (e.g., Mistral)
    if re.fullmatch(r'<0x[0-9A-Fa-f]{2}>', token_string):
        b = token_bytes[0]
        if b in (0xD6, 0xD7) or 0x90 <= b <= 0xBF:
            is_partial_byte = True
        else:
            return False

    try:
        decoded_string = token_bytes.decode('utf-8', errors='strict')
    except UnicodeDecodeError:
        for b in token_bytes:
            if b >= 0x80:
                if not (b in (0xD6, 0xD7) or 0x90 <= b <= 0xBF):
                    return False
        is_partial_byte = True
        decoded_string = token_bytes.decode('utf-8', errors='ignore')
        
    has_hebrew_letter = False
    for char in decoded_string:
        cat = unicodedata.category(char)
        if '\u0590' <= char <= '\u05FF':
            has_hebrew_letter = True
            continue
        if cat.startswith('L'):
            return False
        if cat.startswith('C') and char not in ['\n', '\r', '\t']:
            return False
            
    if REQUIRE_HEBREW_CHAR:
        return has_hebrew_letter or is_partial_byte
    else:
        return True

def extract_and_save_tokens(model_id, output_filename, raw_output_filename):
    print(f"Extracting tokens for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    vocab = tokenizer.get_vocab()
    
    # --- שמירת המילון המקורי (Raw Tokens) ללא סינון ---
    sorted_raw_vocab = dict(sorted(vocab.items(), key=lambda item: item[1]))
    raw_output_data = {
        "model_id": model_id,
        "total_vocab_size": len(vocab),
        "raw_tokens": sorted_raw_vocab
    }
    with open(raw_output_filename, 'w', encoding='utf-8') as f:
        json.dump(raw_output_data, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(sorted_raw_vocab)} raw tokens to {raw_output_filename}")

    # --- שמירת המילון המסונן (Hebrew Allowed Tokens) ---
    allowed_tokens = {}
    for token_string, token_id in vocab.items():
        if is_hebrew_allowed(token_string, model_id):
            allowed_tokens[token_string] = token_id
            
    sorted_allowed_tokens = dict(sorted(allowed_tokens.items(), key=lambda item: item[1]))
    
    output_data = {
        "model_id": model_id,
        "total_vocab_size": len(vocab),
        "hebrew_allowed_token_count": len(sorted_allowed_tokens),
        "hebrew_allowed_tokens": sorted_allowed_tokens
    }
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
    print(f"Saved {len(sorted_allowed_tokens)} Hebrew tokens to {output_filename}\n")

if __name__ == "__main__":
    extract_and_save_tokens("Qwen/Qwen2.5-7B-Instruct", "hebrew_allowed_tokens_qwen.json", "raw_tokens_qwen.json")
    extract_and_save_tokens("mistralai/Mistral-7B-Instruct-v0.3", "hebrew_allowed_tokens_mistral.json", "raw_tokens_mistral.json")