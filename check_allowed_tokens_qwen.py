import json


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
INPUT_FILE = "hebrew_allowed_tokens_qwen.json"
RAW_TOKENS_FILE = "raw_tokens_qwen.json"
OUTPUT_FILE = "hebrew_allowed_tokens_qwen_readable.json"


def main():
    print(f"Reading allowed token ids from {INPUT_FILE}...", flush=True)
    with open(INPUT_FILE, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    print(f"Reading raw token strings from {RAW_TOKENS_FILE}...", flush=True)
    with open(RAW_TOKENS_FILE, "r", encoding="utf-8") as file_handle:
        raw_token_data = json.load(file_handle)

    raw_tokens = raw_token_data.get("raw_tokens", {})
    id_to_token = {token_id: token_str for token_str, token_id in raw_tokens.items()}

    allowed_token_ids = data.get("allowed_token_ids", [])
    readable_tokens = []

    print(f"Resolving {len(allowed_token_ids)} allowed tokens...", flush=True)
    for token_id in allowed_token_ids:
        token_repr = id_to_token.get(token_id, "<unknown>")
        readable_tokens.append(
            {
                "id": token_id,
                "token": token_repr,
            }
        )

    output_data = {
        "model_id": data.get("model_id", MODEL_ID),
        "allowed_tokens": readable_tokens,
    }

    print(f"Writing readable token review file to {OUTPUT_FILE}...", flush=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file_handle:
        json.dump(output_data, file_handle, ensure_ascii=False, indent=4)

    print(f"Done. Wrote {len(readable_tokens)} entries to {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()