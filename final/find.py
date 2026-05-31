import csv
import json
import os
from transformers import AutoConfig

def extract_architecture_data():
    models = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-7B-Instruct",
        "allenai/OLMo-2-1124-7B-Instruct",
        "ibm-granite/granite-3.3-8b-instruct",
        "deepseek-ai/DeepSeek-V3",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "microsoft/Phi-4-mini-instruct",
        "tiiuae/Falcon3-7B-Instruct",
        "dicta-il/dictalm2.0-instruct"
    ]

    csv_file = "architecture.csv"
    md_file = "architecture.md"
    raw_config_dir = "raw_configs"
    
    os.makedirs(raw_config_dir, exist_ok=True)

    fields = [
        "model_id", "hidden_size", "num_layers", "num_attention_heads", 
        "num_kv_heads", "mlp_size", "activation", "norm_type", 
        "position_encoding", "context_length", "vocab_size", "moe_details"
    ]

    all_rows = []

    for model_id in models:
        print(f"\nFetching config for: {model_id}...")
        row = {field: "NA" for field in fields}
        row["model_id"] = model_id
        
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            
            # Save raw JSON
            safe_filename = model_id.replace("/", "_") + ".json"
            json_filepath = os.path.join(raw_config_dir, safe_filename)
            with open(json_filepath, "w", encoding="utf-8") as json_file:
                json.dump(config.to_dict(), json_file, indent=4)
            
            # Extract basic fields
            row["hidden_size"] = getattr(config, "hidden_size", "NA")
            row["num_layers"] = getattr(config, "num_hidden_layers", getattr(config, "num_layers", "NA"))
            row["num_attention_heads"] = getattr(config, "num_attention_heads", "NA")
            row["num_kv_heads"] = getattr(config, "num_key_value_heads", getattr(config, "multi_query_group_num", "NA"))
            row["mlp_size"] = getattr(config, "intermediate_size", getattr(config, "moe_intermediate_size", "NA"))
            row["vocab_size"] = getattr(config, "vocab_size", "NA")
            row["context_length"] = getattr(config, "max_position_embeddings", "NA")
            row["activation"] = getattr(config, "hidden_act", getattr(config, "hidden_activation", "NA"))
            
            # Position Encoding (Improved Logic)
            config_dict = config.to_dict()
            if "rope_theta" in config_dict or "rope_scaling" in config_dict or getattr(config, "position_embedding_type", "") == "rope":
                row["position_encoding"] = "RoPE"
            # Fallback based on architecture type
            elif "llama" in model_id.lower() or "mistral" in model_id.lower() or "qwen" in model_id.lower() or "deepseek" in model_id.lower():
                row["position_encoding"] = "RoPE"

            # Normalization Type
            if hasattr(config, "rms_norm_eps"):
                row["norm_type"] = "RMSNorm"
            elif hasattr(config, "layer_norm_eps"):
                row["norm_type"] = "LayerNorm"
                
            # MoE Details
            moe_details = []
            if hasattr(config, "n_routed_experts"):
                moe_details.append(f"routed:{config.n_routed_experts}")
            if hasattr(config, "n_shared_experts"):
                moe_details.append(f"shared:{config.n_shared_experts}")
            if moe_details:
                row["moe_details"] = ", ".join(moe_details)

        except Exception as e:
            print(f"  -> Error processing {model_id}: {e}")

        all_rows.append(row)

    # Write to CSV
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    # Write to Markdown
    with open(md_file, mode="w", encoding="utf-8") as file:
        file.write("# Model Architectural Choices\n\n")
        file.write("Extracted automatically from Hugging Face `config.json` files.\n\n")
        header_line = "| " + " | ".join(fields) + " |"
        separator_line = "|" + "|".join(["---"] * len(fields)) + "|"
        file.write(header_line + "\n")
        file.write(separator_line + "\n")
        
        for row in all_rows:
            row_values = [str(row[field]) for field in fields]
            file.write("| " + " | ".join(row_values) + " |\n")

    print(f"\nExtraction complete. Position encodings updated.")

if __name__ == "__main__":
    extract_architecture_data()