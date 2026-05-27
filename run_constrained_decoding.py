import json
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

# 1. Implement Constrained Decoding via LogitsProcessor
class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids, vocab_size):
        # Create a boolean mask where True means the token is ALLOWED
        self.allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)
        self.allowed_mask[allowed_token_ids] = True

    def __call__(self, input_ids, scores):
        # Move mask to the same device as scores
        self.allowed_mask = self.allowed_mask.to(scores.device)
        # Set the logits of all DISALLOWED tokens to negative infinity
        scores[:, ~self.allowed_mask] = float('-inf')
        return scores

def main():
    # 2. Define the JSON files containing the allowed tokens for each model
    token_files = [
        '/home/dsi/halamim1/Ex2/hebrew_allowed_tokens_qwen.json',
        '/home/dsi/halamim1/Ex2/hebrew_allowed_tokens_mistral.json'
    ]
    
    # Clear the output file before appending
    output_file = '/home/dsi/halamim1/Ex2/decoding_outputs.jsonl'
    open(output_file, 'w').close()

    # 3. Define 10 English Queries
    queries = [
        "Explain why leaves are green.",
        "What is the capital of France?",
        "Write a short poem about a cat.",
        "How does a microwave oven work?",
        "Who wrote the play Romeo and Juliet?",
        "Give me a simple recipe for chocolate chip cookies.",
        "What are the health benefits of regular exercise?",
        "Describe the plot of the movie The Matrix.",
        "Why is the sky blue during the day?",
        "Translate the following sentence to French: Hello, how are you?"
    ]

    # 4. Loop over each model's token file
    for token_file in token_files:
        try:
            with open(token_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Warning: {token_file} not found. Skipping.")
            continue
            
        allowed_token_ids = data['allowed_token_ids']
        model_id = data['model_id']

        # Load Model and Tokenizer
        print(f"\nLoading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        
        # Ensure critical tokens (like EOS) are allowed so the model can stop
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in allowed_token_ids:
            allowed_token_ids.append(tokenizer.eos_token_id)

        # Create the LogitsProcessorList
        logits_processor = LogitsProcessorList([
            AllowedTokensLogitsProcessor(allowed_token_ids, model.config.vocab_size)
        ])

        results = []

        # 5. Run Generation
        for prompt in queries:
            print(f"Processing: {prompt}")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Unconstrained Decoding
            unconstrained_outputs = model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=False
            )
            unconstrained_text = tokenizer.decode(unconstrained_outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Constrained Decoding
            constrained_outputs = model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=False, 
                logits_processor=logits_processor
            )
            constrained_text = tokenizer.decode(constrained_outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            results.append({
                "prompt": prompt,
                "model": model_id,
                "unconstrained_output": unconstrained_text.strip(),
                "constrained_output": constrained_text.strip()
            })

        # 6. Append to JSONL
        with open(output_file, 'a', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')
                
        # 7. Free memory before loading the next model
        print(f"Cleaning up memory for {model_id}...")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()