import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

def query_model(model_id, prompts, hf_token=None):
    print(f"\n========================================")
    print(f"Loading {model_id}...")
    print(f"========================================")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    # Load model (using float16 and device_map="auto" to handle VRAM automatically)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        dtype=torch.float16,
        token=hf_token
    )
    
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n>>> Running Test Prompt {idx}")
        # Define the conversation using the chat format
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply the model's specific chat template
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize the formatted prompt
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate response
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
        
        # Slice the output to get only the newly generated tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode the response
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"\n--- {model_id} Response ---\n{response.strip()}\n----------------")
    
    # Cleanup to free up VRAM for the next model
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    my_hf_token = os.getenv("HF_TOKEN")
    
    # A diverse set of prompts to test reasoning, instruction following, and constraints
    test_prompts = [
        # 1. Logic and Reasoning (Trick Question)
        "A farmer has 17 sheep. All but 9 run away. How many sheep are left? Explain your reasoning briefly.",
        
        # 2. Strict Instruction Following (Formatting constraint)
        # "List three benefits of regular exercise. Format your response strictly as a valid JSON array of strings. Do not include any explanations, markdown formatting like ```json, or other text.",
        
        # 3. Coding with Constraints
        # "Write a Python function to check if a string is a palindrome. You must NOT use string slicing (e.g. s[::-1]) or the built-in reversed() function."
    ]
    
    models_to_test = [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ]
    
    for model_id in models_to_test:
        query_model(model_id, test_prompts, hf_token=my_hf_token)