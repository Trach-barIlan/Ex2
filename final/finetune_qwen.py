import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 7B Instruct")
    parser.add_argument("--overfit", action="store_true", help="Run in overfit sanity-check mode on 2 examples")
    args = parser.parse_args()

    # 1. Load the model and tokenizer
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # 2. Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Load dataset
    # Yes, it uses full_hebrew_sft_data.jsonl
    dataset = load_dataset("json", data_files="full_hebrew_sft_data.jsonl", split="train")

    # 4. Format the data to use chat template
    def format_chat_template(example):
        # Apply chat template from the model
        example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return example

    dataset = dataset.map(format_chat_template)

    # Overfit setup vs Normal training setup
    if args.overfit:
        print("--- RUNNING IN OVERFIT SANITY CHECK MODE ---")
        dataset = dataset.select(range(2)) # Take only 2 examples
        
        training_args = SFTConfig(
            output_dir="./qwen-hebrew-lora-overfit",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            logging_steps=1, # Log every step to watch the loss drop rapidly
            num_train_epochs=100, # Train for many epochs over the 2 examples
            save_steps=100,
            fp16=False,
            bf16=True,
            optim="adamw_torch",
            report_to="none",
            dataset_text_field="text",
            max_length=512
        )
    else:
        print("--- RUNNING FULL FINE-TUNING ---")
        training_args = SFTConfig(
            output_dir="./qwen-hebrew-lora",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=10,
            num_train_epochs=1, # Run for 1 epoch over the full dataset
            save_steps=50,
            fp16=False,
            bf16=True,
            optim="adamw_torch",
            report_to="none",
            dataset_text_field="text",
            max_length=512
        )

    # 5. Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
    )

    # 6. Start training
    print("Starting training...")
    trainer.train()

    # 7. Save the final model
    output_dir = "./qwen-hebrew-lora-overfit-final" if args.overfit else "./qwen-hebrew-lora-final"
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
