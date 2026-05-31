import re
from transformers import AutoTokenizer

TEXT = """Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense. 

Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere. 

The Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn't think they could bear it if anyone found out about the Potters. Mrs. Potter was Mrs. Dursley's sister, but they hadn't met for several years; in fact, Mrs. Dursley pretended she didn't have a sister, because her sister and her good-for-nothing husband were as unDursleyish as it was possible to be. The Dursleys shuddered to think what the neighbors would say if the Potters arrived in the street. The Dursleys knew that the Potters had a small son, too, but they had never even seen him. This boy was another good reason for keeping the Potters away; they didn't want Dudley mixing with a child like that. 

When Mr. and Mrs. Dursley woke up on the dull, gray Tuesday our story starts, there was nothing about the cloudy sky outside to suggest that strange and mysterious things would soon be happening all over the country. Mr. Dursley hummed as he picked out his most boring tie for work, and Mrs. Dursley gossiped away happily as she wrestled a screaming Dudley into his high chair. 

None of them noticed a large, tawny owl flutter past the window. 

At half past eight, Mr. Dursley picked up his briefcase, pecked Mrs. Dursley on the cheek, and tried to kiss Dudley good-bye but missed, because Dudley was now having a tantrum and throwing his cereal at the walls. "Little tyke," chortled Mr. Dursley as he left the house. He got into his car and backed out of number four's drive. 

It was on the corner of the street that he noticed the first sign of something peculiar -- a cat reading a map. For a second, Mr. Dursley didn't realize what he had seen -- then he jerked his head around to look again. There was a tabby cat standing on the corner of Privet Drive, but there wasn't a map in sight. What could he have been thinking of? It must have been a trick of the light. Mr. Dursley blinked and stared at the cat. It stared back. As Mr. Dursley drove around the corner and up the road, he watched the cat in his mirror. It was now reading the sign that said Privet Drive -- no, looking at the sign; cats couldn't read maps or signs. Mr. Dursley gave himself a little shake and put the cat out of his mind. As he drove toward town he thought of nothing except a large order of drills he was hoping to get that day."""

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

def normalize_tokens(tokenizer, text):
    """Encodes text and returns a list of string tokens with a normalized space character '_' """
    tokens = tokenizer.tokenize(text)
    # Replace BPE 'Ġ' and SentencePiece ' ' with a standard underscore for comparison
    normalized = [t.replace('Ġ', '_').replace(' ', '_') for t in tokens]
    return normalized

# 1. Gather tokenizations for all models
tokenizations = {}
for model in models:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        tokenizations[model] = normalize_tokens(tokenizer, TEXT)
    except Exception as e:
        print(f"Failed to load {model}: {e}")

# We will use Llama, Mistral, and Qwen as our 3 baseline models
model_A = "meta-llama/Llama-3.1-8B-Instruct"
model_B = "mistralai/Mistral-7B-Instruct-v0.3"
model_C = "Qwen/Qwen2.5-7B-Instruct"

tokens_A = tokenizations.get(model_A, [])
tokens_B = tokenizations.get(model_B, [])
tokens_C = tokenizations.get(model_C, [])

# 2. To make the report readable, let's extract just a specific tricky sentence to display
tricky_sentence = "Mrs. Dursley pretended she didn't have a sister, because her sister and her good-for-nothing husband were as unDursleyish as it was possible to be."

print("\n--- Visual Tokenization of the Tricky Sentence ---")
try:
    tok_A = AutoTokenizer.from_pretrained(model_A)
    print(f"\n{model_A}:")
    print(normalize_tokens(tok_A, tricky_sentence))
    
    tok_B = AutoTokenizer.from_pretrained(model_B)
    print(f"\n{model_B}:")
    print(normalize_tokens(tok_B, tricky_sentence))
    
    tok_C = AutoTokenizer.from_pretrained(model_C)
    print(f"\n{model_C}:")
    print(normalize_tokens(tok_C, tricky_sentence))
except Exception as e:
    pass

# 3. Check agreement across the remaining 7 models for the FULL text
print("\n--- Agreement Analysis (Full Text) ---")
remaining_models = [m for m in models if m not in [model_A, model_B, model_C]]

agreements = {model_A: 0, model_B: 0, model_C: 0}
no_agreement = 0

for rem_model in remaining_models:
    rem_tokens = tokenizations.get(rem_model, [])
    
    matched = False
    if rem_tokens == tokens_A:
        agreements[model_A] += 1
        print(f"[Match] {rem_model} perfectly agrees with {model_A}")
        matched = True
    elif rem_tokens == tokens_B:
        agreements[model_B] += 1
        print(f"[Match] {rem_model} perfectly agrees with {model_B}")
        matched = True
    elif rem_tokens == tokens_C:
        agreements[model_C] += 1
        print(f"[Match] {rem_model} perfectly agrees with {model_C}")
        matched = True
        
    if not matched:
        print(f"[Unique] {rem_model} tokenized the text differently from A, B, and C.")
        no_agreement += 1

print("\n--- Final Agreement Tally ---")
print(f"Models agreeing with {model_A}: {agreements[model_A]}")
print(f"Models agreeing with {model_B}: {agreements[model_B]}")
print(f"Models agreeing with {model_C}: {agreements[model_C]}")
print(f"Models with completely unique tokenizations: {no_agreement}")