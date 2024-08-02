import os
import pandas as pd
import json
from datasets import Dataset

# Continue with your model training setup
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit", # New Google 6 trillion tokens model 2.5x faster!
    "unsloth/gemma-2b-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-v0.3", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# Ensure the model is moved to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer_padding_side = "left"
import json
from transformers import pipeline
from evaluate import load
from rouge_score import rouge_scorer
# Load the ROUGE metric
rouge = load("rouge")

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
with open('/home/ramch/AI-AUTOMATED-QA/testdataset.json', 'r') as f:
    testdataset = json.load(f)

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
email_prompt = "Generate a concise and relevant subject line for the following email body:\n\n{}\n\nSubject Line:"
all_scores = []
print("Calculating the scores..")
# Use index-based access for Dataset objects
for i in range(len(testdataset[:100])):
    example = testdataset[i]  # Access element like a list
    instruction = example['instruction']
    input_text = example['input']
    references = example['output']  # List of references
    # Tokenize input text and generate prediction
    inputs = tokenizer(
    [
       email_prompt.format(input_text)
    ], return_tensors="pt").to("cuda")
    #outputs = model.generate(**inputs, max_new_tokens=64, use_cache = True)
    outputs = model.generate(
    **inputs,
    max_new_tokens=64,  # Limit length to encourage short outputs
    num_return_sequences=1,  # Generate one sequence
    no_repeat_ngram_size=2,  # Avoid repeating phrases
    early_stopping=True  # Stop when the model thinks it’s done
    )

    prediction = tokenizer.batch_decode(outputs)
    print("Input Text:")
    print(input_text)
    print("\nGenerated Output Tokens:")
    print(outputs[0])  # This prints the raw token IDs, which might not be very readable

    print("\nDecoded Prediction:")
    print(prediction)
    # Calculate ROUGE scores against each reference and store the best
    best_scores = {'rouge1': {'fmeasure': 0, 'precision': 0, 'recall': 0},
                   'rouge2': {'fmeasure': 0, 'precision': 0, 'recall': 0},
                   'rougeL': {'fmeasure': 0, 'precision': 0, 'recall': 0}}
    print(references)

    for ref in references:
        print("ref", ref)
        scores = scorer.score(prediction[0], ref)
        print(scores)
        for metric, score in scores.items():
            if score.fmeasure > best_scores[metric]['fmeasure']:
                best_scores[metric] = {'fmeasure': score.fmeasure,
                                       'precision': score.precision,
                                       'recall': score.recall}
    print(best_scores)
    all_scores.append(best_scores)

# Aggregate scores (e.g., calculate average)
avg_scores = {
    'rouge1': {'fmeasure': 0, 'precision': 0, 'recall': 0},
    'rouge2': {'fmeasure': 0, 'precision': 0, 'recall': 0},
    'rougeL': {'fmeasure': 0, 'precision': 0, 'recall': 0}
}
print("Calculating avg scores..")
for scores in all_scores:
    for metric in avg_scores:
        for key in ['fmeasure', 'precision', 'recall']:
            if key in scores[metric]:
                avg_scores[metric][key] += scores[metric][key]

for metric in avg_scores:
    for key in ['fmeasure', 'precision', 'recall']:
        avg_scores[metric][key] /= len(all_scores)

# Print the average ROUGE scores
print(avg_scores)

with open('mistral_base_avg_scores.json', 'w') as f:
    json.dump(avg_scores, f, indent=4)
