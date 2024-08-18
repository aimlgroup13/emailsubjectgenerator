import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from unsloth import FastLanguageModel
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token

email_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        # Construct text with EOS token
        text = email_prompt.format(instruction, input_text, output_text) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }

# Path to your dataset.json file
json_file_path = '/home/ramch/AI-AUTOMATED-QA/dataset.json'

# Load JSON data into a Python list
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Convert the list of dictionaries to a Hugging Face Dataset
dataset = Dataset.from_list(data)
# Apply the formatting function to the dataset
dataset = dataset.map(formatting_prompts_func, batched=True)

from sklearn.model_selection import train_test_split
dataset_dict = dataset.train_test_split(test_size=0.005)

print(dataset)
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

from transformers import TrainingArguments
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    dataset_text_field="text",
    args=TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        logging_strategy='steps',
    ),
)
# Ensure the model is moved to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Start training
trainer.train()
print("Training Completed")
model_name = "nagthgr8/subject-gpt2"
print("Saving the model")
model.push_to_hub(model_name)
tokenizer.push_to_hub(model_name)
print("Done")
