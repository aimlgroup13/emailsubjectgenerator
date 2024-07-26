import os

def parse_subject_file(file_path, is_test=False):
    with open(file_path, 'r') as file:
        content = file.read()
    if '@subject' not in content:
        raise ValueError("Invalid file format. '@subject' not found.")
    email_body, remainder = content.split('@subject', 1)
    if is_test:
        # Extract content until the first annotation (if any)
        email_subject = remainder.split('@ann', 1)[0].strip()
    else:
        email_subject = remainder.strip()
    return email_body.strip(), email_subject.strip()

def convert_to_conversational_format(email_body, email_subject):
    return [
        {"role": "user", "content": email_body},
        {"role": "assistant", "content": email_subject}
    ]

def process_directory(directory_path):
    all_conversations = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.subject'):
            file_path = os.path.join(directory_path, filename)
            email_body, email_subject = parse_subject_file(file_path)
            conversation = convert_to_conversational_format(email_body, email_subject)
            all_conversations.append(conversation)
    return all_conversations

train_directory_path = '/home/ramch/AI-AUTOMATED-QA/AESLC/enron_subject_line/train'
train_conversations = process_directory(train_directory_path)

test_directory_path = '/home/ramch/AI-AUTOMATED-QA/AESLC/enron_subject_line/test'
test_conversations = process_directory(test_directory_path)

# Continue with your model training setup
from transformers import TrainingArguments
from trl import SFTTrainer
import torch
import tensorflow as tf
from tensorflow import keras

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


# Define TrainingArguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    eval_strategy="steps",
    warmup_steps=0,
    num_train_epochs=1,  # This will be overridden by max_steps
    max_steps=1,  # Adjusted based on the calculated steps
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
    save_total_limit=3,
    # DeepSpeed-related arguments removed
)
# Ensure the model is moved to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from datasets import Dataset

# Tokenize conversations
def tokenize_function(examples):
    return tokenizer(examples['content'], padding='max_length', truncation=True, max_length=max_seq_length)


# Prepare training and evaluation datasets
flattened_train_conversations = [message for conversation in train_conversations for message in conversation]
flattened_test_conversations = [message for conversation in test_conversations for message in conversation]

train_dataset = Dataset.from_dict({'content': [message['content'] for message in flattened_train_conversations], 'role': [message['role'] for message in flattened_train_conversations]})
eval_dataset = Dataset.from_dict({'content': [message['content'] for message in flattened_test_conversations], 'role': [message['role'] for message in flattened_test_conversations]})

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Initialize SFTTrainer with TrainingArguments
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=training_args,  # Pass TrainingArguments instance here
)

# Start training
trainer.train()
print("Training Completed")
model_name = "nagthgr8/subject-mistrel"
print("Saving the model")
model.push_to_hub(model_name)
tokenizer.push_to_hub(model_name)
print("Done")
