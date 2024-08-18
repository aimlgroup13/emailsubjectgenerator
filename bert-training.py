from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import json
from sklearn.model_selection import train_test_split

# Load the tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
# Define the email prompt template and EOS token
email_prompt = """Below is an instruction that describes a task, paired with an input that provides further context.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # EOS token is necessary

# Load your dataset from the JSON file
json_file_path = '/home/ramch/AI-AUTOMATED-QA/dataset.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Convert the list of dictionaries to a Hugging Face Dataset
dataset = Dataset.from_list(data)

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        # Construct the text without EOS token
        text = email_prompt.format(instruction, input_text, output_text)
        texts.append(text)
    return {"text": texts}

def tokenize_and_encode(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

def format_and_tokenize(examples):
    formatted_examples = formatting_prompts_func(examples)
    tokenized_inputs = tokenize_and_encode(formatted_examples["text"])
    return tokenized_inputs

def format_example(examples):
    text = email_prompt.format(examples["instruction"], examples["input"], examples["output"])
    tokenized_inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    tokenized_labels = tokenizer(examples["output"], padding=True, truncation=True, return_tensors="pt")
    return {
        "input_ids": tokenized_inputs["input_ids"].flatten().tolist(),
        "attention_mask": tokenized_inputs["attention_mask"].flatten().tolist(),
        "labels": tokenized_labels["input_ids"].flatten().tolist(),
    }

# Tokenize the datasets
def tokenize_function(examples):
    inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    labels = []
    for text in examples['text']:
        # Extract the response part from the text
        response_start = text.find('### Response:')
        response_text = text[response_start:]
        # Tokenize the response text as labels
        target = tokenizer(response_text, truncation=True, padding='max_length', max_length=512)
        labels.append(target['input_ids'])
    # Add the labels to the inputs dictionary
    inputs['labels'] = labels
    return inputs

print("Getting formatted dataset")
formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

print("Getting tokenized dataset")
tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

# Split the dataset into training and evaluation sets
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

# Check train dataset list lengths
print("Train Dataset:")
for i, batch in enumerate(train_dataset):
    print(f"Batch {i+1}:")
    print("Input IDs length:", len(batch['input_ids']))
    print("Attention Mask length:", len(batch['attention_mask']))
    print("Labels length:", len(batch['labels']))
    break  # Remove this line to print lengths for all batches

# Check test dataset list lengths
print("Test Dataset:")
for i, batch in enumerate(test_dataset):
    print(f"Batch {i+1}:")
    print("Input IDs length:", len(batch['input_ids']))
    print("Attention Mask length:", len(batch['attention_mask']))
    break  # Remove this line to print lengths for all batches


from transformers import EncoderDecoderModel, BertTokenizer, TrainingArguments, Trainer

# Configure the model
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.max_length = 256
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True
model.config.length_penalty = 2.0
model.config.num_beams = 4

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Reduced batch size
    per_device_eval_batch_size=2,   # Reduced batch size
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    predict_with_generate=True,
)

# Initialize the Trainer
# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)
print("Training the model")
# Train the model
trainer.train()

print("Push to hub")
# Push the model to the Hugging Face hub
trainer.push_to_hub("bert-aeslc-subject-line")
