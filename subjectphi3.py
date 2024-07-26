import os
import gc
import sys
import logging

import datasets
from datasets import Dataset as HF_Dataset
from peft import LoraConfig
import torch
import transformers
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import json
from torch.utils.data import Dataset, random_split

logging.basicConfig(level=logging.DEBUG)


"""
A simple example on using SFTTrainer and Accelerate to finetune Phi-3 models.
This example utilizes DeepSpeed ZeRO3 offload to reduce memory usage.
Please follow these steps to run the script:
1. Install dependencies: 
    pip install accelerate bitsandbytes peft transformers trl datasets deepspeed
2. Setup accelerate and deepspeed config based on the machine used:
    accelerate config
3. check accelerate config:
    accelerate env
4. Run the code:
    accelerate launch sample_finetune.py
"""

logger = logging.getLogger(__name__)

###################
# Hyper-parameters
###################

from transformers import TrainingArguments
num_processes = 1
training_config = {
    "fp16": False,
    "do_eval": False,
    "learning_rate": 1e-05,
    "log_level": "info",
    "logging_steps": 500,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "max_steps": 722,
    "output_dir": "./checkpoint_dir",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 4,
    "per_device_train_batch_size": 4,
    "remove_unused_columns": True,
    "save_steps": 1000,
    "save_total_limit": 3,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs":{"use_reentrant": False},
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.2,
    "evaluation_strategy": "steps",
    "eval_steps": 1000
    }

peft_config = {
    "lora_alpha": 16,  # Adjust alpha value
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
    "modules_to_save": ["encoder.layer.0", "encoder.layer.1"]
}
train_conf = TrainingArguments(**training_config)
peft_conf = LoraConfig(**peft_config)

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process a small summary
logger.warning(
    f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
    + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
)
logger.info(f"Training/evaluation parameters {train_conf}")
logger.info(f"PEFT parameters {peft_conf}")

################
# Model Loading
################
checkpoint_path = "microsoft/Phi-3-mini-4k-instruct"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
    torch_dtype=torch.bfloat16,
    device_map=None
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
# Move the model to GPU
if torch.cuda.is_available():
    model = model.to('cuda')
logger.info(f"device: {next(model.parameters()).device}")  # Should print 'cuda:0' or similar
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

##################
# Data Processing
##################
def apply_chat_template(example, tokenizer):
    # Example processing function that applies a chat template and tokenizes
    input_text = example['input']
    output_text = example['output']
    # Apply any template or processing required for your use case
    formatted_input = f"Instruction: {input_text}\nResponse:"
    # Tokenize the formatted text
    tokenized_input = tokenizer(formatted_input, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    tokenized_output = tokenizer(output_text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    return {
        'input_ids': tokenized_input['input_ids'].squeeze(),
        'attention_mask': tokenized_input['attention_mask'].squeeze(),
        'labels': tokenized_output['input_ids'].squeeze(),  # Include labels
        'text': input_text
    }

def parse_subject_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content based on the markers
    sections = content.split('@subject')
    if len(sections) < 2:
        return None  # Invalid format

    input_text = sections[0].strip()
    rest = sections[1]
    subject = rest.split('@ann')[0].strip()
    # Extract annotations
    annotations = {}
    for i, line in enumerate(rest.split('@ann')[1:]):
        key, value = line.split('\n', 1)
        annotations[key] = value.strip()

    return {
        'input': input_text,
        'output': subject,
        'annotations': annotations
    }

def load_data_from_folder(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.subject'):
            file_path = os.path.join(folder_path, filename)
            parsed_data = parse_subject_file(file_path)
            if parsed_data:
                data.append(parsed_data)
    return data

train_folder_path = '/home/ramch/AI-AUTOMATED-QA/AESLC/enron_subject_line/train'
raw_data = load_data_from_folder(train_folder_path)
print(f"Total Train Records: {len(raw_data)} records")


# Assume raw_data is a list of dictionaries with 'input' and 'output' fields
original_dataset = HF_Dataset.from_dict({
    'input': [entry['input'] for entry in raw_data],
    'output': [entry['output'] for entry in raw_data]
})
# Apply chat template to the original dataset
processed_dataset = original_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=1,
    remove_columns=["input", "output"],  # Use the original columns
    desc="Applying chat template"
)

# Define sizes for training and testing
train_size = int(0.8 * len(processed_dataset))
test_size = len(processed_dataset) - train_size

# Split the dataset
train_dataset, test_dataset = random_split(processed_dataset, [train_size, test_size])

# Function to print a few records
def print_samples(dataset, num_samples=5):
    for entry in dataset:
        print(entry['text'])
        tokenized_data = tokenizer(entry['text'], padding=True, truncation=True, return_tensors="pt")
        print("Input IDs:", tokenized_data["input_ids"])
        break

# Print samples from train_dataset
print("Sample records from train_dataset:")
print_samples(train_dataset)

# Print samples from test_dataset
print("Sample records from test_dataset:")
print_samples(test_dataset)

column_names = list(processed_dataset.features)
print("Dataset features:", column_names)

print("TrainingArguments Configuration:", train_conf)
with open('deepspeed_config.json') as f:
    deepspeed_config = json.load(f)
print("DeepSpeed Configuration:", deepspeed_config)

###########
# Training
###########
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    peft_config=peft_conf,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    max_seq_length=2048,
    dataset_text_field="text",
    tokenizer=tokenizer,
    packing=True
)
train_result = trainer.train()
print('Training completed!')
# Clear unnecessary variables
del train_result
#del metrics

# Run garbage collection
gc.collect()

print('Evaluating the model')
metrics = trainer.evaluate()
metrics["eval-samples"] = len(test_dataset)
print('Saving the metrics')
trainer.save_metrics("eval", metrics)

del metrics
gc.collect()

# Save model
#trainer.save_model(train_conf.output_dir)
print('Saving the model..')
model.push_to_hub("nagthgr8/subject-phi3")
print('Done')
del trainer
gc.collect()
print('The End & the beginning of new Era!')
