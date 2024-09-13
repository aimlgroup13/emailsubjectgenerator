import os
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Define the custom dataset class
class SubjectDataset(torch.utils.data.Dataset):
    def __init__(self, directory, tokenizer, max_length=256, is_test=False):
        self.directory = directory
        self.tokenizer = tokenizer
        self.files = os.listdir(directory)
        self.max_length = max_length
        self.is_test = is_test
        self.email_prompt = """Below is an instruction that describes a task, paired with an input that provides further context.

### Instruction:
Predict the subject line of the email.

### Input:
{}

### Response:
{}"""

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files[idx])
        with open(file_path, 'r') as f:
            text = f.read()

        # Extract the email body and subject line from the text
        parts = text.split('@subject')
        body = parts[0].strip()
        if self.is_test and '@ann0' in text:
            subject = parts[1].split('@ann0')[0].strip()
        else:
            subject = parts[1].strip()

        # Create the email prompt with the input body
        prompt = self.email_prompt.format(body, "")

        # Tokenize the prompt with padding and truncation
        inputs = self.tokenizer(
            [prompt],
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        # Tokenize the subject line with padding and truncation
        labels = self.tokenizer(
           subject,
           return_tensors='pt',
           max_length=self.max_length,
           padding='max_length',
           truncation=True
        )
        # Flatten tensors and ensure matching lengths
        input_ids = inputs['input_ids'].squeeze()  # Remove batch dimension
        attention_mask = inputs['attention_mask'].squeeze()  # Remove batch dimension
        label_ids = labels['input_ids'].squeeze()  # Remove batch dimension

        # Ensure labels are padded to the same length as input_ids
        if len(label_ids) < len(input_ids):
            padding_length = len(input_ids) - len(label_ids)
            label_ids = torch.cat([label_ids, torch.zeros(padding_length, dtype=torch.long)], dim=0)
        elif len(label_ids) > len(input_ids):
            label_ids = label_ids[:len(input_ids)]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids
        }

# Load the tokenizer and add padding token
model_name = "gpt2"  # Use "gpt2-medium", "gpt2-large", etc., for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load the model
model = GPT2LMHeadModel.from_pretrained(model_name)

# Resize model embeddings after adding a new padding token
model.resize_token_embeddings(len(tokenizer))

# Load the dataset
train_dataset = SubjectDataset(directory='/home/ramch/AI-AUTOMATED-QA/AESLC/enron_subject_line/train', tokenizer=tokenizer, max_length=256)
eval_dataset = SubjectDataset(directory='/home/ramch/AI-AUTOMATED-QA/AESLC/enron_subject_line/test', tokenizer=tokenizer, max_length=256, is_test=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    save_steps=500
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

model.push_to_hub("nagthgr8/subject-gpt2")
tokenizer.push_to_hub("nagthgr8/subject-gpt2")
