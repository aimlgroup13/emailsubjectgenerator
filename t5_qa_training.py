import json
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from torch.optim import AdamW


def is_bfloat16_supported():
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8  # A100 GPUs and above


class QADataset(Dataset):
    def __init__(self, json_path, tokenizer, split='train', max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(json_path, split)

    def load_data(self, json_path, split):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        
        if 'answer1' in item and 'answer2' in item:
            # For DEV and TEST, we have two answers
            answers = [item['answer1'], item['answer2']]
            # Use the first answer for training purposes
            answer = answers[0]
        elif 'answer' in item:
            # For TRAIN, we have one answer
            answer = item['answer']
        else:
            # Handle cases where data is not available as expected
            raise ValueError("Unexpected data format")
        # Print question and answer
        print(f"Question: {question}")
        print(f"Answer: {answer}")

        # Tokenize the question and answer
        inputs = self.tokenizer(
            question,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        labels = self.tokenizer(
            answer,
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



tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Create Dataset instances for each split
train_dataset = QADataset('/home/ramch/AI-AUTOMATED-QA/AIMLQA/aiml-qa-dataset.json', tokenizer, split='TRAIN')
dev_dataset = QADataset('/home/ramch/AI-AUTOMATED-QA/AIMLQA/aiml-qa-dataset.json', tokenizer, split='DEV')
test_dataset = QADataset('/home/ramch/AI-AUTOMATED-QA/AIMLQA/aiml-qa-dataset.json', tokenizer, split='TEST')

# Create DataLoaders for each split
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
dev_dataloader = DataLoader(dev_dataset, batch_size=4, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
from transformers import AdamW

# Define the model
model = T5ForConditionalGeneration.from_pretrained('t5-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-4)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=inputs, attention_mask=batch['attention_mask'].to(device), labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if step % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item()}")

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}] - Average Loss: {avg_loss}")

    # Evaluate
    model.eval()
    total_eval_loss = 0.0
    with torch.no_grad():
        for batch in dev_dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=inputs, attention_mask=batch['attention_mask'].to(device), labels=labels)
            loss = outputs.loss
            total_eval_loss += loss.item()
    avg_eval_loss = total_eval_loss / len(dev_dataloader)
    print(f"Epoch [{epoch+1}] - Dev Loss: {avg_eval_loss}")

model_name = "qa-t5-base"
save_directory = f"results/{model_name}"
# Create the directory if it does not exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Optionally save the model
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
# Push model and tokenizer to Hugging Face Model Hub
model.push_to_hub(f"nagthgr8/{model_name}", commit_message="Initial model push")
tokenizer.push_to_hub(f"nagthgr8/{model_name}", commit_message="Initial tokenizer push")
