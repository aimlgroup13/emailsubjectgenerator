import json
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from transformers import BertTokenizerFast, BertForQuestionAnswering, AdamW, TrainingArguments, Trainer
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
        context = item['context']
        question = item['question']

        if 'answer1' in item and 'answer2' in item:
            answers = [item['answer1'], item['answer2']]
            answer = answers[0]
        elif 'answer' in item:
            answer = item['answer']
        else:
            raise ValueError("Unexpected data format")

        # Encode context and question with return_offsets_mapping to map tokens to original text
        inputs = self.tokenizer(
            question,
            context,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True   # Get token-to-text alignment
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        offsets = inputs['offset_mapping'].squeeze()

        # Find the start and end char positions of the answer in the context
        answer_start_char = context.find(answer)
        answer_end_char = answer_start_char + len(answer)

        # Initialize start and end positions for tokens
        start_position = 0
        end_position = 0

        # Find the token positions that match the answer span
        for idx, (start_offset, end_offset) in enumerate(offsets):
            if start_offset <= answer_start_char and end_offset >= answer_start_char:
                start_position = idx
            if start_offset <= answer_end_char and end_offset >= answer_end_char:
                end_position = idx
                break

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': torch.tensor(start_position, dtype=torch.long),
            'end_positions': torch.tensor(end_position, dtype=torch.long)
        }

# Use BERT-base for Question Answering
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Create Dataset instances for each split
train_dataset = QADataset('/home/ramch/AI-AUTOMATED-QA/AIMLQA/qa-dataset.json', tokenizer, split='TRAIN')
dev_dataset = QADataset('/home/ramch/AI-AUTOMATED-QA/AIMLQA/qa-dataset.json', tokenizer, split='DEV')
test_dataset = QADataset('/home/ramch/AI-AUTOMATED-QA/AIMLQA/qa-dataset.json', tokenizer, split='TEST')

# Create DataLoaders for each split
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
dev_dataloader = DataLoader(dev_dataset, batch_size=4, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
from transformers import AdamW

# Define the model
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
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=inputs,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Log every 10 steps
        if step % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}] - Average Training Loss: {avg_loss:.4f}")

    # Evaluation loop
    model.eval()
    total_eval_loss = 0.0
    with torch.no_grad():
        for batch in dev_dataloader:
            inputs = batch['input_ids'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )

            loss = outputs.loss
            total_eval_loss += loss.item()

    avg_eval_loss = total_eval_loss / len(dev_dataloader)
    print(f"Epoch [{epoch+1}] - Dev Loss: {avg_eval_loss:.4f}")

model_name = "qa-bert-base"
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
