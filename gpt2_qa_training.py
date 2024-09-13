from torch.utils.data import Dataset
import torch
import os
import json
from transformers import GPT2Tokenizer
from torch.optim import AdamW

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
            answers = [item['answer1'], item['answer2']]
            answer = answers[0]
        elif 'answer' in item:
            answer = item['answer']
        else:
            raise ValueError("Unexpected data format")

        # Concatenate question and answer with a delimiter for GPT-2
        text = f"Question: {question} Answer: {answer}"

        # Tokenize the concatenated text
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        input_ids = encoding['input_ids'].squeeze()
        labels = input_ids.clone()  # For GPT-2, labels are the same as inputs

        return {
            'input_ids': input_ids,
            'labels': labels
        }


from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Create Dataset instances
train_dataset = QADataset('/home/ramch/AI-AUTOMATED-QA/AIMLQA/aiml-qa-dataset.json', tokenizer, split='TRAIN')
dev_dataset = QADataset('/home/ramch/AI-AUTOMATED-QA/AIMLQA/aiml-qa-dataset.json', tokenizer, split='DEV')
test_dataset = QADataset('/home/ramch/AI-AUTOMATED-QA/AIMLQA/aiml-qa-dataset.json', tokenizer, split='TEST')

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-4)
# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    output_dir='./results',
    evaluation_strategy='epoch',
    logging_dir='./logs',
    num_train_epochs=3,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
)

# Train model
trainer.train()

model_name = 'gpt2-qa'
save_directory = f"results/{model_name}"
# Create the directory if it does not exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

model.push_to_hub(f"nagthgr8/{model_name}", commit_message="Initial model push")
tokenizer.push_to_hub(f"nagthgr8/{model_name}", commit_message="Initial tokenizer push")
