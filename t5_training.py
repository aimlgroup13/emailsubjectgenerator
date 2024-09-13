import os
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from torch.optim import AdamW


def is_bfloat16_supported():
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8  # A100 GPUs and above

# Define the model
model = T5ForConditionalGeneration.from_pretrained('t5-small')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=2e-4)  # Adjust learning rate as needed

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
        parts = text.split('@subject')
        # Extract the email body and subject line from the text
        body = parts[0].strip()
        if self.is_test and '@ann0' in text:
            subject = parts[1].split('@ann0')[0].strip()
        else:
            subject = parts[1].strip()

        # Create the email prompt with the input body
        prompt = self.email_prompt.format(body, "")
        if not prompt or not isinstance(prompt, str):
            raise ValueError(f"Invalid prompt: {prompt}")

        # Tokenize the prompt and subject line with truncation and padding
        inputs = self.tokenizer(
            [prompt],
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
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

# Define the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
# Create the train and test datasets
train_dataset = SubjectDataset('/home/ramch/AI-AUTOMATED-QA/AESLC/enron_subject_line/train', tokenizer)
test_dataset = SubjectDataset('/home/ramch/AI-AUTOMATED-QA/AESLC/enron_subject_line/test', tokenizer, is_test=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Disabled gradient accumulation
    gradient_checkpointing=True,
    fp16=False,
    evaluation_strategy='steps',
    logging_steps=1,
    logging_strategy='steps',
    warmup_steps=5,
    learning_rate=1e-4,  # Reduced learning rate
    weight_decay=0.01,
    lr_scheduler_type='linear',
    seed=3407,
    max_grad_norm=1.0,
)
# Add this line before trainer.train()
torch.cuda.empty_cache()
#trainer = Trainer(
#    model=model,
#    args=training_args,
#    train_dataset=train_dataset,
#    eval_dataset=test_dataset,
#    compute_metrics=lambda pred: {'accuracy': torch.sum(pred.label_ids == pred.predictions.argmax(-1)) / len(pred.label_ids)},
#)

#trainer.train()
from torch.utils.data import DataLoader

# Create DataLoader for training
train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,  # Smaller batch size due to memory constraints
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# Create DataLoader for evaluation
eval_dataloader = DataLoader(
    test_dataset,
    batch_size=1,  # Match batch size to training
    shuffle=False,  # No need to shuffle during evaluation
    num_workers=0,
    pin_memory=True
)
# Training Loop
num_epochs = 1  # Number of epochs for training
total_steps = len(train_dataloader)  # Total number of batches

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    total_samples = 0
    # Enumerate to get the batch index
    for step, batch in enumerate(train_dataloader):
        inputs = batch['input_ids'].to(device)  # Your input tensor
        labels = batch['labels'].to(device)     # Your target tensor
        optimizer.zero_grad()  # Clear the gradients

        # Forward pass
        outputs = model(inputs, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        # Calculate gradient norms
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        total_grad_norm += grad_norm
        optimizer.step()

        if step % 10 == 0:  # Adjust print frequency as needed
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}]")
            print(f"Loss: {loss.item()}")
            print(f"Gradient Norm: {grad_norm}")

    # Print average metrics for the epoch
    avg_loss = total_loss / len(train_dataloader)
    avg_grad_norm = total_grad_norm / len(train_dataloader)
    print(f"Epoch [{epoch+1}] - Average Loss: {avg_loss}")
    print(f"Epoch [{epoch+1}] - Average Gradient Norm: {avg_grad_norm}")
    running_loss = 0.0  # To keep track of the total loss for each epoch
    # Set the model to evaluation mode
    model.eval()
    total_eval_loss = 0.0
    with torch.no_grad():  # No need to calculate gradients during evaluation
    # Enumerate to get the batch index
     for step, batch in enumerate(eval_dataloader):
         with torch.no_grad():
            inputs = batch['input_ids'].to(device)  # Your input tensor
            labels = batch['labels'].to(device)     # Your target tensor
            # Forward pass
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            total_eval_loss += loss.item()
    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    print(f"Epoch [{epoch+1}] - Evaluation Loss: {avg_eval_loss}")

model.push_to_hub("nagthgr8/subject-prompt-t5-small")
tokenizer.push_to_hub("nagthgr8/subject-prompt-t5-small")
