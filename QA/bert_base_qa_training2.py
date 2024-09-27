from transformers import AutoTokenizer, BertModel
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from torch.optim import AdamW
import matplotlib.pyplot as plt
import datetime
# Define a custom classification head for QA
class QuestionAnsweringHead(nn.Module):
    def __init__(self):
        super(QuestionAnsweringHead, self).__init__()
        self.start_logits = nn.Linear(768, 1)  # Adjusted size
        self.end_logits = nn.Linear(768, 1)  # Adjusted size

    def forward(self, hidden_states):
        start_logits = self.start_logits(hidden_states)
        end_logits = self.end_logits(hidden_states)
        return start_logits, end_logits

# Initialize BERT model and QA head
model = BertModel.from_pretrained('bert-base-uncased')
qa_head = QuestionAnsweringHead()
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# Define the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
qa_head.to(device)

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
#model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
#tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

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
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
epochs = []
training_loss = []
eval_loss = []
num_epochs = 5
gradient_values = []
loss_values = []

# Train the model
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    epochs.append(epoch + 1)  # Track epoch numbers
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        start_logits, end_logits = qa_head(hidden_states)

        # Calculate losses
        start_loss = criterion(start_logits.squeeze(-1), start_positions)
        end_loss = criterion(end_logits.squeeze(-1), end_positions)

        # Combine losses
        loss = start_loss + end_loss
        loss_values.append(loss.item())
        # Backward pass
        loss.backward()
        # Access gradients
        total_norm = 0
        count = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
                count += 1
        total_norm = total_norm ** 0.5
        gradient_values.append(total_norm)
        # Update model parameters
        optimizer.step()
        # Print loss
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch+1}, Loss: {avg_train_loss}')
    training_loss.append(avg_train_loss)
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
              attention_mask=attention_mask
            )
            hidden_states = outputs.last_hidden_state
            start_logits, end_logits = qa_head(hidden_states)

            # Calculate losses
            start_loss = criterion(start_logits.squeeze(-1), start_positions)
            end_loss = criterion(end_logits.squeeze(-1), end_positions)
            loss = start_loss + end_loss
            total_eval_loss += loss.item()

    avg_eval_loss = total_eval_loss / len(dev_dataloader)
    print(f"Epoch [{epoch+1}] - Dev Loss: {avg_eval_loss:.4f}")
    eval_loss.append(avg_eval_loss)

model_name = "qa-bert-base"
save_directory = f"results/{model_name}"
# Create the directory if it does not exist
#if not os.path.exists(save_directory):
#    os.makedirs(save_directory)

# Optionally save the model
#model.save_pretrained(save_directory)
#tokenizer.save_pretrained(save_directory)
# Push model and tokenizer to Hugging Face Model Hub
#model.push_to_hub(f"nagthgr8/{model_name}", commit_message="Initial model push")
#tokenizer.push_to_hub(f"nagthgr8/{model_name}", commit_message="Initial tokenizer push")
# Create directory if it doesn't exist
training_loss_dir = 'training-loss'
if not os.path.exists(training_loss_dir):
    os.makedirs(training_loss_dir)
now = datetime.datetime.now()
filename = now.strftime("%Y%m%d_%H%M%S") + "_bert_base_qa_loss_plot.png"
fl_grad = now.strftime("%Y%m%d_%H%M%S") + "_bert_base_qa_gradient_plot.png"
fig_loss, ax_loss = plt.subplots(figsize=(8, 6))
fig_gradient, ax_gradient = plt.subplots(figsize=(8, 6))
ax_loss.plot(epochs, training_loss, label='Training Loss', marker='o', color='blue')
ax_loss.plot(epochs, eval_loss, label='Dev Loss', marker='o', color='orange')
ax_loss.set_xlabel('Epochs')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('Training and Dev Loss over Epochs')
ax_loss.legend()
ax_loss.grid(True)
fig_loss.savefig(os.path.join(training_loss_dir, filename))
# Plot gradient norms in the second subplot
ax_gradient.plot(gradient_values)
ax_gradient.set_xlabel('Iterations')
ax_gradient.set_ylabel('Gradient Norm')
ax_gradient.set_title('Gradient Norm over Iterations')
fig_gradient.savefig(os.path.join(training_loss_dir, fl_grad))
plt.close('all')
