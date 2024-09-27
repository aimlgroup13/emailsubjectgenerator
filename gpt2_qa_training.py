from torch.utils.data import Dataset
import torch
import os
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer, TrainerCallback
from torch.optim import AdamW
import matplotlib.pyplot as plt
import datetime
import pprint
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

class LossGradientCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.training_losses = []
        self.evaluation_losses = []
        self.gradient_values = []
    def on_train_begin(self, args, state, control, **kwargs):
        self._train_loss = 0
        self._globalstep = 0
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        for key, value in logs.items():
            if "loss" in key.lower():
                self.training_losses.append(value)
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.evaluation_losses.append(metrics['eval_loss'])

    def on_step_end(self, args, state, control, **kwargs):
        # Logging gradient norms requires accessing the model's parameters
        # and computing the gradient norms manually
        gradient_norm = 0
        for param in self.trainer.model.parameters():
            if param.grad is not None:
                gradient_norm += param.grad.norm().item() ** 2
        gradient_norm = gradient_norm ** 0.5
        self.gradient_values.append(gradient_norm)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="steps",
    eval_steps=1,  # Evaluate every 1 step
    logging_steps=1,
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_on_each_node=True,
    report_to="all",
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
)
# Initialize callback with trainer
callback = LossGradientCallback(trainer)
trainer.add_callback(callback)
# Update trainer with callback
trainer.callbacks = [callback]
# Train model
trainer.train()

model_name = 'gpt2-qa'
save_directory = f"results/{model_name}"
#Create the directory if it does not exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

model.push_to_hub(f"nagthgr8/{model_name}", commit_message="Initial model push")
tokenizer.push_to_hub(f"nagthgr8/{model_name}", commit_message="Initial tokenizer push")
# Access captured values
training_loss = callback.training_losses
eval_loss = callback.evaluation_losses
gradient_values = callback.gradient_values

print(training_loss)
print(eval_loss)


training_loss_dir = 'training-loss'
if not os.path.exists(training_loss_dir):
    os.makedirs(training_loss_dir)
now = datetime.datetime.now()
filename = now.strftime("%Y%m%d_%H%M%S") + "_gpt2_qa_loss_plot.png"
fl_grad = now.strftime("%Y%m%d_%H%M%S") + "_gpt2_qa_gradient_plot.png"
fig_loss, ax_loss = plt.subplots(figsize=(8, 6))
fig_gradient, ax_gradient = plt.subplots(figsize=(8, 6))
ax_loss.plot(range(len(training_loss)), training_loss, label='Training Loss', marker='o', color='blue')
ax_loss.plot(range(len(eval_loss)), eval_loss, label='Dev Loss', marker='o', color='orange')
ax_loss.set_xlabel('Epochs')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('Training and Dev Loss over Epochs for GPT2')
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
model_name = "gpt2-qa"
save_directory = f"results/{model_name}"
