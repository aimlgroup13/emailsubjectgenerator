import torch
import os
import pandas as pd
import json
from transformers import TrainingArguments, TrainerCallback
from datasets import Dataset
import matplotlib.pyplot as plt
import datetime

# Continue with your model training setup
from unsloth import FastLanguageModel
import torch
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
    model_name = "unsloth/Qwen2.5-0.5B", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["question"]
    inputs       = examples["question"]
    if "answer1" in examples and examples["answer1"]:
        outputs = examples["answer1"]
    else:
        outputs = examples.get("answer", "No answer available.")
    texts = []
    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        # Construct text with EOS token
        text = alpaca_prompt.format(instruction, input_text, output_text) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }

# Path to your dataset.json file
json_file_path = '/home/ramch/AI-AUTOMATED-QA/AIMLQA/aiml-qa-dataset.json'

# Load JSON data into a Python list
with open(json_file_path, 'r') as file:
    data = json.load(file)
# Convert the list of dictionaries to a Hugging Face Dataset
train_dataset = Dataset.from_list(data['TRAIN'])
# Apply the formatting function to the dataset
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
eval_dataset = Dataset.from_list(data['DEV'])
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
from sklearn.model_selection import train_test_split
#dataset_dict = dataset.train_test_split(test_size=0.005)

# Verify the formatted dataset
print(train_dataset)
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
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

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        evaluation_strategy="steps",
        eval_steps=1,  # Evaluate every 1 step
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_on_each_node=True,
        report_to="all", 
   ),
)
# Ensure the model is moved to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
callback = LossGradientCallback(trainer)
trainer.add_callback(callback)
# Update trainer with callback
trainer.callbacks = [callback]
# Start training
trainer.train()
print("Training Completed")
model_name = "nagthgr8/qwen-qa"
print("Saving the model")
model.push_to_hub(model_name)
tokenizer.push_to_hub(model_name)
print("Done")
training_loss = callback.training_losses
eval_loss = callback.evaluation_losses
gradient_values = callback.gradient_values

print(training_loss)
print(eval_loss)
training_loss_dir = 'training-loss'
if not os.path.exists(training_loss_dir):
    os.makedirs(training_loss_dir)
now = datetime.datetime.now()
filename = now.strftime("%Y%m%d_%H%M%S") + "_qwen_qa_loss_plot.png"
fl_grad = now.strftime("%Y%m%d_%H%M%S") + "_qwen_qa_gradient_plot.png"
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
