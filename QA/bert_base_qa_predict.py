from transformers import AutoTokenizer, BertModel
import torch
import torch.nn as nn
import json

# Load the trained model and tokenizer
model_path = 'nagthgr8/qa-bert-base'

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
model = BertModel.from_pretrained(model_path)
qa_head = QuestionAnsweringHead()
tokenizer = AutoTokenizer.from_pretrained(model_path)
# Define the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
qa_head.to(device)

# Set the model to evaluation mode
model.eval()

def predict_answer(question, max_length=256):
    # Tokenize the question
    inputs = tokenizer(
        question,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    )

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'].to(model.device),
            attention_mask=inputs['attention_mask'].to(model.device)
        )

    # Extract hidden states
    hidden_states = outputs.last_hidden_state

    # Compute start and end logits using qa_head
    start_logits, end_logits = qa_head(hidden_states)

    # Get the start and end indices
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits) + 1  # +1 because end_idx is inclusive

    # Ensure that the predicted end index is after the start index
    if end_idx <= start_idx:
        return "No valid answer found"

    # Convert tokens back to string and extract the answer span from question
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_idx:end_idx])
    )

    # Clean up the answer (e.g., remove extra spaces)
    answer = answer.strip()

    return answer

# Load your dataset for inference
def load_inference_data(json_path, split='test'):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data[split]

# Define paths and parameters
json_path = '/home/ramch/AI-AUTOMATED-QA/AIMLQA/qa-dataset.json'
split = 'DEV'  # Change to 'TEST' or 'TRAIN' as needed
inference_data = load_inference_data(json_path, split)

# Iterate over the dataset and make predictions
for item in inference_data:
    question = item['question']
    context = item['context']
    
    # For dev and test splits, consider all possible answers
    if split in ['DEV', 'TEST']:
        answer1 = item.get('answer1', '')
        answer2 = item.get('answer2', '')
        # Predict answers
        predicted_answer = predict_answer(question)
        print(f"Question: {question}")
        print(f"Predicted Answer: {predicted_answer}")
    else:
        # For train split, only predict one answer
        predicted_answer = predict_answer(question, context)
        print(f"Question: {question}")
        print(f"Predicted Answer: {predicted_answer}")
