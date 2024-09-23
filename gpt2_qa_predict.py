from torch.utils.data import Dataset
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Replace 'your-model-directory' with the directory where your model is saved
model = GPT2LMHeadModel.from_pretrained('nagthgr8/gpt2-qa')
tokenizer = GPT2Tokenizer.from_pretrained('nagthgr8/gpt2-qa')
# Load your dataset for inference
#def load_inference_data(json_path, split='test'):
#    with open(json_path, 'r') as f:
#        data = json.load(f)
#    return data[split]

# Define paths and parameters
#json_path = '/home/ramch/AI-AUTOMATED-QA/AIMLQA/aiml-qa-dataset.json'
#split = 'DEV'  # Change to 'TEST' or 'TRAIN' as needed
#dev_dataset = load_inference_data(json_path, split)

import torch

def generate_answer(question, max_length=512):
    # Tokenize the input question
    inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    
    # Generate a response
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    
    # Decode the generated response
    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_answer

#for item in dev_dataset:
#    question = item['question']
#    true_answer = item.get('answer1')  # or item.get('answer2'), depending on your setup
    
#    # Generate the answer from the model
#    generated_answer = generate_answer(question, model, tokenizer)
    
#    # Print or save the question, true answer, and generated answer
#    print(f"Question: {question}")
#    print(f"True Answer: {true_answer}")
#    print(f"Generated Answer: {generated_answer}")
#    print("-" * 50)

#from sklearn.metrics import f1_score, accuracy_score

# Collect generated answers and true answers
#true_answers = [item.get('answer1') for item in dev_dataset]
#generated_answers = [generate_answer(item['question'], model, tokenizer) for item in dev_dataset]

# Compute evaluation metrics
# Assuming you have a method to compute EM or F1 score
#em_score = compute_exact_match(true_answers, generated_answers)
#f1 = compute_f1(true_answers, generated_answers)

#print(f"Exact Match Score: {em_score}")
#print(f"F1 Score: {f1}")

