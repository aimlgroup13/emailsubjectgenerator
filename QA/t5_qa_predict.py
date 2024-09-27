from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import json

# Load the trained model and tokenizer
model_path = 'nagthgr8/qa-t5-base'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

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

    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'].to(model.device),
            attention_mask=inputs['attention_mask'].to(model.device),
            max_length=max_length
        )

    # Decode the predictions
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return decoded_output

# Load your dataset for inference
#def load_inference_data(json_path, split='test'):
#    with open(json_path, 'r') as f:
#        data = json.load(f)
#    return data[split]

# Define paths and parameters
#json_path = '/home/ramch/AI-AUTOMATED-QA/AIMLQA/aiml-qa-dataset.json'
#split = 'DEV'  # Change to 'TEST' or 'TRAIN' as needed
#inference_data = load_inference_data(json_path, split)

# Iterate over the dataset and make predictions
#for item in inference_data:
#    question = item['question']
    
    # For dev and test splits, consider all possible answers
#    if split in ['DEV', 'TEST']:
#        answer1 = item.get('answer1', '')
#        answer2 = item.get('answer2', '')
#        # Predict answers
#        predicted_answer1 = predict_answer(question, tokenizer, model)
#        predicted_answer2 = predict_answer(question, tokenizer, model)
#        print(f"Question: {question}")
#        print(f"Predicted Answer 1: {predicted_answer1}")
#        print(f"Predicted Answer 2: {predicted_answer2}")
#    else:
#        # For train split, only predict one answer
#        predicted_answer = predict_answer(question, tokenizer, model)
#        print(f"Question: {question}")
#        print(f"Predicted Answer: {predicted_answer}")
