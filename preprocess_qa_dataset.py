import csv
import json
import re
# Paths to your files
csv_file_path = '/home/ramch/AI-AUTOMATED-QA/AIMLQA/question_context.csv'
json_file_path = '/home/ramch/AI-AUTOMATED-QA/AIMLQA/qa-dataset.json'
output_json_file_path = '/home/ramch/AI-AUTOMATED-QA/AIMLQA/qa-dataset.json'

# Function to read CSV and return a dictionary of questions and contexts
def read_csv_to_dict(csv_path):
    question_context_dict = {}
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            question = row['question']
            context = row['context']
            question_context_dict[question] = context
    return question_context_dict

# Function to read JSON file
def read_json(json_path):
    with open(json_path, mode='r') as file:
        data = json.load(file)
    return data

# Function to write JSON file
def write_json(json_path, data):
    with open(json_path, mode='w') as file:
        json.dump(data, file, indent=4)

# Function to clean text
def clean_text(text):
    # Remove leading/trailing spaces and new lines
    text = text.strip()
    # Replace new line characters with a space
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters (except basic punctuation like commas and periods)
    text = re.sub(r'[^\w\s,.!?]', '', text)
    return text

# Update the JSON data with contexts from CSV
def update_json_with_context(json_data, context_dict):
    # Update TRAIN section
    for item in json_data.get('TRAIN', []):
        question = item.get('question')
        if question.lower() in context_dict:
            item['context'] = context_dict[question.lower()]
    
    # Update TEST section
    for item in json_data.get('TEST', []):
        question = item.get('question')
        if question.lower() in context_dict:
            item['context'] = context_dict[question.lower()]
    
    # Update DEV section
    for item in json_data.get('DEV', []):
        question = item.get('question')
        if question.lower() in context_dict:
            item['context'] = context_dict[question.lower()]

# Main script execution
if __name__ == '__main__':
    # Read CSV
    question_context_dict = read_csv_to_dict(csv_file_path)
    context_dict = {key.lower(): value for key, value in question_context_dict.items()}
    print(context_dict)
    # Read JSON
    json_data = read_json(json_file_path)
    
    # Update JSON
    update_json_with_context(json_data, context_dict)
    
    # Write updated JSON to file
    write_json(output_json_file_path, json_data)
    
    print(f'Updated JSON file saved to {output_json_file_path}')
