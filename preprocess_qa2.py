import json
import re

json_file_path = '/home/ramch/AI-AUTOMATED-QA/AIMLQA/qa-dataset.json'
output_json_file_path = '/home/ramch/AI-AUTOMATED-QA/AIMLQA/qa-dataset.json'

# Function to clean the context by stripping leading/trailing spaces and removing unnecessary double quotes
def clean_context(text):
    text = text.strip()  # Remove leading/trailing spaces
    if text.startswith('"') and text.endswith('"'):  # Remove unnecessary double quotes
        text = text[1:-1]
    return text

# Function to read JSON file
def read_json(json_path):
    with open(json_path, mode='r') as file:
        data = json.load(file)
    return data

# Function to write JSON file
def write_json(json_path, data):
    with open(json_path, mode='w') as file:
        json.dump(data, file, indent=4)


# Main script execution
if __name__ == '__main__':
    # Read JSON
    json_data = read_json(json_file_path)

    for item in json_data.get('TRAIN', []):
        item['context'] = clean_context(item['question'])

    for item in json_data.get('TEST', []):
        item['context'] = clean_context(item['question'])

    for item in json_data.get('DEV', []):
        item['context'] = clean_context(item['question'])

    # Write updated JSON to file
    write_json(output_json_file_path, json_data)

    print(f'Updated JSON file saved to {output_json_file_path}')
