import csv
import json

def csv_to_json(csv_files, json_file_path):
    data = {}
    
    for dataset_type, file_path in csv_files.items():
        dataset_data = []
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                dataset_data.append(row)
        data[dataset_type] = dataset_data
    
    with open(json_file_path, mode='w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

# Example usage
csv_files = {
    'TRAIN': '/home/ramch/AI-AUTOMATED-QA/AIMLQA/aiml-qa-train.csv',
    'DEV': '/home/ramch/AI-AUTOMATED-QA/AIMLQA/aiml-qa-dev.csv',
    'TEST': '/home/ramch/AI-AUTOMATED-QA/AIMLQA/aiml-qa-test.csv'
}
json_file_path = '/home/ramch/AI-AUTOMATED-QA/AIMLQA/aiml-qa-dataset.json'
csv_to_json(csv_files, json_file_path)
