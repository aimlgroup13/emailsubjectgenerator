import json
import csv
import os
json_file_path = '/home/ramch/AI-AUTOMATED-QA/AIMLQA/qa-dataset.json'
csv_train = '/home/ramch/AI-AUTOMATED-QA/AIMLQA/train-dataset.csv'
csv_test = '/home/ramch/AI-AUTOMATED-QA/AIMLQA/test-dataset.csv'
csv_dev = '/home/ramch/AI-AUTOMATED-QA/AIMLQA/dev-dataset.csv'

def json_to_csv(data, csv_path):
    existing_data = {}

    # Step 1: Read existing CSV data if the file exists
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                question = row['question']
                answer = row['answer']
                if question in existing_data:
                    existing_data[question].append(answer)
                else:
                    existing_data[question] = [answer]

    # Step 2: Process JSON data
    for entry in data:
        question = entry['question']
        answers = []

        if 'answer' in entry:
            answers.append(entry['answer'])
        if 'answer1' in entry:
            answers.append(entry['answer1'])
        if 'answer2' in entry:
            answers.append(entry['answer2'])

        if answers:
            combined_answer = ' '.join(answers)  # Combine answers if there are multiple
            if question in existing_data:
                existing_data[question].append(combined_answer)
            else:
                existing_data[question] = [combined_answer]

    # Step 3: Write updated data to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question', 'answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for question, answers in existing_data.items():
            combined_answer = ' '.join(answers)  # Combine all answers for this question
            writer.writerow({'question': question, 'answer': combined_answer})

# Function to read JSON file
def read_json(json_path):
    with open(json_path, mode='r') as file:
        data = json.load(file)
    return data

if __name__ == "__main__":

    # Read JSON
    json_data = read_json(json_file_path)
    train_data = json_data.get('TRAIN', [])
    json_to_csv(train_data, csv_train)
    test_data = json_data.get('TEST', [])
    json_to_csv(test_data, csv_test)
    dev_data = json_data.get('DEV', [])
    json_to_csv(dev_data, csv_dev)
