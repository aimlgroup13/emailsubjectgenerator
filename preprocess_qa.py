import json

# Load your JSON data from a file
with open('/home/ramch/AI-AUTOMATED-QA/AIMLQA/qa-dataset.json', 'r') as file:
    data = json.load(file)

# Open a text file to save questions where context is missing
with open('/home/ramch/AI-AUTOMATED-QA/AIMLQA/questions.txt', 'w') as output_file:
    # Iterate through TRAIN, TEST, and DEV
    for split in ['TRAIN', 'TEST', 'DEV']:
        for entry in data.get(split, []):
            if entry['context'] == entry['question']:
                # Write the question to the output file if context is missing
                output_file.write(f"{entry['question']}\n")

print("Questions with missing context have been written to 'questions.txt'.")
