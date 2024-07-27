import os
import pandas as pd
folder_path = '/home/ramch/AI-AUTOMATED-QA/AESLC/enron_subject_line/train'
email_prompt = """
Instruction: {instruction}

Input: {input}

Output: {output}
"""
# List to hold dataframes
dataframes = []

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.subject'):
        file_path = os.path.join(folder_path, filename)
        
        # Read the file content (assuming it's in a text format)
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Create a dataframe for the content
        df = pd.DataFrame({'filename': [filename], 'content': [content]})
        
        # Append the dataframe to the list
        dataframes.append(df)

# Concatenate all dataframes into one
dataset = pd.concat(dataframes, ignore_index=True)

# Display the dataset
print(dataset.head())
def formatting_prompts_func(examples):
    texts = []
    problematic_records = []

    for idx, content in enumerate(examples["content"]):
        try:
            # Split the content by '@subject'
            parts = content.split('@subject')

            if len(parts) == 2:
                # The first part is the input text
                input_text = parts[0].strip()
                # The second part is the subject line
                output_text = parts[1].strip()
            else:
                # Handle cases where '@subject' might not be present
                input_text = content.strip()
                output_text = ""

            # Define the instruction if needed; otherwise, leave it empty
            instruction = "Extract the subject from the text."

            # Format the text with EOS_TOKEN
            text = email_prompt.format(instruction=instruction, input=input_text, output=output_text) + EOS_TOKEN
            texts.append(text)
        
        except Exception as e:
            # Log the problematic record
            problematic_records.append({
                "index": idx,
                "content": content,
                "error": str(e)
            })
    
    # Dump the problematic records to a file
    if problematic_records:
        with open('problematic_records.json', 'w') as f:
            import json
            json.dump(problematic_records, f, indent=4)

    return { "text": texts }

# Assuming `dataset` is your combined DataFrame
formatted_dataset = formatting_prompts_func(dataset)
