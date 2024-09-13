import torch
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline
from evaluate import load
from rouge_score import rouge_scorer
# Load the ROUGE metric
rouge = load("rouge")

#model_name = "nagthgr8/subject-prompt-t5-small"
model_name = "t5-small"
# Load the model
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Move the model to CUDA (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with open('/home/ramch/AI-AUTOMATED-QA/testdataset.json', 'r') as f:
    testdataset = json.load(f)
email_prompt = """Below is an instruction that describes a task, paired with an input that provides further>

### Instruction:
Predict the subject line of the email.

### Input:
{}

### Response:
{}"""
def predict_subject_line(input_text, max_length=512):
    # Format the input text with the email prompt
    formatted_input = email_prompt.format(input_text, "")

    # Tokenize the input text
    inputs = tokenizer(
        [formatted_input],
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    ).to(device)

    # Determine which token to use for starting the sequence
    if tokenizer.bos_token_id is not None:
        # Use the BOS token if available
        start_token_id = tokenizer.bos_token_id
    elif tokenizer.pad_token_id is not None:
        # Fall back to using the pad token
        start_token_id = tokenizer.pad_token_id
    else:
        raise ValueError("Tokenizer does not have a BOS or PAD token defined.")

    # Prepare the decoder input IDs using the chosen token
    decoder_input_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=device)

    # Generate predictions
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,
        num_beams=4,  # Using beam search for better predictions
        early_stopping=True,
        decoder_input_ids=decoder_input_ids
    )

    # Decode the predicted subject line
    predicted_subject = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return predicted_subject

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
all_scores = []
print("Calculating the scores..")
# Use index-based access for Dataset objects
for i in range(len(testdataset[:100])):
    example = testdataset[i]  # Access element like a list
    instruction = example['instruction']
    input_text = example['input']
    references = example['output']  # List of references
    prediction = predict_subject_line(input_text)
    # Calculate ROUGE scores against each reference and store the best
    best_scores = {'rouge1': {'fmeasure': 0, 'precision': 0, 'recall': 0},
                   'rouge2': {'fmeasure': 0, 'precision': 0, 'recall': 0},
                   'rougeL': {'fmeasure': 0, 'precision': 0, 'recall': 0}}
    print(references)
    for ref in references:
        print("ref", ref)
        scores = scorer.score(prediction, ref)
        print(scores)
        for metric, score in scores.items():
            if score.fmeasure > best_scores[metric]['fmeasure']:
                best_scores[metric] = {'fmeasure': score.fmeasure,
                                       'precision': score.precision,
                                       'recall': score.recall}
    print(best_scores)
    all_scores.append(best_scores)

# Aggregate scores (e.g., calculate average)
avg_scores = {
    'rouge1': {'fmeasure': 0, 'precision': 0, 'recall': 0},
    'rouge2': {'fmeasure': 0, 'precision': 0, 'recall': 0},
    'rougeL': {'fmeasure': 0, 'precision': 0, 'recall': 0}
}
print("Calculating avg scores..")
for scores in all_scores:
    for metric in avg_scores:
        for key in ['fmeasure', 'precision', 'recall']:
            if key in scores[metric]:
                avg_scores[metric][key] += scores[metric][key]

for metric in avg_scores:
    for key in ['fmeasure', 'precision', 'recall']:
        avg_scores[metric][key] /= len(all_scores)

# Print the average ROUGE scores
print(avg_scores)

with open('t5_pre-avg_scores.json', 'w') as f:
    json.dump(avg_scores, f, indent=4)
