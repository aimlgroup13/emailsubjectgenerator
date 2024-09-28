import json
import torch
from transformers import pipeline
from evaluate import load
from rouge_score import rouge_scorer
# Load the ROUGE metric
rouge = load("rouge")

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "sbtraining2020/qwen_qa1", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        
    )
    model = model.to("cuda").eval()
    tokenizer_padding_side = "left"
    # email_prompt = Copied from above
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
with open('/home/ramch/AI-AUTOMATED-QA/AIMLQA/aiml-qa-dataset.json', 'r') as f:
    data = json.load(f)
testdataset = data.get('TEST')

# Define email prompt and EOS token
email_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def format_input(instruction, input_text, response_text=""):
    return email_prompt.format(instruction, input_text, response_text) + EOS_TOKEN
max_length = 256
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

all_scores = []
print("Calculating the scores..")

# Use index-based access for Dataset objects
for i in range(len(testdataset[:100])):
    example = testdataset[i]  # Access element like a list
    question = example['question']
    answer1 = example['answer1']
    answer2 = example['answer2']  # List of references
    # Tokenize input text and generate prediction
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
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Question:")
    print(question)
    print("\nPredicted Answer:")
    print(answer)
    # Calculate ROUGE scores against each reference and store the best
    best_scores = {'rouge1': {'fmeasure': 0, 'precision': 0, 'recall': 0},
                   'rouge2': {'fmeasure': 0, 'precision': 0, 'recall': 0},
                   'rougeL': {'fmeasure': 0, 'precision': 0, 'recall': 0}}
    scores = scorer.score(answer, answer1)
    for metric, score in scores.items():
        if score.fmeasure > best_scores[metric]['fmeasure']:
            best_scores[metric] = {'fmeasure': score.fmeasure,
                                   'precision': score.precision,
                                   'recall': score.recall}
    all_scores.append(best_scores)
    scores = scorer.score(answer, answer2)
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

with open('qwen_avg_scores.json', 'w') as f:
    json.dump(avg_scores, f, indent=4)
