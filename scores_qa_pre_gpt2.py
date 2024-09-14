import json
import torch
from transformers import pipeline
from evaluate import load
from rouge_score import rouge_scorer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# Load the ROUGE metric
rouge = load("rouge")
max_length = 512
# Load the trained model and tokenizer
model_path = 'nagthgr8/gpt2-qa'
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = model.to("cuda").eval()
tokenizer.pad_token = tokenizer.eos_token
with open('/home/ramch/AI-AUTOMATED-QA/AIMLQA/aiml-qa-dataset.json', 'r') as f:
    data = json.load(f)
testdataset = data.get('TEST')
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
    inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    inputs = inputs.to('cuda')
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
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.partition("Answer: ")[2]
    print("Question:")
    print(question)
    print("\nPredicted Answer:")
    print(answer)
    # Calculate ROUGE scores against each reference and store the best
    best_scores = {'rouge1': {'fmeasure': 0, 'precision': 0, 'recall': 0},
                   'rouge2': {'fmeasure': 0, 'precision': 0, 'recall': 0},
                   'rougeL': {'fmeasure': 0, 'precision': 0, 'recall': 0}}
    scores = scorer.score(answer, answer1)
    print(scores)
    for metric, score in scores.items():
        if score.fmeasure > best_scores[metric]['fmeasure']:
            best_scores[metric] = {'fmeasure': score.fmeasure,
                                   'precision': score.precision,
                                   'recall': score.recall}
    all_scores.append(best_scores)
    scores = scorer.score(answer, answer2)
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

with open('gpt2_pre_qa_avg_scores.json', 'w') as f:
    json.dump(avg_scores, f, indent=4)
