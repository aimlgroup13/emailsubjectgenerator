import json
import torch
from transformers import pipeline
from evaluate import load
from rouge_score import rouge_scorer
from transformers import T5ForConditionalGeneration, T5Tokenizer
# Load the ROUGE metric
rouge = load("rouge")
max_length = 256
# Load the trained model and tokenizer
model_path = 'nagthgr8/qa-t5-base'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model = model.to("cuda").eval()
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

with open('t5_qa_avg_scores.json', 'w') as f:
    json.dump(avg_scores, f, indent=4)
