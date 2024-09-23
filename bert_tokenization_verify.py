import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering
import json

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('nagthgr8/qa-bert-base')

def extract_answer(decoded_text, context):
    # Simple span-based extraction
    answer_start = decoded_text.find(context)
    answer_end = answer_start + len(context)


    # Slice decoded text to extract answer
    answer = decoded_text[answer_start:answer_end]


    return answer

def verify_tokenization(item):
    context = item['context']
    question = item['question']
    answer = item['answer'] if 'answer' in item else item['answer1']

    # Encode context and question
    inputs = tokenizer(
        question,
        context,
        return_tensors='pt',
        max_length=256,
        padding='max_length',
        truncation=True,
        return_offsets_mapping=True
    )

    input_ids = inputs['input_ids'].squeeze()
    attention_mask = inputs['attention_mask'].squeeze()
    offsets = inputs['offset_mapping'].squeeze()

    # Print tokenization info
    print("Question:", question)
    print("Context:", context)
    print("Answer:", answer)

    print("\nTokenized Input IDs:", input_ids)
    print("Tokenized Attention Mask:", attention_mask)
    print("Tokenized Offsets:", offsets)

    # Verify answer positions
    answer_start_char = context.find(answer)
    answer_end_char = answer_start_char + len(answer)

    start_position = 0
    end_position = 0

    for idx, (start_offset, end_offset) in enumerate(offsets):
        if start_offset <= answer_start_char and end_offset >= answer_start_char:
            start_position = idx
        if start_offset <= answer_end_char and end_offset >= answer_end_char:
            end_position = idx
            break

    print("\nCalculated Start Position:", start_position)
    print("Calculated End Position:", end_position)
    # Print tokenized input text
    print("\nTokenized Input Text:")
    for idx, input_id in enumerate(input_ids):
        print(f"Token {idx}: {tokenizer.decode(input_id, skip_special_tokens=True)}")
    # Decode input IDs
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)


    # Print decoded text
    print("Decoded Text:")
    print(decoded_text)
    # Extract answer
    answer = extract_answer(decoded_text, context)
    # Print extracted answer
    print("\nExtracted Answer:")
    print(answer)


# Load dataset
with open('/home/ramch/AI-AUTOMATED-QA/AIMLQA/qa-dataset.json', 'r') as f:
    data = json.load(f)


# Verify tokenization for the first item
#verify_tokenization(data['TRAIN'][0])

model = BertForQuestionAnswering.from_pretrained('nagthgr8/qa-bert-base')

# Set the model to evaluation mode
model.eval()

def predict_answer(question, context, max_length=256):
    # Tokenize the question and context
    inputs = tokenizer(
       question,
       context,
       return_tensors='pt',
       max_length=max_length,
       padding='max_length',
       truncation='only_second',
       return_offsets_mapping=True
       # This truncates the context if it exceeds max_length
    )
    print('Tokenizer max-length:',tokenizer.model_max_length)
    print('Input IDs shape:', inputs['input_ids'].shape)
    print('Input IDs length:', inputs['input_ids'][0].shape[0])
    print('Truncated context:', tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
    # Predict start and end logits
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'].to(model.device),
            attention_mask=inputs['attention_mask'].to(model.device)
        )

    # Extract start and end logits
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    # Get the start and end indices
    start_idx = torch.argmax(start_logits)
    end_idx_candidates = torch.topk(end_logits, k=5)[1][-1] + 1  # Consider top 5 end logits
    print('Start index:', start_idx)
    # Extract answer for each end index candidate
    answers = []
    if 'offset_mapping' in inputs:
        for end_idx in end_idx_candidates:
            answer_start = inputs['offset_mapping'][0][start_idx][0]
            answer_end = inputs['offset_mapping'][0][end_idx-1][1]
            answer = context[answer_start:answer_end]
            answers.append(answer)
    
    # Select the longest answer
    predicted_answer = max(answers, key=len)
    return predicted_answer
ans = predict_answer(data['TRAIN'][0]['question'], data['TRAIN'][0]['context'], 512)
print('Predicted Answer: ')
print(ans)
