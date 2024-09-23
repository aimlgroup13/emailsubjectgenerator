#from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
#from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def generate_context(answer, model_name='t5-base', max_length=300):
#    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
#    tokenizer = AutoTokenizer.from_pretrained(model_name)
#    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#    prefix = f"{answer} is"
#    prompt = f"Explain {answer} in the context of deep learning and tensor operations."
#    prompt = f"Describe a scenario where {answer} is applied in deep learning."
    #prompt = f"Explain the concept of '{answer}' in the context of deep learning and tensor operations:"
# Keyword-based text generation
#    keywords = answer.split()
#    prompt = "Generate context related to " + ", ".join(keywords)
    #inputs = tokenizer.encode(prefix + " " + prompt, return_tensors='pt')
    prompt = f"What is the context for {answer}" 
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones_like(inputs)

#    outputs = model.generate(
#        inputs,
#        attention_mask=attention_mask,
#        max_length=max_length,
#        num_beams=8,
#        no_repeat_ngram_size=4,
#        early_stopping=True,
#    )
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=12,
        no_repeat_ngram_size=6,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        early_stopping=True,
    )
#    outputs = model.generate(
#        inputs,
#        attention_mask=attention_mask,
#        max_length=max_length,
#        min_length=100,
#        num_beams=12,
#        no_repeat_ngram_size=8,
#        do_sample=True,
#        top_k=30,
#        top_p=0.9,
#        early_stopping=True,
#    )
    context = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #context = context[len(prefix + prompt):].strip()
    context = context.strip()
    return context

#answer = "Concatenation combines two tensors by adding them together along a specified dimension."
answer = "Classification and regression are distinct tasks, each with its own significance. The importance of each depends on the specific problem being addressed."
context = generate_context(answer)
print("Generated Context:", context)
