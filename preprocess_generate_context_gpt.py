from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_context(answer, model_name='gpt2-medium', max_length=300):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    prompt = f"Generate a short context for '{answer}':"
#    prompt = f"What is the context for {answer}" 
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones_like(inputs)

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=8,
        no_repeat_ngram_size=4,
        early_stopping=True,
    )
    context = tokenizer.decode(outputs[0], skip_special_tokens=True)
    context = context.strip()
 # Split context into paragraphs
    paragraphs = context.split("\n\n")
    
    # Return only the first paragraph
    return paragraphs[0]    

#answer = "Concatenation combines two tensors by adding them together along a specified dimension."
#answer = "Classification and regression are distinct tasks, each with its own significance. The importance of each depends on the specific problem being addressed."
answer = "Slicing is essential for data manipulation, enabling extraction of specific portions. It is employed in tasks such as feature selection, cropping images, accessing time intervals, or filtering based on specific conditions."
context = generate_context(answer)
print("Generated Context:", context)
