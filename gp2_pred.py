import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the saved model from Hugging Face
model_name = "nagthgr8/subject-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # You can also use tokenizer.pad_token = tokenize>

# Load the model
model = GPT2LMHeadModel.from_pretrained(model_name)
# Move the model to CUDA (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_subject_line(input_text, max_length=256, max_new_tokens=50):
    # Define the email prompt format
    email_prompt = """Below is an instruction that describes a task, paired with an input that provides further context.

### Instruction:
Predict the subject line of the email.

### Input:
{}

### Response:
"""

    # Format the input with the email prompt
    formatted_input = email_prompt.format(input_text)

    # Tokenize the input
    inputs = tokenizer(
        formatted_input,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    ).to(model.device)

    # Generate predictions with modified parameters
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=max_new_tokens,
        num_beams=4,
        temperature=0.7,  # Control randomness in generation
        top_p=0.9,  # Use nucleus sampling
        repetition_penalty=1.2,  # Penalize repeating the same phrases
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,  # Ensure generation stops at EOS token
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode the output
    predicted_subject = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the generated subject line by removing the prompt part
    response_start = "### Response:"
    if response_start in predicted_subject:
        predicted_subject = predicted_subject.split(response_start)[-1].strip()

    return predicted_subject

# Example input text
input_text = """As an introduction: The referenced counterparty is trying to straighten out the online products it wants to trade with us (it's open for some, but not all of the products it wants to trade).
When I looked at Profile Manager to see what they had already been approved to trade, I noticed that this counterparty was opened to trade many of the European products, including specifically European physical power products (which is unusual for a US based counterparty to trade).
Before I go in and change the profile for this counterparty, I wanted to check with you and see if London had specifically opened them to trade the products they were opened to trade (especially physical power).
What I usually do when I am approving US based counterparties, is that with respect to Non Us & Canadian products, as long as they are an Eligible Swap Participant I open them up for all products except European physical power, which I never open them for.
Also, Credit Derivatives, I never open anyone up for unless we really mean it.
So, my question to you is did London actively approve the non US & Canadian based products (especially physical power).
If you didn't, then I am going to take this as an oversight of the pre-approval process and shut them down for European physical power."""

predicted_subject = predict_subject_line(input_text)

print("Predicted Subject:", predicted_subject)
