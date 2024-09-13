import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the saved model from Hugging Face
model_name = "nagthgr8/subject-prompt-t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load the T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Move the model to CUDA (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
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
        max_length=max_length,    # Set the maximum length for the output
        num_beams=5,              # Beam search for better results
        early_stopping=True,      # Stop generation when end-of-sequence token is generated
        temperature=1.0           # Adjust creativity (0.7 is often a good starting point)
    )

    # Decode the generated output
    predicted_subject = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return predicted_subject
# Make a prediction

#input_text = """As an introduction:  The referenced counterparty is trying to straighten out  the online products it wants to trade with us (it's open for some, but not  all of the products it wants to trade).
#When I looked at Profile Manager to see what they had already been approved  to trade, I noticed that this counterparty was opened to trade many of the  European products, including specifically European physical power products  (which is unusual for a US based counterparty to trade).
#Before I go in and change the profile for this counterparty, I wanted to  check with you and see if London had specifically opened them to trade the  products they were opened to trade (especially physical power).
#What I usually do when I am approving US based counterparties, is that with  respect to Non Us & Canadian products, as long as they are an Eligible Swap  Participant I open them up for all products except European physical power,  which I never open them for.
#Also, Credit Derivatives, I never open anyone  up for unless we really mean it.
#So, my question to you is did London actively approve the non US & Canadian  based products (especially physical power).
#If you didn't, then I am going  to take this as an oversight of the pre-approval process and shut them down  for European physical power."""
#predicted_subject = predict_subject_line(input_text)

#print("Predicted Subject:", predicted_subject)
