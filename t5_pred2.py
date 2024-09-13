import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the saved model from Hugging Face
model_name = "nagthgr8/subject-t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load the T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Move the model to CUDA (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_subject_line( input_text, max_length=512):
    # Tokenize the input text
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    inputs = inputs.to(device)

    # Specify the decoder input IDs (starting token)
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])  # Start with pad token
    decoder_input_ids = decoder_input_ids.to(device)

    # Get the predicted subject line
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
input_text = """At the request of Michael Bridges, I am attaching our proposed form of Letter  of Interest.
If the agreement meets with your approval, please return a copy  to me via fax no.
(713) 646-3490.
If you have any comments or questions you  would like to discuss, please do not hesitate to call me at (713) 853-3399,  Mike at (713) 345-4079 or Bob Shults at (713) 853-0397.
We look forward to  hearing from you."""
predicted_subject = predict_subject_line(input_text)

print("Predicted Subject:", predicted_subject)
