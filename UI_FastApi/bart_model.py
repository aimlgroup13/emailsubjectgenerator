import torch
from transformers import BartForConditionalGeneration, BartTokenizer
load_in_4bit = True

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
MODEL_NAME = "sbtraining2020/email_bart"
hf_tokenizer = BartTokenizer.from_pretrained( MODEL_NAME) # "sbtraining2020/esubjectgen_llama31_clean")
hf_model = BartForConditionalGeneration.from_pretrained(MODEL_NAME,
               device_map="auto",
               load_in_4bit=load_in_4bit,
               ) 

def summarize_text(text, max_length=500):
    """
    Generates a email subject for the given text using a pre-trained model.

    Args:
        text (str): Email text to generate subject.
        max_length (int): The maximum length of the input text for the model.

    Returns:
        str: The generated subject for the input text.
    """
    # Encode the input text using the tokenizer. The 'pt' indicates PyTorch tensors.
    inputs = hf_tokenizer.encode(text, return_tensors="pt", max_length=max_length, truncation=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the encoded text to the same device as the model (e.g., GPU or CPU)
    inputs = inputs.to(device)
    
    # Generate summary IDs with the model. num_beams controls the beam search width.
    # early_stopping is set to False for a thorough search, though it can be set to True for faster results.
    summary_ids = hf_model.generate(inputs, max_length=500, num_beams=30, early_stopping=True)

    # Decode the generated IDs back to text, skipping special tokens like padding or EOS.
    summary = hf_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Return the generated summary
    return summary

#email_text = """Plove is going to go to Dallas.
#We are going to leave next Friday when he  gets done (7ish) and go up for the game.
#The game is at 11 in the morning,  so we will come home directly after it.
#Plove says he has a friend who has a  place in Dallas that we can crash at if we don't want to pay for a hotel.
#Do you want to go?
#        """
        
#result = summarize_text(email_text)

#print(result)
        