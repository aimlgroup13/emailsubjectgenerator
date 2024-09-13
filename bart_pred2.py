import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BartForConditionalGeneration, BartTokenizer

MODEL_NAME = "sbtraining2020/email_bart"

max_seq_length = 2048
dtype = None
load_in_4bit = True

from transformers import AutoTokenizer, AutoModelForCausalLM
hf_tokenizer =BartTokenizer.from_pretrained(MODEL_NAME) # AutoTokenizer.from_pretrained( MODEL_NAME) # "sbtraining2020/esubjectgen_llama31_clean")
hf_model = BartForConditionalGeneration.from_pretrained(MODEL_NAME) #"sbtraining2020/esubjectgen_llama31_clean")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_model.to(device)
def predict_bart(text, max_length=500):
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

    # Move the encoded text to the same device as the model (e.g., GPU or CPU)
    inputs = inputs.to(device)

    # Generate summary IDs with the model. num_beams controls the beam search width.
    # early_stopping is set to False for a thorough search, though it can be set to True for faster results.
    subject_ids = hf_model.generate(inputs, max_length=500, num_beams=30, early_stopping=False)

    # Decode the generated IDs back to text, skipping special tokens like padding or EOS.
    subject = hf_tokenizer.decode(subject_ids[0], skip_special_tokens=True)

    # Return the generated subject
    return subject


#email_text = """
#Michelle Here are my very minor comments.
#However we still need to wait on any additions, based on meeting with SME's today.
#One concern is the firing of the learner who performs  bad in the final two scenarios.
#Do we face any copyright issues using the CNN type themes?
#In addition, I think we need to stay clear of anything that remotely seems like California or anything that really happen with Enron?
#(i.e.So-cal Waha) In addition, comments on regulatory issues may be a problem (i.e.California Legislature).
#Sheri  When you read all the scripts together and due to the similar mechanics being taught it appears very repetitious.
#Thus I do believe we need to maybe use a "Dateline" type theme for one, and a "60 Minute" type theme for another scenario vice just the CNN type theme.
#In the last two scenarios can we include a promotion out of the associate program for the stellar performers (i.e.title change to manager)?
#Cheers Kirk
#"""

#result = predict_bart(email_text)
#print(result)
