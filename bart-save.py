from transformers import BartForConditionalGeneration, BartTokenizer, AutoConfig

# Define the path to your checkpoint folder
checkpoint_path = "/home/ramch/AI-AUTOMATED-QA/results/checkpoint-4000"

# Load the model and tokenizer
model = BartForConditionalGeneration.from_pretrained(checkpoint_path)
tokenizer = BartTokenizer.from_pretrained(checkpoint_path)

# Push the model and tokenizer directly to your Hugging Face account
model.push_to_hub("nagthgr8/subject-bart")
tokenizer.push_to_hub("nagthgr8/subject-bart")
