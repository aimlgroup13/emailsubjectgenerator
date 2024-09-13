from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoConfig

# Define the path to your checkpoint folder
checkpoint_path = "/home/ramch/AI-AUTOMATED-QA/results/checkpoint-2709"

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)

# Push the model and tokenizer directly to your Hugging Face account
#model.push_to_hub("nagthgr8/subject-t5-small")
tokenizer.push_to_hub("nagthgr8/subject-t5-small")
