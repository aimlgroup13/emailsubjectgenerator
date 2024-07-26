from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Define the model and tokenizer
model_name = "nagthgr8/subject-phi3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the pipeline for text generation
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Example input
input_text = "Please authorize the following products for approval. Customers are expecting to see them on 1/14. PG&E Citygate-Daily Physical, BOM Physical, Monthly Index Physical Malin-Daily Physical, BOM Physical, Monthly Index Physical Keystone-Monthly Index Physical Socal Border-Daily Physical, BOM Physical, Monthly Index Physical PG&E Topock-Daily Physical, BOM Physical, Monthly Index Physical Please approve and forward to Dale Neuner Thank you"

# Generate predictions
predictions = generator(input_text, max_new_tokens=50, num_return_sequences=1)
print(predictions)
