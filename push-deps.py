from huggingface_hub import HfApi

model_name = "nagthgr8/subject-phi3"
local_checkpoint_dir = "/home/ramch/AI-AUTOMATED-QA/checkpoint_dir"

files_to_upload = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "added_tokens.json",
    "all_results.json",
    "eval_results.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "training_args.bin"
]

api = HfApi()

for file in files_to_upload:
    api.upload_file(
        path_or_fileobj=f"{local_checkpoint_dir}/{file}",
        path_in_repo=file,
        repo_id=model_name,
        repo_type="model"
    )

print("All necessary files have been uploaded.")
