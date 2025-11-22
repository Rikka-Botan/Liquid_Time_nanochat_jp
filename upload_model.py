from huggingface_hub import login, upload_folder, upload_file
import os
login(token="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")
upload_folder(
    folder_path="/home/ubuntu/.cache/nanochat/chatsft_checkpoints/d20",
    path_in_repo="pytorch_model.bin",  # Default model checkpoint name
    repo_id="RikkaBotan/nanochat_d20_saint_iberis",  # Replace with your username/repo name
    repo_type="model"
)
upload_folder(
    folder_path="/home/ubuntu/.cache/nanochat/tokenizer",
    path_in_repo="tokenizer.json",  # Default model checkpoint name
    repo_id="RikkaBotan/nanochat_d20_saint_iberis",  # Replace with your username/repo name
    repo_type="model"
)
upload_folder(
    folder_path="/home/ubuntu/.cache/nanochat/report",
    path_in_repo="report",  # Default model checkpoint name
    repo_id="RikkaBotan/nanochat_d20_saint_iberis",  # Replace with your username/repo name
    repo_type="model"
)