from huggingface_hub import snapshot_download

local_dir = r"C:\PhD\BERT_Project\emBERT\data\SzegedNER"

# Downloads the entire repository structure locally
snapshot_download(
    repo_id="ficsort/SzegedNER",
    repo_type="dataset",
    local_dir=local_dir
)

print(f"Dataset downloaded completely to: {local_dir}")