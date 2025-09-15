import os
from huggingface_hub import HfApi, create_repo, upload_folder


def main():
    repo_id = os.environ.get("HF_REPO", "your-username/gpt-oss-20b-qlora-vsfc")
    adapter_dir = os.environ.get("ADAPTER_DIR", "gpt-oss-20b-qlora-vsfc")
    private = os.environ.get("HF_PRIVATE", "false").lower() == "true"

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Please set HF_TOKEN with write access.")

    api = HfApi(token=token)
    create_repo(repo_id, private=private, exist_ok=True, token=token)

    # Upload the LoRA adapter folder
    upload_folder(
        repo_id=repo_id,
        folder_path=adapter_dir,
        commit_message="Upload GPT-OSS 20B QLoRA adapter for UIT-VSFC",
        token=token,
    )

    print(f"Uploaded adapter from '{adapter_dir}' to 'https://huggingface.co/{repo_id}'")


if __name__ == "__main__":
    main()


