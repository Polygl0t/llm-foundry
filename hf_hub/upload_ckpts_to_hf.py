"""
Upload Model Checkpoints to Hugging Face Hub with Branch Organization

This script uploads training checkpoints to a Hugging Face model repository,
organizing each checkpoint into its own git branch for easy version management.

Directory Structure Expected:
    root_dir/
    ├── main/           # Goes to main branch
    │   ├── config.json
    │   ├── model.safetensors
    │   └── ...
    ├── step_00100/     # Goes to step_00100 branch
    ├── step_00200/     # Goes to step_00200 branch
    └── metadata.csv    # Uploaded to main branch

Usage:
    export HF_TOKEN=your_token_here
    python upload_ckpts_to_hf.py \
        --token YOUR_HF_TOKEN \
        --repo_id username/model-name \
        --root_dir /path/to/checkpoints
"""
from huggingface_hub import create_repo, login, create_branch, HfApi
import argparse
import os
import sys

def upload_checkpoints_to_hf(token, repo_id, root_dir):
    # Use environment variable for token security
    token = os.getenv("HF_TOKEN", token)
    if not token:
        print("Error: Please set HF_TOKEN environment variable")
        sys.exit(1)
    
    try:
        api = HfApi(token=token)
        login(token=token)
        
        repo_id = repo_id
        root_dir = root_dir

        # Check if root directory exists
        if not os.path.exists(root_dir):
            print(f"Error: Root directory {root_dir} does not exist")
            return
        
        # Create repository
        print(f"Creating repository {repo_id}...")
        repo = create_repo(repo_id, private=True, repo_type="model", exist_ok=True)
        print(f"Repository created/verified: {repo}")
        
        # List all folders that start with "step-" or are named "main"
        step_folders = [f for f in os.listdir(root_dir) 
                       if os.path.isdir(os.path.join(root_dir, f))]
        
        if not step_folders:
            print("No step folders found")
            return
        
        print(f"Found folders: {step_folders}")
        
        for step in step_folders:
            step_folder_path = os.path.join(root_dir, step)
            
            try:
                if step == "main":
                    print(f"Uploading {step} to main branch...")
                    api.upload_folder(
                        folder_path=step_folder_path, 
                        repo_id=repo_id, 
                        revision="main"
                    )
                    print(f"✓ Uploaded {step_folder_path} to {repo_id} at branch main")
                else:
                    print(f"Creating branch {step} and uploading...")
                    # Create branch first
                    create_branch(repo_id, branch=step, exist_ok=True)
                    # Upload to the branch
                    api.upload_folder(
                        folder_path=step_folder_path, 
                        repo_id=repo_id, 
                        revision=step
                    )
                    print(f"✓ Uploaded {step_folder_path} to {repo_id} at branch {step}")
                    
            except Exception as e:
                print(f"✗ Error uploading {step}: {str(e)}")
                continue
        
        try:
            
            extra_files = [f for f in os.listdir(root_dir) 
                        if f.lower().endswith(('.csv', '.parquet', '.md', '.png', 'license', '.yaml'))]
            
            if extra_files:
                print(f"Found extra files: {extra_files}")
                for file in extra_files:
                    file_path = os.path.join(root_dir, file)
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=file,
                        repo_id=repo_id,
                        revision="main"
                    )
                    print(f"✓ Uploaded {file} to main branch")
            else:
                print("No extra files found in root directory")

        except Exception as e:
            print(f"✗ Error uploading extra files: {str(e)}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model checkpoints to Hugging Face Hub")
    parser.add_argument("--token", type=str, required=True, default=None, help="Hugging Face API token (or set HF_TOKEN env variable)")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID (e.g., username/repo_name)")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing step folders")
    
    args = parser.parse_args()
    
    upload_checkpoints_to_hf(args.token, args.repo_id, args.root_dir)
