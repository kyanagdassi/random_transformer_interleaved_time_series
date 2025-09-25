import os
from huggingface_hub import HfApi, HfFolder, hf_hub_download
import glob
import time

api = HfApi(token=os.getenv("HF_TOKEN"))



# Set repository name - change username to your HF username
repo_id = "sultan-daniels/leon_checkpoints_250811"

# #path to data
# data_dir = "/data/shared/ICL_Kalman_Experiments/train_and_test_data/ortho_haar"

# Path to checkpoints
checkpoints_dir = "/data/shared/ICL_Kalman_Experiments/model_checkpoints/GPT2/250811_153304.c3b82b_multi_sys_trace_linear_state_dim_5_lr_1.584893192461114e-05_num_train_sys_40000/checkpoints"


# Create the repository
api.create_repo(repo_id, 
                repo_type="model",
                private=False,
                exist_ok=True
            )
print(f"Created repository: {repo_id}")

#set the model to require access requests
api.update_repo_settings(
    repo_id=repo_id,
    gated="manual",
    repo_type="model"  # <-- Add this line to specify the repo type
)
print(f"Updated repository settings for: {repo_id}")

# # Uncomment the following lines to upload data files
# for i in range(1,2):
#     # Construct the file path
#     file_path = os.path.join(data_dir, f"train_ortho_haar_ident_C_state_dim_5.pkl")
#     if os.path.exists(file_path):
#         print(f"Uploading {file_path} to {repo_id}...")
#         try:
#             api.upload_file(
#                 path_or_fileobj=file_path,
#                 path_in_repo=f"train_ortho_haar_ident_C_state_dim_5.pkl",
#                 repo_id=repo_id,
#                 repo_type="dataset"
#             )
#             print(f"Successfully uploaded {file_path}")
#         except Exception as e:
#             print(f"Error uploading {file_path}: {e}")
#     else:
#         print(f"File {file_path} does not exist, skipping upload.")

# Uncomment the following lines to upload checkpoint files

# Get list of files already uploaded
try:
    # Fix for the rfilename attribute error
    file_list = api.list_repo_files(repo_id, repo_type="model")
    # Check if file_list contains strings or objects
    if file_list and isinstance(file_list[0], str):
        uploaded_files = set(file_list)
    else:
        uploaded_files = set(file.rfilename for file in file_list)
    print(f"Found {len(uploaded_files)} files already uploaded\n\n")
except Exception as e:
    print(f"Error listing files: {e}")
    uploaded_files = set()

# Get all checkpoint files
checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "*"))
total_files = len(checkpoint_files)
remaining_files = []
for f in checkpoint_files:
    filename = os.path.basename(f)
    # Check both the filename directly and potential path variations
    if filename not in uploaded_files and f"/{filename}" not in uploaded_files:
        remaining_files.append(f)
print(f"Total files: {total_files}, Remaining to upload: {len(remaining_files)}")

# Upload remaining files with rate limiting
for i, file_path in enumerate(remaining_files):
    filename = os.path.basename(file_path)
    print(f"Uploading {i+1}/{len(remaining_files)}: {filename}...")
    
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"Successfully uploaded {filename}")
    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            wait_time = 3600  # 1 hour in seconds
            print(f"Rate limit hit. Waiting for {wait_time//60} minutes before continuing...")
            time.sleep(wait_time)
            # Try again after waiting
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"Successfully uploaded {filename} after waiting")
            except Exception as retry_error:
                print(f"Failed to upload {filename} even after waiting: {retry_error}")
                print("Please run the script again later")
                break
        else:
            print(f"Error uploading {filename}: {e}")
            continue

print("Upload complete!")


# #download a file from the hub
# target_path = "/data/shared/ICL_Kalman_Experiments/train_and_test_data/"
# repo_id = "sultan-daniels/train_and_test_data"
# file_name = "val_ortho_haar_ident_C_state_dim_5.pkl"
# subfolder = "ortho_haar"
# file_path = hf_hub_download(
#     repo_id=repo_id,
#     filename=file_name,
#     repo_type="dataset",
#     revision="main",
#     subfolder=subfolder,
#     local_dir=target_path,
# )
# print(f"Downloaded {file_name} to {file_path}")


# api.upload_folder(
#     folder_path="/data/shared/ICL_Kalman_Experiments/train_and_test_data",
#     repo_id="sultan-daniels/train_and_test_data",
#     repo_type="dataset",
# )

# from huggingface_hub import create_repo

# # Repository details
# repo_id = "sultan-daniels/try2"
# local_repo = "../try2"  # Absolute path recommended
# # Create a new repo
# create_repo(
#     repo_id=repo_id,
#     repo_type="model",
#     private=True,
#     exist_ok=True
# )
# api.update_repo_settings(
#     repo_id=repo_id,
#     gated="manual"
# )

# # Create checkpoint directory
# checkpoint_path = os.path.join(local_repo, "checkpoint-2000", "pytorch_model.bin")
# os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
# # Create a dummy file
# with open(checkpoint_path, "wb") as f:
#     f.write(os.urandom(1024))  # Write 1KB of random data

# # Upload directly to repo
# api.upload_file(
#     path_or_fileobj=checkpoint_path,
#     path_in_repo="checkpoint-2000/pytorch_model.bin",
#     repo_id=repo_id,
#     repo_type="model"
# )

