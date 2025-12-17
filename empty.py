import os
import shutil

def empty_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Remove all contents of the directory
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove file or symlink
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directory
    else:
        print(f"Directory {directory_path} does not exist.")

