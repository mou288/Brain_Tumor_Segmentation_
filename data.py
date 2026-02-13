import kagglehub
import shutil
import os


path = kagglehub.dataset_download("atikaakter11/brain-tumor-segmentation-dataset")


current_dir = os.path.dirname(os.path.abspath(__file__))

destination = os.path.join(current_dir, "brain_tumor_dataset")

# Copy dataset to script folder
shutil.copytree(path, destination, dirs_exist_ok=True)

print("Dataset copied to:", destination)
