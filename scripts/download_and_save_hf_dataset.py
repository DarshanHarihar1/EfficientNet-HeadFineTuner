import os
from datasets import load_dataset
from PIL import Image

ds = load_dataset("bharat-raghunathan/indian-foods-dataset")

# taking 10% of training dataset for validation
train_val = ds["train"].train_test_split(test_size=0.1, seed=42)


hf_train = train_val["train"]
hf_val   = train_val["test"]
hf_test  = ds["test"]

def make_folder_structure(root_dir, class_names):
    """
    Create root_dir/train/<class_name>/,
           root_dir/val/<class_name>/,
           root_dir/test/<class_name>/
    for each class_name.
    """
    for split in ["train", "val", "test"]:
        for cls in class_names:
            os.makedirs(os.path.join(root_dir, split, cls), exist_ok=True)

class_names = ds["train"].features["label"].names
make_folder_structure("data", class_names)

def save_split(split_dataset, split_name):
    """
    split_dataset: a Dataset object (e.g. hf_train, hf_val, hf_test)
    split_name: one of "train", "val", "test"
    """
    for idx, example in enumerate(split_dataset):
        pil_img = example["image"].convert("RGB")

        int_label = example["label"]
        cls_name = class_names[int_label]

        filename = f"{idx:05d}_{cls_name}.jpg"
        save_path = os.path.join("data", split_name, cls_name, filename)

        pil_img.save(save_path, format="JPEG")

save_split(hf_train, "train")
save_split(hf_val,   "val")
save_split(hf_test,  "test")

print("All images saved")