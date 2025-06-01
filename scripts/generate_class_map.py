import os
import json

def main():
    """
    Scans data/train/ for subfolders and writes class_map.json
    with keys "0", "1", ..., mapping to folder names in sorted order.
    """
    train_dir = os.path.join("data", "train")
    if not os.path.isdir(train_dir):
        print(f"ERROR: {train_dir} does not exist. Populate data/train/ first.")
        return

    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    class_map = {str(idx): cls_name for idx, cls_name in enumerate(classes)}

    out_path = "class_map.json"
    with open(out_path, "w") as f:
        json.dump(class_map, f, indent=4)

    print(f"Generated {out_path} with {len(classes)} classes:")
    for idx, cls in class_map.items():
        print(f"  {idx} → {cls}")

if __name__ == "__main__":
    main()
