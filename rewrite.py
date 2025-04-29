import os

# Class remap by dataset
class_remap = {
    "datasets/ring": 0,
    "datasets/earring": 1,
    "datasets/dress": 2
}

def rewrite_labels(dataset_path, new_class_id):
    for split in ["train", "valid"]:
        label_dir = os.path.join(dataset_path, split, "labels")
        if not os.path.exists(label_dir):
            print(f"❌ Missing {label_dir}, skipping...")
            continue

        label_paths = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".txt")]

        for path in label_paths:
            with open(path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                parts[0] = str(new_class_id)  # Replace class ID
                new_lines.append(" ".join(parts) + "\n")

            with open(path, "w") as f:
                f.writelines(new_lines)

# Apply remapping
for dataset_name, class_id in class_remap.items():
    rewrite_labels(dataset_name, class_id)
