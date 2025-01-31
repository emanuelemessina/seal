from dataset.dataset import get_all_characters

DATASET_NAME = "seal2061"

all_characters = get_all_characters()
classes = {i: all_characters[i] for i in range(len(all_characters))}

with open("yolo.yaml", "w", encoding='utf8') as f:
    f.write(f"path: {DATASET_NAME}\ntrain: train/images\nval: dev/images\ntest: test/images\n\n")
    f.write("names:\n")
    names = ""
    for i in range(len(all_characters)):
        names += f"\t{i}: {all_characters[i]}\n"
    f.write(names)

f.close()