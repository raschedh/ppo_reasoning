from datasets import load_from_disk

dataset = load_from_disk('data_hf/countdown_dataset')["train"]
print(dataset)