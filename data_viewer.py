import os

BASE_PATH = "C:/Users/Khwaish/Google Drive Streaming/My Drive/scm_project"
CACHE_PATH = os.path.join(BASE_PATH, "cache/datasets")
OUTPUT_PATH = os.path.join(BASE_PATH, "datasets/processed")

try:
    stage1_file = os.path.join(OUTPUT_PATH, "stage1_pretrain.jsonl")
    with open(stage1_file, 'r', encoding='utf-8') as f:
        for i, item in enumerate(iterable=f,start=4):
            if i==0:
                print(item)
            else:
                break
            
except FileNotFoundError:
    print(f"Error: File not found at '{stage1_file}'.")
    print("Please make sure the file exists and the path is correct.")
except Exception as e:
    print(f"{e}")