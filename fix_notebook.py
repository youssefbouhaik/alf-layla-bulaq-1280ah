import json

with open('Train_Bulaq_Transformer_Colab_v3.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Fix Dataset Memory Leak
        if 'Dataset.from_list(pairs)' in source:
            new_source = """import json, gzip, random
from datasets import load_dataset
import gc

data_path = "data/bulaq_sentence_pairs.jsonl.gz"
chunk_size = 4
current_src, current_tgt = [], []

def inject_ocr_noise(text):
    if random.random() < 0.2: text = text.replace('بلغني', 'بلدني')
    if random.random() < 0.2: text = text.replace('الملك', 'الماك')
    if random.random() < 0.1: text = text.replace('قالت', 'فالت')
    words = text.split()
    if len(words) > 2 and random.random() < 0.15:
        idx_w = random.randint(0, len(words) - 2)
        words[idx_w] = words[idx_w] + words[idx_w+1]
        del words[idx_w+1]
        text = ' '.join(words)
    return text

# WRITE DIRECTLY TO DISK TO SAVE 20GB OF RAM
with open('temp_dataset.jsonl', 'w', encoding='utf-8') as out_f:
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                src, tgt = row.get("corrupt_ocr", "").strip(), row.get("ground_truth", "").strip()
                if src and tgt and src != tgt:
                    current_src.append(inject_ocr_noise(src))
                    current_tgt.append(tgt)
                    if len(current_src) >= chunk_size:
                        out_f.write(json.dumps({'input_text': ' '.join(current_src), 'target_text': ' '.join(current_tgt)}) + '\\n')
                        current_src, current_tgt = [], []

del current_src, current_tgt
gc.collect()

# LOAD FROM DISK VIA HUGGINGFACE (Memory Mapped, zero RAM footprint)
raw_ds = load_dataset('json', data_files='temp_dataset.jsonl')['train']
raw_ds = raw_ds.train_test_split(test_size=0.1, seed=42)
train_ds = raw_ds["train"]
val_ds = raw_ds["test"]
print(f"\\nTrain: {len(train_ds):,} | Val: {len(val_ds):,}")
"""
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            
        # 2. Lower max length to prevent XLA explosion
        if 'max_src_len = 1024' in source:
            new_source = source.replace('1024', '512')
            cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in new_source.splitlines()]

with open('Train_Bulaq_Transformer_Colab_v4_LowRAM.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Saved as v4!")
