import json

with open('Train_Bulaq_Transformer_Colab.ipynb', 'r') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    source_text = "".join(cell.get('source', []))
    
    # Patch data loading
    if "data_path = \"data/bulaq_sentence_pairs.jsonl.gz\"" in source_text:
        new_source = """import json, gzip, random
from datasets import Dataset

data_path = "data/bulaq_sentence_pairs.jsonl.gz"
pairs = []

# --- NEW: Contextual Chunking & Noise Injection ---
chunk_size = 4  # Group sentences into paragraphs for wider context
current_src = []
current_tgt = []

def inject_ocr_noise(text):
    if random.random() < 0.2: text = text.replace('بلغني', 'بلدني')
    if random.random() < 0.2: text = text.replace('الملك', 'الماك')
    if random.random() < 0.1: text = text.replace('قالت', 'فالت')
        
    # SYNTHETIC CONGEALING (Simulate paper-saving lithograph kerning)
    words = text.split()
    if len(words) > 2 and random.random() < 0.15:
        idx_w = random.randint(0, len(words) - 2)
        words[idx_w] = words[idx_w] + words[idx_w+1]
        del words[idx_w+1]
        text = ' '.join(words)
        
    return text

with gzip.open(data_path, "rt", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            src = row.get("corrupt_ocr", "").strip()
            tgt = row.get("ground_truth", "").strip()
            
            if src and tgt and src != tgt:
                src = inject_ocr_noise(src)
                current_src.append(src)
                current_tgt.append(tgt)
                
                if len(current_src) >= chunk_size:
                    pairs.append({
                        'input_text': ' '.join(current_src),
                        'target_text': ' '.join(current_tgt)
                    })
                    current_src = []
                    current_tgt = []

print(f"✅ Loaded {len(pairs):,} paragraph-level correction pairs!")

# Show sample pairs
for i in range(2):
    print(f"\\n{i+1}. ❌ OCR:   {pairs[i]['input_text'][:80]}...")
    print(f"   ✅ Truth: {pairs[i]['target_text'][:80]}...")

# Split 90% train, 10% validation
raw_ds = Dataset.from_list(pairs).train_test_split(test_size=0.1, seed=42)
train_ds = raw_ds["train"]
val_ds = raw_ds["test"]
print(f"\\nTrain: {len(train_ds):,} | Val: {len(val_ds):,}")"""
        nb['cells'][idx]['source'] = [line + '\n' for line in new_source.split('\n')]
        nb['cells'][idx]['source'][-1] = nb['cells'][idx]['source'][-1].strip('\n')

    # Patch max length
    if "max_src_len = 256" in source_text:
        new_source = []
        for line in cell['source']:
            line = line.replace("max_src_len = 256", "max_src_len = 1024")
            line = line.replace("max_tgt_len = 256", "max_tgt_len = 1024")
            new_source.append(line)
        nb['cells'][idx]['source'] = new_source

    # Patch batch size
    if "batch_size = 16 if is_tpu else 4" in source_text:
        new_source = []
        for line in cell['source']:
            line = line.replace("batch_size = 16 if is_tpu else 4", "batch_size = 8 if is_tpu else 2")
            line = line.replace("grad_accum = 2 if is_tpu else 8", "grad_accum = 4 if is_tpu else 16")
            new_source.append(line)
        nb['cells'][idx]['source'] = new_source

with open('Train_Bulaq_Transformer_Colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
