import json

with open('Train_Bulaq_Transformer_Colab.ipynb', 'r') as f:
    nb = json.load(f)

# Find cell 6 (Data loading) and replace with our noise injection logic
for idx, cell in enumerate(nb['cells']):
    source_text = "".join(cell.get('source', []))
    if "data_path = 'data/bulaq_ocr_fallacies_dataset.jsonl'" in source_text:
        new_data_code = """import json
import random
from datasets import Dataset

data_path = 'data/bulaq_ocr_fallacies_dataset.jsonl'
pairs = []

# --- NEW: Contextual Chunking & Noise Injection ---
chunk_size = 4  # Group sentences into paragraphs for wider context
current_src = []
current_tgt = []

def inject_ocr_noise(text):
    # Dynamically degrade input to force the model to learn formulaic correction
    if random.random() < 0.2:
        text = text.replace('بلغني', 'بلدني')
    if random.random() < 0.2:
        text = text.replace('الملك', 'الماك')
    if random.random() < 0.1:
        text = text.replace('قالت', 'فالت')
        
    # SYNTHETIC CONGEALING (Simulate paper-saving lithograph kerning)
    words = text.split()
    if len(words) > 2 and random.random() < 0.15:
        idx_w = random.randint(0, len(words) - 2)
        words[idx_w] = words[idx_w] + words[idx_w+1]
        del words[idx_w+1]
        text = ' '.join(words)
        
    return text

with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            src = row.get('corrupt_ocr', '').strip()
            tgt = row.get('ground_truth', '').strip()
            
            if src and tgt:
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

print(f'Loaded {len(pairs):,} contextual paragraph chunks!')

# Show sample pairs
for i in range(2):
    print(f"{i+1}. OCR: '{pairs[i]['input_text']}'  -->  Truth: '{pairs[i]['target_text']}'\\n")

# Split 90% train, 10% validation
raw_ds = Dataset.from_list(pairs).train_test_split(test_size=0.1, seed=42)
train_ds = raw_ds['train']
val_ds = raw_ds['test']
print(f'Train: {len(train_ds):,} paragraphs | Val: {len(val_ds):,} paragraphs')"""
        nb['cells'][idx]['source'] = [line + '\n' for line in new_data_code.split('\n')]
        nb['cells'][idx]['source'][-1] = nb['cells'][idx]['source'][-1].strip('\n')

    # Prep cell
    if "def preprocess_function" in source_text:
        new_prep_code = """max_length = 1024  # EXPANDED FOR PARAGRAPH-LEVEL CONTEXT

def preprocess_function(batch):
    model_inputs = tokenizer(batch['input_text'], max_length=max_length, padding='max_length', truncation=True)
    labels = tokenizer(text_target=batch['target_text'], max_length=max_length, padding='max_length', truncation=True)
    # Replace padding token id with -100 so it is ignored by the loss
    labels['input_ids'] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels['input_ids']
    ]
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

print('Tokenizing paragraph datasets (this may take a bit longer)...')
tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=['input_text', 'target_text'])
tokenized_val = val_ds.map(preprocess_function, batched=True, remove_columns=['input_text', 'target_text'])
print('Ready for contextual training!')"""
        nb['cells'][idx]['source'] = [line + '\n' for line in new_prep_code.split('\n')]
        nb['cells'][idx]['source'][-1] = nb['cells'][idx]['source'][-1].strip('\n')

    # Batch size reduction
    if 'Seq2SeqTrainingArguments' in source_text:
        new_source = []
        for line in cell['source']:
            if 'per_device_train_batch_size' in line:
                # Find the number and replace it
                import re
                line = re.sub(r'per_device_train_batch_size=\d+', 'per_device_train_batch_size=8', line)
            if 'per_device_eval_batch_size' in line:
                line = re.sub(r'per_device_eval_batch_size=\d+', 'per_device_eval_batch_size=8', line)
            new_source.append(line)
        nb['cells'][idx]['source'] = new_source

with open('Train_Bulaq_Transformer_Colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

