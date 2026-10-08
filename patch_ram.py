import json

with open('Train_Bulaq_Transformer_Colab_v3.ipynb', 'r') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    source_text = "".join(cell.get('source', []))
    
    # Patch Cell 5 (Tokenization) to delete raw data and run garbage collection
    if "tokenized_val = val_ds.map(preprocess" in source_text:
        new_source = []
        for line in cell['source']:
            new_source.append(line)
            if "print(f'✅ Tokenized:" in line:
                new_source.insert(-1, "import gc\n")
                new_source.insert(-1, "del train_ds, val_ds, raw_ds, pairs\n")
                new_source.insert(-1, "gc.collect() # Free up CPU RAM before training!\n")
        nb['cells'][idx]['source'] = new_source

    # Patch Cell 6 (Training Args) to disable pin_memory
    if "training_args = Seq2SeqTrainingArguments(" in source_text:
        new_source = []
        for line in cell['source']:
            new_source.append(line)
            if "gradient_checkpointing=True" in line:
                new_source.append("    dataloader_pin_memory=False, # ⚡ Stop hoarding CPU RAM\n")
                new_source.append("    dataloader_num_workers=0,    # ⚡ Stop multiprocessing RAM spikes\n")
        nb['cells'][idx]['source'] = new_source

with open('Train_Bulaq_Transformer_Colab_v3.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
