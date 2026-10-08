import json

with open('Train_Bulaq_Transformer_Colab_v3.ipynb', 'r') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    source_text = "".join(cell.get('source', []))
    if "gradient_checkpointing=" in source_text:
        new_source = []
        for line in cell['source']:
            if "batch_size = 2 if is_tpu else 1" in line:
                new_source.append("batch_size = 1\n")
            elif "grad_accum = 16 if is_tpu else 32" in line:
                new_source.append("grad_accum = 32\n")
            elif "gradient_checkpointing=False if is_tpu else True" in line:
                new_source.append("    gradient_checkpointing=True, # ⚡ ENABLED to save activation memory on 1024 seq_len\n")
            else:
                new_source.append(line)
        nb['cells'][idx]['source'] = new_source

with open('Train_Bulaq_Transformer_Colab_v3.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
