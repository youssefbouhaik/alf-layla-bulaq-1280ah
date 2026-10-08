import json

with open('Train_Bulaq_Transformer_Colab_v3.ipynb', 'r') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    source_text = "".join(cell.get('source', []))
    if "batch_size = 8 if is_tpu else 2" in source_text:
        new_source = []
        for line in cell['source']:
            line = line.replace("batch_size = 8 if is_tpu else 2", "batch_size = 2 if is_tpu else 1")
            line = line.replace("grad_accum = 4 if is_tpu else 16", "grad_accum = 16 if is_tpu else 32")
            new_source.append(line)
        nb['cells'][idx]['source'] = new_source

with open('Train_Bulaq_Transformer_Colab_v3.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
