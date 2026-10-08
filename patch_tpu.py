import json

with open('Train_Bulaq_Transformer_Colab_v3.ipynb', 'r') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    source_text = "".join(cell.get('source', []))
    if "model = model.to(device)" in source_text:
        new_source = []
        for line in cell['source']:
            if "model = model.to(device)" in line:
                new_source.append("# Removed manual model.to(device) - Hugging Face Trainer handles this for TPU!\n")
            else:
                new_source.append(line)
        nb['cells'][idx]['source'] = new_source

with open('Train_Bulaq_Transformer_Colab_v3.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
