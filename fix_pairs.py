import json

with open('Train_Bulaq_Transformer_Colab_v5_EMERGENCY.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'del train_ds, val_ds, raw_ds, pairs' in source:
            source = source.replace('del train_ds, val_ds, raw_ds, pairs', 'del train_ds, val_ds, raw_ds')
            cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in source.splitlines()]

with open('Train_Bulaq_Transformer_Colab_v5_EMERGENCY.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
