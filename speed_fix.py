import json

with open('Train_Bulaq_Transformer_Colab_v4_LowRAM.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Optimize TPU Batching for pure speed (Batch size 1 chokes the TPU)
        if 'batch_size = 1\n' in source:
            source = source.replace('batch_size = 1\n', 'batch_size = 16\n')
            source = source.replace('grad_accum = 32\n', 'grad_accum = 2\n')
            # 2. Lower to 2 epochs to guarantee it finishes under 25 mins
            source = source.replace('num_train_epochs=3,', 'num_train_epochs=2,')
            
            cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in source.splitlines()]

with open('Train_Bulaq_Transformer_Colab_v4_LowRAM_SPEED.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Saved as SPEED version!")
