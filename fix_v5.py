import json

with open('Train_Bulaq_Transformer_Colab_v4_LowRAM_SPEED.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Drop batch size to prevent XLA CPU RAM Explosion
        if 'batch_size = 16\n' in source:
            source = source.replace('batch_size = 16\n', 'batch_size = 4\n')
            source = source.replace('grad_accum = 2\n', 'grad_accum = 8\n')
            
            # Also add eval_accumulation_steps to prevent eval RAM spikes
            if 'report_to="none",' in source:
                source = source.replace('report_to="none",', 'report_to="none",\n    eval_accumulation_steps=1,')
            
            cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in source.splitlines()]

with open('Train_Bulaq_Transformer_Colab_v5_EMERGENCY.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Saved v5!")
