import json

with open('Train_Bulaq_Transformer_Colab_v3.ipynb', 'r') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    source_text = "".join(cell.get('source', []))
    if "is_tpu = False" in source_text and "import torch_xla" in source_text:
        new_source = []
        for line in cell['source']:
            new_source.append(line)
            if "import torch_xla.core.xla_model as xm" in line:
                new_source.append("    torch.xla = torch_xla # ⚡ HOTFIX FOR HF TRAINER BUG ⚡\n")
        nb['cells'][idx]['source'] = new_source

with open('Train_Bulaq_Transformer_Colab_v3.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
