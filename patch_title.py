import json

with open('Train_Bulaq_Transformer_Colab_v3.ipynb', 'r') as f:
    nb = json.load(f)

source = nb['cells'][0]['source']
new_source = []
for line in source:
    line = line.replace('v2 (Sentence-Level)', 'v3 (Paragraph-Level)')
    line = line.replace('sentence-level', 'paragraph-level')
    line = line.replace('v2 Fixes', 'v3 Upgrades')
    new_source.append(line)

# Add a v3 specific note
new_source.append("\n### 🌪️ v3 Architecture Upgrades\n")
new_source.append("- **Context Expansion**: `max_length` increased to 1024 for full paragraph context.\n")
new_source.append("- **Synthetic Congealing**: Random space deletion (15%) to train against lithograph kerning.\n")
new_source.append("- **Formulaic Noise Injection**: Intentional target degradation (e.g., `بلغني` -> `بلدني`) to force contextual correction.\n")

nb['cells'][0]['source'] = new_source

with open('Train_Bulaq_Transformer_Colab_v3.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
