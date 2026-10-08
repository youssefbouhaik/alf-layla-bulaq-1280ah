import json

with open('Train_Bulaq_Transformer_Colab.ipynb', 'r') as f:
    nb = json.load(f)

# Find cell 6 and update the inject_ocr_noise function
for idx, cell in enumerate(nb['cells']):
    source_text = "".join(cell.get('source', []))
    if 'def inject_ocr_noise' in source_text:
        new_source = []
        for line in cell['source']:
            new_source.append(line)
            if "text = text.replace('قالت', 'فالت')" in line:
                # Add the congealing logic right after this
                new_source.append("\n    # SYNTHETIC CONGEALING (Simulate paper-saving lithograph kerning)\n")
                new_source.append("    words = text.split()\n")
                new_source.append("    if len(words) > 2 and random.random() < 0.15:\n")
                new_source.append("        idx = random.randint(0, len(words) - 2)\n")
                new_source.append("        words[idx] = words[idx] + words[idx+1]\n")
                new_source.append("        del words[idx+1]\n")
                new_source.append("        text = ' '.join(words)\n")
        nb['cells'][idx]['source'] = new_source
        break

with open('Train_Bulaq_Transformer_Colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
