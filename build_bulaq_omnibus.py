import json
import os
import glob
import re
import subprocess
import html

OMNIBUS_TYP_PATH = "/Users/alrarsung/Working_dir_Alf_Layla/Raw_Volume_Splits/Alf_Layla_Complete_Omnibus_Pure_OCR.typ"
RAW_JSON_PATH = "/Users/alrarsung/Working_dir_Alf_Layla/alf_layla_complete_ocr_1295.json"
HEALED_JSON_PATH = "/Users/alrarsung/Downloads/alf_layla_healed_1295.json"
ILLUSTRATIONS_DIR = "/Users/alrarsung/Working_dir_Alf_Layla/illustrations"
OUTPUT_TYP = "/Users/alrarsung/Downloads/Alf_Layla_Bulaq_1280AH_Curated_Omnibus.typ"
OUTPUT_PDF = "/Users/alrarsung/Downloads/Alf_Layla_Bulaq_1280AH_Curated_Omnibus.pdf"

print("1. Loading raw and healed JSON files...")
with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
    raw_pages = json.load(f)

with open(HEALED_JSON_PATH, "r", encoding="utf-8") as f:
    healed_pages = json.load(f)

# Build line replacement map (raw prose -> healed prose)
# Also build bayt map (raw sadr/ajuz -> healed sadr/ajuz)
prose_map = {}
bayt_map = {}

def clean_healed(text):
    s = text.strip()
    # Fix opening corruption
    if "الله الرمح" in s or "الرّجمِّن" in s or ("الله" in s and "الرمحين" in s):
        return "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    if "العالميز" in s:
        s = s.replace("العالميز", "العالمين")
    if "رموانا عمد" in s:
        s = s.replace("رموانا عمد", "ومولانا محمد")
    if "رمولا ناعمد" in s:
        s = s.replace("رمولا ناعمد", "ومولانا محمد")
    return s

for rp, hp in zip(raw_pages, healed_pages):
    for ri, hi in zip(rp.get("items", []), hp.get("items", [])):
        itype = ri.get("type")
        if itype == "prose":
            r_txt = ri.get("text", "").strip()
            h_txt = clean_healed(hi.get("text", "")).strip()
            if r_txt and h_txt:
                prose_map[r_txt] = h_txt
        elif itype == "poetry_bayt":
            r_sadr = ri.get("sadr", "").strip()
            r_ajuz = ri.get("ajuz", "").strip()
            # If healed item preserved sadr/ajuz or has healed text
            h_sadr = hi.get("sadr", r_sadr).strip()
            h_ajuz = hi.get("ajuz", r_ajuz).strip()
            bayt_map[(r_sadr, r_ajuz)] = (h_sadr, h_ajuz)

print(f"✅ Mapped {len(prose_map):,} prose lines and {len(bayt_map):,} bayt pairs.")

print("2. Indexing all 94 illustrations...")
ill_by_page = {}
images = glob.glob(os.path.join(ILLUSTRATIONS_DIR, "**", "*.*"), recursive=True)
images = [f for f in images if os.path.isfile(f) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

def norm(s):
    s = re.sub(r'[\u064B-\u065F\u0670]', '', s)
    s = re.sub(r'[أإآا]', 'ا', s)
    s = re.sub(r'[ىي]', 'ي', s)
    s = re.sub(r'[ةه]', 'ه', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    return ' '.join(s.split())

for img in sorted(images):
    fn = os.path.basename(img)
    m = re.match(r'page_(\d+)(?:_\((\d+)\))?_(.*)\.(?:png|jpg|jpeg)', fn)
    if not m: continue
    fn_p = int(m.group(1))
    alt_p = m.group(2)
    caption = m.group(3).replace("_", " ").strip()
    
    # Correction for the single verified typographical offset in filename
    if fn_p == 1053 and "نور الدين و معه" in caption:
        target_page = 1063
    else:
        target_page = fn_p
        
    ill_by_page.setdefault(target_page, []).append({
        "path": os.path.abspath(img),
        "filename": fn,
        "caption": caption,
        "norm_caption": norm(caption)
    })

print(f"✅ Indexed {sum(len(v) for v in ill_by_page.values())} illustrations across {len(ill_by_page)} pages.")

print("3. Reading Omnibus master template...")
with open(OMNIBUS_TYP_PATH, "r", encoding="utf-8") as f:
    omnibus_text = f.read()

# Partition omnibus template by page markers
marker_regex = re.compile(
    r'(#v\(0\.35cm\)\s*\n'
    r'#align\(center\)\[\s*\n'
    r'\s*#line\(length: 25%, stroke: 0\.4pt \+ rgb\(\"#c4a88f\"\)\)\s*\n'
    r'\s*#text\(font: \"Amiri\", size: 8pt, fill: rgb\(\"#8b4513\"\)\)\[— صَفْحَة المَلَفّ: (\d+) —\]\s*\n'
    r'\]\s*\n'
    r'#v\(0\.2cm\))'
)

matches = list(marker_regex.finditer(omnibus_text))
print(f"Found {len(matches)} page dividers in Omnibus.")

# Helper to render an illustration block in Typst
def make_illustration_block(ill):
    cap = ill["caption"]
    img_p = ill["path"]
    return f"""
#v(0.35cm)
#align(center)[
  #block(
    stroke: 0.8pt + rgb("#8b4513"),
    radius: 3pt,
    inset: 6pt,
    fill: rgb("#fcf9f2"),
    [
      #image("{img_p}", width: 88%)
      #v(0.4em)
      #text(font: "Amiri", size: 8.5pt, fill: rgb("#7a3b1e"), weight: "bold")[#text(size: 7.5pt)[✦] {cap}]
    ]
  )
]
#v(0.35cm)
"""

# Process chunks page by page
result_chunks = []
last_end = 0

total_prose_replaced = 0
total_bayts_formatted = 0
total_illustrations_inserted = 0

for i, m in enumerate(matches):
    content = omnibus_text[last_end:m.start()]
    curr_page = 2 if i == 0 else int(matches[i-1].group(2))
    marker_full = m.group(1)
    next_page = int(m.group(2))
    
    # 1. Replace prose lines in content with ByT5 healed text
    # We process line by line to preserve all Typst macros, title boxes, and headings
    lines = content.splitlines()
    new_lines = []
    
    ills_for_this_page = list(ill_by_page.get(curr_page, []))
    inserted_ills = set()
    
    for line in lines:
        stripped = line.strip()
        
        # Check if this line is a caption of an illustration on this page
        # e.g., (ووقفت تحت الشجرة وقالت لهما بالاشارة انزلا)
        matched_ill = None
        for ill in ills_for_this_page:
            if ill["filename"] in inserted_ills:
                continue
            norm_line = norm(stripped)
            common = set(norm_line.split()).intersection(set(ill["norm_caption"].split()))
            if len(common) >= 3 or (len(common) >= 2 and len(ill["norm_caption"].split()) <= 3):
                matched_ill = ill
                break
                
        if matched_ill:
            # Replace caption line with the authentic framed illustration plate!
            new_lines.append(make_illustration_block(matched_ill))
            inserted_ills.add(matched_ill["filename"])
            total_illustrations_inserted += 1
            continue
            
        # Check if line is prose and has healed version
        if stripped in prose_map:
            healed_text = prose_map[stripped]
            # preserve original indent / newline
            new_lines.append(healed_text)
            total_prose_replaced += 1
        elif "الله الرّجمِّن الرّحِين" in stripped:
            new_lines.append("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ")
            total_prose_replaced += 1
        else:
            new_lines.append(line)
            
    # If there are any illustrations on this page that weren't an inline caption match,
    # append them right before the page marker!
    for ill in ills_for_this_page:
        if ill["filename"] not in inserted_ills:
            new_lines.append(make_illustration_block(ill))
            inserted_ills.add(ill["filename"])
            total_illustrations_inserted += 1
            
    page_processed_content = "\n".join(new_lines)
    result_chunks.append(page_processed_content)
    result_chunks.append("\n\n" + marker_full + "\n\n")
    last_end = m.end()

# Process final chunk (page 1295)
final_content = omnibus_text[last_end:]
final_lines = final_content.splitlines()
new_final_lines = []
for line in final_lines:
    stripped = line.strip()
    if stripped in prose_map:
        new_final_lines.append(prose_map[stripped])
        total_prose_replaced += 1
    else:
        new_final_lines.append(line)
result_chunks.append("\n".join(new_final_lines))

print(f"4. Processing completed:")
print(f"   - Total prose lines healed & injected: {total_prose_replaced:,}")
print(f"   - Total illustrations placed: {total_illustrations_inserted} / 94")

full_master_typ = "".join(result_chunks)

# Ensure Typst font is set to Amiri
full_master_typ = full_master_typ.replace('#set text(font: "Amiri",', '#set text(font: ("Amiri", "Geeza Pro"),')

print(f"5. Writing output master Typst to {OUTPUT_TYP}...")
with open(OUTPUT_TYP, "w", encoding="utf-8") as f:
    f.write(full_master_typ)

typ_size_mb = os.path.getsize(OUTPUT_TYP) / (1024 * 1024)
print(f"✅ Typst source written ({typ_size_mb:.2f} MB).")

print("6. Compiling master PDF with Typst...")
cmd = ["/opt/homebrew/bin/typst", "compile", "--root", "/", OUTPUT_TYP, OUTPUT_PDF]
subprocess.run(cmd, check=True)

pdf_size_mb = os.path.getsize(OUTPUT_PDF) / (1024 * 1024)
print(f"🎉 MASTER PDF COMPILED SUCCESSFULLY!")
print(f"   File: {OUTPUT_PDF}")
print(f"   Size: {pdf_size_mb:.2f} MB")
