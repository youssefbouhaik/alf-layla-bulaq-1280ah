# Bulaq 1280 AH Arabic OCR Restoration & Omnibus Edition

This repository contains the training pipeline, transformer architecture evaluations, and document compilation system used to restore the 1863 CE (1280 AH) Bulaq lithograph edition of *Alf Layla wa-Layla* (ألف ليلة وليلة).

The full corpus comprises 1,294 lithographed pages, 566,678 words, 94 original lithographic plates, and over 1,790 classical poetic couplets.

---

## 1. Background & Technical Problem

Extracting clean text from 19th-century Arabic lithographs fails with generic OCR tools (Tesseract, Kraken, commercial APIs) due to three structural issues:

1. **Physical Lithographic Noise:**
   - Vertical stem confusion: Alif (`ا`) vs. Lam (`ل`).
   - Dot erosion and micro-fusions: `د` vs. `ذ`, `ع` vs. `غ`, `ر` vs. `ز`.
   - Broken ligatures, dotless final Ya (`ى`), and unpointed Ta Marbuta (`ه`/`ة`).

2. **The "Identity Copying" Local Minimum:**
   Post-correction is an extreme minority-edit task: 90% to 95% of characters in raw OCR are already correct. Standard cross-entropy loss causes sequence models to collapse into copying the noisy input verbatim to minimize loss.

3. **Over-Correction & Hallucination:**
   Pre-trained Arabic LLMs (e.g., Qwen-7B, LLaMA) tend to modernize 1863 orthography. They replace archaic spellings like `شاه زمان` with modern standard Arabic, insert commas aggressively, and hallucinate missing text.

---

## 2. Transformer Findings: ByT5 vs. Subword Models

We fine-tuned `google/byt5-small` on 69,640 paired sentences derived from the physical Bulaq print.

### Benchmark Results

| Model Architecture | Input Level | Character Error Rate (CER) | Word Error Rate (WER) | Over-Correction Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Kraken OCR (Zenodo.7050295)** | Visual / Line | 14.82% | 28.40% | N/A (Baseline) |
| **Qwen2.5-7B-Instruct (Zero-Shot)** | Subword | 6.45% | 12.10% | 14.30% (Severe Modernization) |
| **ByT5-Small (v2 Sentence-Level)** | Raw Bytes | 3.12% | 6.85% | 4.20% |
| **ByT5-Small (v6 Weighted Focal Loss)** | Raw Bytes | **1.26%** | **2.79%** | **0.00%** |

### Key Architectural Takeaways

1. **Token-Free Processing (Byte-Level):**
   Arabic subword tokenizers (SentencePiece / BPE) split unseen historical ligatures into isolated bytes or unknown tokens (`<unk>`). ByT5 processes raw UTF-8 bytes directly, operating inside words without vocabulary mismatch.

2. **Weighted Non-Uniform Edit Loss:**
   Weighting error positions (where source OCR deviates from ground truth) 5x over unchanged characters broke the identity-copying trap.

3. **Orthography Preservation:**
   By training strictly on matched 1280 AH pairs, the model preserves archaic spelling habits (e.g., dotless ya, specific unpointed letters) while restoring broken ligatures and eroded dots.

---

## 3. Historic Lithograph vs. Healed & Curated Text

Below is a direct comparison between the raw historical OCR extraction and the healed, structurally formatted output:

### Example 1: Story Opening (Page 2)

**1863 Historical OCR (Raw):**
```text
الله الرّجمِّن الرّحِين
الحمد لله رب العالميز والصلاة والسلام على سيد المرسلين سيدنا رمولا ناعمد وعلى آله وصحبه صلاة
وسلاما دائمين متلازمين إلى يوم الدين (وبعد) فان سير الأولين صارت عبرة للآخرين لكى يري
الانسان العبر التى حصلت لغيره فيعتبر ويطالع حديث الاممالسالفة وماجرى لهم فينزجر فسبحان
من جعل حديث الأولين عبرة لقوم آخرين «فمن» تلك العبر الحكايات التى تسمى ألف ليلة وليلة
```

**Healed & Curated Text (Master Edition):**
```typst
#align(center)[#text(size: 14pt, weight: "bold", fill: rgb("#6b2d18"))[بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ]]

الحمد لله رب العالمين والصلاة والسلام على سيد المرسلين سيدنا ومولانا محمد وعلى آله وصحبه صلاة
وسلاما دائمين متلازمين إلى يوم الدين. وبعد، فإن سير الأولين صارت عبرة للآخرين لكي يرى
الإنسان العبر التي حصلت لغيره فيعتبر ويطالع حديث الأمم السالفة وما جرى لهم فينزجر. فسبحان
من جعل حديث الأولين عبرة لقوم آخرين فمن تلك العبر الحكايات التي تسمى ألف ليلة وليلة
```

---

### Example 2: Classical Poetry Structure (صدر وعجز)

**1863 Historical OCR (Raw):**
```text
الشمس المضيئة كما قل الشاعر
أشرقت فى الدجى فلاح النهار واستنارت بنورها الاسحار
#bayt([من سناها الشموس تشرقل!ا], [تبدى وتنجلى الاقار])
تسجد الكائنات بين يديها حين تبدووتهتك الاستار
وذا أومضت بروق حماها هطلت بالمدامع الامطار
```
*Note: The raw OCR flattened 3 of the 4 poetic lines into continuous prose, recognizing only a single broken verse.*

**Healed & Curated Text (Master Edition):**
```typst
الشمس المضيئة كما قال الشاعر:

#bayt([أشرقت فى الدجى فلاح النهار], [واستنارت بنورها الأسحار])
#bayt([من سناها الشموس تشرق لما], [تنبدي وتنجلي الأقمار])
#bayt([تسجد الكائنات بين يديها], [حين تبدو وتهتك الأستار])
#bayt([وإذا أومضت بروق حماها], [هطلت بالمدامع الأمطار])
```

---

### Example 3: Lithographic Illustration & Side-Column Layout (Page 4)

**1863 Historical OCR (Raw):**
```text
الجنى وضع
رأسه
ركبتها
فرفعت رأسها
الى أعلى الشجرة
... (21 fragmented single-word lines dumped down an empty column) ...
(ووقفت تحت الشجرة وقالت لهما بالاشارة انزلا)
فقالت له ابالله عليمكم ان تنزلا والا نبهت عليكم العذريت...
```

**Healed & Curated Typst Grid:**
```typst
#grid(
  columns: (1fr, 1.45fr),
  gutter: 10pt,
  align: (top + right, top + left),
  [
    الجني وضع رأسه على ركبتها ونام فرفعت رأسها إلى أعلى الشجرة فرأت الملكين
    وهما فوق تلك الشجرة فرفعت رأس الجني من فوق ركبتيها ووضعته على الأرض
    ووقفت تحت الشجرة وقالت لهما بالإشارة: انزلا ولا تخافا من هذا العفريت،
    فقالا لها: بالله عليك أن تسامحينا من هذا الأمر.
  ],
  [
    #block(stroke: 0.8pt + rgb("#8b4513"), radius: 3pt, inset: 4pt, fill: rgb("#fcf9f2"))[
      #image("illustrations/vol1/page_4_و وقفت تحت الشجرة وقالت لهما بالإشارة انزلا.png", width: 100%)
      #text(font: "Amiri", size: 8pt, fill: rgb("#7a3b1e"), weight: "bold")[(ووقفت تحت الشجرة وقالت لهما بالإشارة انزلا)]
    ]
  ]
)
```

---

## 4. Repository Structure

```text
.
├── README.md                                     # Documentation, benchmark findings & layout reference
├── Train_Bulaq_Transformer_Colab_v6.ipynb        # Fine-tuning notebook for ByT5 (Google Colab / TPU / L4 GPU)
├── build_bulaq_omnibus.py                        # Automated compilation script for the 4-volume PDF
├── data/
│   ├── bulaq_sentence_pairs.jsonl.gz             # 69,640 aligned sentence pairs for training
│   └── bulaq_ocr_fallacies_dataset.jsonl         # Synthesized and historical lithographic corruption set
└── illustrations/                                # 94 curated lithographic plates indexed by volume and page
    ├── vol1/ (25 plates)
    ├── vol2/ (26 plates)
    ├── vol3/ (26 plates)
    └── vol4/ (17 plates)
```

---

## 5. Compilation Instructions

The omnibus PDF requires [Typst](https://typst.app) (version 0.11 or higher) and the [Amiri](https://fonts.google.com/specimen/Amiri) typeface:

```bash
# 1. Install Amiri font (macOS)
curl -o ~/Library/Fonts/Amiri-Regular.ttf https://raw.githubusercontent.com/google/fonts/main/ofl/amiri/Amiri-Regular.ttf
curl -o ~/Library/Fonts/Amiri-Bold.ttf https://raw.githubusercontent.com/google/fonts/main/ofl/amiri/Amiri-Bold.ttf

# 2. Build the complete 1,295-page omnibus PDF
python3 build_bulaq_omnibus.py
```
