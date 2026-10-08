import gradio as gr
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_id = "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct"

# Lazy loading so the UI opens instantly, and the model only loads when you click the button
model = None
processor = None

def load_model():
    global model, processor
    if model is None:
        print(f"Loading {model_id} onto {device}...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        processor = AutoProcessor.from_pretrained(model_id)

def process_image(image_path):
    if not image_path: return "", ""
    load_model()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "قم باستخراج النص العربي من هذه الصورة بدقة."},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    rendered_html = f"<div dir='rtl' style='font-family: Arial, sans-serif; font-size: 22px; line-height: 1.8;'>{raw_output}</div>"
    return rendered_html, raw_output

with gr.Blocks(title="Qari-OCR Local Server") as ui:
    gr.Markdown("<h1 style='text-align: center;'>📖 Qari-OCR (Local Mac Host)</h1>")
    
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="filepath", label="Upload Bulaq Page")
            btn = gr.Button("Extract Text", variant="primary")
        
        with gr.Column():
            html_out = gr.HTML(label="Rendered Layout")
            raw_out = gr.Textbox(label="Raw Output (with HTML tags)", lines=6)
            
    btn.click(process_image, inputs=[img_in], outputs=[html_out, raw_out])

ui.launch(server_name="127.0.0.1", server_port=7860)
