import os
import gc
import random
import tempfile
import torch
import devicetorch
import gradio as gr
import numpy as np
from PIL import Image

from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from dfloat11 import DFloat11Model

MAX_SEED = np.iinfo(np.int32).max

pipe = FluxKontextPipeline.from_pretrained("fuliucansheng/FLUX.1-Kontext-dev-diffusers", torch_dtype=torch.bfloat16)
DFloat11Model.from_pretrained(
    "DFloat11/FLUX.1-Kontext-dev-DF11",
    device="cpu",
    bfloat16_model=pipe.transformer,
)
pipe.enable_model_cpu_offload()

def infer(input_image, prompt, seed=42, randomize_seed=False, guidance_scale=2.5, steps=28, progress=gr.Progress(track_tqdm=True)):
    """
    Perform image editing using the FLUX.1 Kontext pipeline.
    
    This function takes an input image and a text prompt to generate a modified version
    of the image based on the provided instructions. It uses the FLUX.1 Kontext model
    for contextual image editing tasks.
    
    Args:
        input_image (PIL.Image.Image): The input image to be edited. Will be converted
            to RGB format if not already in that format.
        prompt (str): Text description of the desired edit to apply to the image.
            Examples: "Remove glasses", "Add a hat", "Change background to beach".
        seed (int, optional): Random seed for reproducible generation. Defaults to 42.
            Must be between 0 and MAX_SEED (2^31 - 1).
        randomize_seed (bool, optional): If True, generates a random seed instead of
            using the provided seed value. Defaults to False.
        guidance_scale (float, optional): Controls how closely the model follows the
            prompt. Higher values mean stronger adherence to the prompt but may reduce
            image quality. Range: 1.0-10.0. Defaults to 2.5.
        steps (int, optional): Controls how many steps to run the diffusion model for.
            Range: 1-30. Defaults to 28.
        progress (gr.Progress, optional): Gradio progress tracker for monitoring
            generation progress. Defaults to gr.Progress(track_tqdm=True).
    
    Returns:
        tuple: A 3-tuple containing:
            - PIL.Image.Image: The generated/edited image
            - int: The seed value used for generation (useful when randomize_seed=True)
            - gr.update: Gradio update object to make the reuse button visible
    
    Example:
        >>> edited_image, used_seed, button_update = infer(
        ...     input_image=my_image,
        ...     prompt="Add sunglasses",
        ...     seed=123,
        ...     randomize_seed=False,
        ...     guidance_scale=2.5
        ... )
    """
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    if input_image:
        input_image = input_image.convert("RGB")
        image = pipe(
            image=input_image, 
            prompt=prompt,
            guidance_scale=guidance_scale,
            width = input_image.size[0],
            height = input_image.size[1],
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]
    else:
        image = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]

    gradio_temp_dir = os.environ.get('GRADIO_TEMP_DIR', tempfile.gettempdir())
    temp_file_path = os.path.join(gradio_temp_dir, "image.png")
    image.save(temp_file_path, format="PNG")
    print(f"Image saved in: {temp_file_path}")

    gc.collect()
    devicetorch.empty_cache(torch)

    return image, temp_file_path, seed, gr.Button(visible=True)

def infer_example(input_image, prompt):
    image, temp_file_path, seed, _ = infer(input_image, prompt)
    gc.collect()
    devicetorch.empty_cache(torch)
    return image,temp_file_path, seed

css="""
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

/* Root variables for light mode */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --dark-gradient: linear-gradient(135deg, #2a2a72 0%, #009ffd 100%);
    
    /* Light mode colors */
    --bg-primary: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    --card-bg: rgba(255, 255, 255, 0.95);
    --text-primary: #2d3748;
    --text-secondary: #4a5568;
    --text-muted: #718096;
    --border-color: rgba(102, 126, 234, 0.2);
    --hover-bg: rgba(102, 126, 234, 0.05);
    --shadow-color: rgba(102, 126, 234, 0.15);
    --input-bg: rgba(255, 255, 255, 0.9);
    --accordion-bg: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
}

/* Dark mode colors */
.dark {
    --bg-primary: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    --card-bg: rgba(26, 32, 44, 0.95);
    --text-primary: #f7fafc;
    --text-secondary: #e2e8f0;
    --text-muted: #a0aec0;
    --border-color: rgba(102, 126, 234, 0.3);
    --hover-bg: rgba(102, 126, 234, 0.1);
    --shadow-color: rgba(0, 0, 0, 0.3);
    --input-bg: rgba(45, 55, 72, 0.9);
    --accordion-bg: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
}

/* Auto-detect system theme */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-primary: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        --card-bg: rgba(26, 32, 44, 0.95);
        --text-primary: #f7fafc;
        --text-secondary: #e2e8f0;
        --text-muted: #a0aec0;
        --border-color: rgba(102, 126, 234, 0.3);
        --hover-bg: rgba(102, 126, 234, 0.1);
        --shadow-color: rgba(0, 0, 0, 0.3);
        --input-bg: rgba(45, 55, 72, 0.9);
        --accordion-bg: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
    }
}

/* Global styles */
* {
    color: var(--text-primary) !important;
}

body {
    font-family: 'Poppins', sans-serif !important;
    background: var(--bg-primary);
    background-attachment: fixed;
    color: var(--text-primary) !important;
}

/* Animated background particles */
body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(circle at 20% 30%, rgba(255, 255, 255, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 80% 60%, rgba(255, 255, 255, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 40% 80%, rgba(255, 255, 255, 0.05) 0%, transparent 50%);
    animation: float 20s ease-in-out infinite;
    pointer-events: none;
    z-index: -1;
}

@keyframes float {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    33% { transform: translateY(-20px) rotate(1deg); }
    66% { transform: translateY(20px) rotate(-1deg); }
}

/* Main container styling */
#col-container {
    margin: 2rem auto;
    max-width: 90vw;
    padding: 2rem;
    background: var(--card-bg);
    border-radius: 20px;
    box-shadow: 0 20px 60px var(--shadow-color);
    backdrop-filter: blur(10px);
    animation: slideUp 0.6s ease-out;
    border: 1px solid var(--border-color);
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Title styling */
h1 {
    color: #667eea !important;
    text-align: center;
    font-size: 3rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    text-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
}

/* Ensure title is always visible */
h1, h1 * {
    color: #667eea !important;
    background: none !important;
    -webkit-background-clip: unset !important;
    -webkit-text-fill-color: unset !important;
    background-clip: unset !important;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.02); }
}

/* Subtitle and paragraph styling */
#col-container p {
    text-align: center;
    color: var(--text-secondary) !important;
    font-size: 1.1rem;
    margin-bottom: 1rem;
    animation: fadeIn 0.8s ease-out 0.2s both;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Image upload areas */
.input-image {
    border-radius: 15px;
    overflow: hidden;
    background: var(--input-bg);
    transition: all 0.3s ease;
    position: relative;
    border: 2px solid var(--border-color);
}

.input-image:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px var(--shadow-color);
    border-color: #667eea;
}

.input-image img {
    height: 70vh !important;
    object-fit: contain;
}

/* Row styling */
#row {
    min-height: 40vh !important;
    background: var(--input-bg);
    padding: 1rem;
    border-radius: 15px;
    border: 2px dashed var(--border-color);
    transition: all 0.3s ease;
}

#row:hover {
    border-color: #667eea;
    background: var(--hover-bg);
}

#row-height {
    height: 65px !important;
}

/* Button styling */
button {
    background: var(--primary-gradient) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 30px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3) !important;
    position: relative !important;
    overflow: hidden !important;
}

button * {
    color: white !important;
}

button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    transition: left 0.5s ease;
}

button:hover::before {
    left: 100%;
}

button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.5) !important;
}

button:active {
    transform: translateY(-1px) !important;
}

/* Run button special styling */
#run-btn {
    background: var(--secondary-gradient) !important;
    font-size: 1.1rem !important;
    padding: 15px 40px !important;
    animation: glow 2s ease-in-out infinite;
    letter-spacing: 1px;
}

@keyframes glow {
    0%, 100% { box-shadow: 0 5px 20px rgba(245, 87, 108, 0.4); }
    50% { box-shadow: 0 5px 30px rgba(245, 87, 108, 0.7); }
}

/* Text input styling */
textarea, input[type="text"] {
    border: 2px solid var(--border-color) !important;
    border-radius: 15px !important;
    padding: 15px !important;
    transition: all 0.3s ease !important;
    background: var(--input-bg) !important;
    font-size: 1rem !important;
    color: var(--text-primary) !important;
}

textarea::placeholder, input[type="text"]::placeholder {
    color: var(--text-muted) !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    transform: scale(1.02);
    outline: none !important;
}

/* Slider styling */
input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    background: transparent !important;
    height: 6px;
}

input[type="range"]::-webkit-slider-track {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    height: 6px;
    border-radius: 3px;
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    background: white;
    border: 3px solid #667eea;
    border-radius: 50%;
    cursor: pointer;
    transition: all 0.3s ease;
}

input[type="range"]::-webkit-slider-thumb:hover {
    transform: scale(1.2);
    box-shadow: 0 0 15px rgba(102, 126, 234, 0.5);
}

/* Accordion styling */
.accordion, details {
    border-radius: 15px !important;
    overflow: hidden !important;
    margin-top: 1rem !important;
    background: var(--accordion-bg) !important;
    border: 1px solid var(--border-color) !important;
    transition: all 0.3s ease !important;
}

.accordion:hover, details:hover {
    box-shadow: 0 5px 20px var(--shadow-color);
}

/* Label styling */
label, .label {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.5rem !important;
}

/* Card hover effects */
.gr-box, .gradio-container > div {
    transition: all 0.3s ease !important;
    border-radius: 15px !important;
    color: var(--text-primary) !important;
}

/* File download area */
.file-preview, .upload-container {
    background: var(--input-bg) !important;
    border-radius: 10px !important;
    padding: 10px !important;
    transition: all 0.3s ease !important;
    border: 1px solid var(--border-color) !important;
}

.file-preview:hover, .upload-container:hover {
    transform: scale(1.02);
    box-shadow: 0 5px 15px var(--shadow-color);
}

/* Checkbox styling */
input[type="checkbox"] {
    width: 20px !important;
    height: 20px !important;
    accent-color: #667eea !important;
    cursor: pointer !important;
}

/* Reuse button special effect */
#reuse-btn {
    background: var(--accent-gradient) !important;
    animation: bounce 2s ease-in-out infinite;
}

@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
}

/* Examples section styling */
.examples, .examples-holder {
    margin-top: 2rem;
    padding: 1.5rem;
    background: var(--accordion-bg);
    border-radius: 15px;
    border: 1px solid var(--border-color);
}

/* Gradio specific overrides */
.gradio-container {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.gradio-container * {
    color: var(--text-primary) !important;
}

.gradio-container .gr-button {
    color: white !important;
}

.gradio-container .gr-button * {
    color: white !important;
}

/* Fix text visibility in all components */
.gr-form, .gr-box, .gr-panel {
    background: var(--input-bg) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
}

/* Ensure all text is visible */
span, p, div, label, input, textarea, button {
    color: var(--text-primary) !important;
}

/* Override any transparent text */
.gr-textbox, .gr-textbox input, .gr-textbox textarea {
    color: var(--text-primary) !important;
    background: var(--input-bg) !important;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    #col-container {
        padding: 1rem;
        margin: 1rem;
    }
    
    h1 {
        font-size: 2rem !important;
    }
    
    button {
        padding: 10px 20px !important;
    }
}

/* Force visibility for all text elements */
* {
    color: var(--text-primary) !important;
}

/* Exception for buttons which should stay white */
button, button * {
    color: white !important;
}

/* Exception for gradient text - disabled */
/* h1, h1 * {
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
} */

/* Hide unnecessary UI elements */
.gradio-container .contain, .gradio-container .wrap {
    border: none !important;
}

/* Hide top bar/header elements */
.gradio-container > .contain:first-child {
    display: none !important;
}

.gradio-container header {
    display: none !important;
}

/* Hide any unnecessary borders/lines */
.block.gradio-html {
    border: none !important;
    background: none !important;
}

.gradio-container .prose {
    border: none !important;
}

.container {
    border: none !important;
}

/* Remove default Gradio styling that creates unwanted bars */
.gradio-container > div:first-child {
    border-bottom: none !important;
    border-top: none !important;
}
"""

with gr.Blocks(css=css, title="DFloat-Kontext") as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# 🎨 FLUX.1 Kontext [dev] ✨
        
<p style="text-align: center; font-size: 1.2rem; color: #764ba2;">
    🖼️ Transform your images with AI-powered editing magic! 🪄
</p>
<p style="text-align: center; color: #667eea;">
    Upload an image and describe your desired changes - watch the magic happen! 
</p>
        """)
        with gr.Row(equal_height=True):
            with gr.Column():
                input_image = gr.Image(label="📸 Upload Your Image", type="pil", elem_classes="input-image", elem_id="row")
                    
            with gr.Column():
                result = gr.Image(label="✨ Your Transformed Creation", show_label=True, interactive=False, elem_classes="input-image", elem_id="row")
                reuse_button = gr.Button("♻️ Reuse this image", visible=False, elem_id="reuse-btn")
        
        with gr.Row(equal_height=True):
            with gr.Column():
                prompt = gr.Text(
                    label="✏️ Your Creative Prompt",
                    show_label=True,
                    lines=3,
                    max_lines=3,
                    placeholder="✨ Describe your magical transformation... \nExamples: 'Turn the sky purple with floating crystals', 'Add fairy wings', 'Make it cyberpunk style'",
                    container=True,
                    scale=1
                )

            with gr.Column():
                download_image = gr.File(label="💾 Download Your Masterpiece", elem_id="row-height", scale=0)
                run_button = gr.Button("🚀 Transform Image!", scale=1, elem_id="run-btn")

        with gr.Row():
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                
                seed = gr.Slider(
                    label="🎲 Seed (for reproducible results)",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                
                randomize_seed = gr.Checkbox(label="🎰 Randomize seed", value=True)
                
                guidance_scale = gr.Slider(
                    label="🎯 Guidance Scale (prompt strength)",
                    minimum=1,
                    maximum=10,
                    step=0.1,
                    value=2.5,
                )       
                
                steps = gr.Slider(
                    label="🏃 Steps (quality vs speed)",
                    minimum=1,
                    maximum=40,
                    value=28,
                    step=1
                )
            
        examples = gr.Examples(
            examples=[
                ["flowers.png", "turn the flowers into sunflowers"],
                ["monster.png", "make this monster ride a skateboard on the beach"],
                ["cat.png", "make this cat happy"]
            ],
            inputs=[input_image, prompt],
            outputs=[result, download_image, seed],
            fn=infer_example,
            cache_examples=False
        )
            
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [input_image, prompt, seed, randomize_seed, guidance_scale, steps],
        outputs = [result, download_image, seed, reuse_button]
    )
    reuse_button.click(
        fn = lambda image: image,
        inputs = [result],
        outputs = [input_image]
    )

demo.launch(server_name="127.0.0.1", mcp_server=False)