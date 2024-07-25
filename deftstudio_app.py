import gradio as gr
import replicate
import openai
import trimesh
import numpy as np
from PIL import Image
import requests
import io
import tempfile
import os
import base64
from huggingface_hub import InferenceClient

# Set API tokens
os.environ["REPLICATE_API_TOKEN"] = "YOUR REPLICATE TOKEN"
# Initialize the Replicate client
rep_client = replicate.Client()

# Set your OpenAI API key
OPENAI_API_KEY = "YOUR OPENAI API KEY"
openai.api_key = OPENAI_API_KEY

# Initialize the Mixtral client
client1 = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

def format_prompt(message, history):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt

# Function to generate text
def generate(prompt, history, temperature=0.2, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)
    
    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=256,
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(prompt, history)
    stream = client1.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""
    
    for response in stream:
        output += response.token.text
        yield output
    return output

# Create the chatbot interface
mychatbot = gr.Chatbot(
    avatar_images=["metarail.png", "mcr_logo.PNG"], # Ensure you have the correct paths for these images
    bubble_full_width=False, show_label=False, show_copy_button=True, likeable=True,)

# 2D Defect Simulator
predefined_defects = [
    "Missing bolts on railway track",
    "Cracks on railway track",
    "Overgrown vegetation near railway track",
    "Broken railings on railway bridge",
    "Debris on railway track",
    "Damaged railway platform"
]

# Material defects structure
material_defects = {
    "Steel": ["Rust and Corrosion", "Pitting Corrosion", "Surface Cracks", "Wear Patterns", "Spalling", "Scaling"],
    "Glass": ["Cracks", "Chips", "Scratches", "Frosting"],
    "Aluminum": ["Corrosion", "Scratches and Dents", "Anodizing Wear"],
    "Wood": ["Rot and Decay", "Cracks and Splits", "Weathering"],
    "Plastics and Polymers": ["Cracking and Crazing", "UV Degradation", "Heat Distortion"],
    "Rubber": ["Cracking", "Hardening and Brittleness", "Surface Wear"],
    "Composite Materials": ["Delamination", "Impact Damage", "Fiber Wearing"],
    "Ceramics": ["Crackling", "Chipping and Pitting", "Glaze Deterioration"]
}

# Function to ask rail defect question
def ask_rail_defect_question(question):
    openai.api_key = OPENAI_API_KEY
    structured_prompt = f"Translate the following user input into a concise, detailed visual description for a 3D model based on this input: '{question}'. Focus only on the defect’s appearance, texture qualities, and visual effects it would have on the material. Start the description directly with no extra words."
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "Provide a concise, detailed visual description of the material's defect texture, focusing on visual and tactile qualities. Do not include any additional context or introductory phrases. Imagine the textures on railway components, but describe only the texture and material."},
            {"role": "user", "content": structured_prompt}
        ],
    )
    refined_description = response.choices[0].message['content']
    return refined_description.strip()

# Function to generate images from prompts
def generate_images(prompt):
    prediction = rep_client.predictions.create(
        version="ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        input={"prompt": prompt}
    )
    prediction.wait()
    if prediction.status == "succeeded":
        image_url = prediction.output[0]
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        image = image.crop((0, 0, min(image.size), min(image.size)))  # Optionally crop the image to a square
        image.save("defect.png")
        return image
    return None

def process_railway_defect(prompt):
    try:
        prediction = rep_client.predictions.create(
            version="ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
            input={"prompt": prompt, "scheduler": "K_EULER"}
        )
        prediction.wait()
        if prediction.status == "succeeded" and prediction.output:
            image_url = prediction.output[0]
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content))
            image.save("defect.png")
            return image
        else:
            return "Failed to generate defect."
    except Exception as e:
        return f"An error occurred: {e}"

# Function to create data URL from PIL image
def image_to_data_url(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_image}"

# Converts an image to a 3D model
def image_to_3d(image):
    """
    Converts img to 3d.
    """
    image_url = image_to_data_url(image)
    try:
        prediction = rep_client.predictions.create(
            version="e0d3fe8abce3ba86497ea3530d9eae59af7b2231b6c82bedfc32b0732d35ec3a",
            input={"image_path": image_url, "do_remove_background": True}
        )
        prediction.wait()
        if prediction.status == "succeeded":
            return prediction.output
        else:
            return "Failed to convert image to 3D. Please try again."
    except Exception as e:
        return f"An error occurred: {e}"

def image_to_data_urlin(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# Function to inpaint images
def inpaint_texture(image, prompt, num_images=1):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image_data_url = image_to_data_urlin(image)
    images = []

    for _ in range(num_images):
        input_data = {
            "seed": 87870,  # Ensure a seed is provided for consistency in results
            "cfg_text": 7.5,
            "cfg_image": 1.2,
            "resolution": 512,
            "input_image": image_data_url,
            "instruction_text": prompt
        }

        prediction = rep_client.predictions.create(
            version="10e63b0e6361eb23a0374f4d9ee145824d9d09f7a31dcd70803193ebc7121430",
            input=input_data
        )
        prediction.wait()

        if prediction.status == "succeeded":
            if prediction.output:
                image_url = prediction.output
                print(f"Generated Image URL: {image_url}")  # Debugging line to check the URL
                try:
                    response = requests.get(image_url)
                    response.raise_for_status()  # Check for HTTP errors
                    img = Image.open(io.BytesIO(response.content))
                    return img  # Return a single image
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching image from URL: {e}")
                    return None
            else:
                print("Prediction succeeded but no output URL was found.")
                return None
        else:
            print(f"Prediction failed: {prediction.status}")
            return None

# Function to update defect options
def update_defect_options(selected_material):
    return gr.update(value='', choices=material_defects[selected_material])



def visualize_custom_texture(glb_file):
    # Load the original mesh
    mesh = trimesh.load(glb_file.name, force='mesh')
    custom_texture = Image.open('defect.png').convert('RGB')

    # Apply the new texture
    material = trimesh.visual.texture.SimpleMaterial(image=custom_texture)
    color_visuals = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, image=custom_texture, material=material)
    textured_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, visual=color_visuals, validate=True, process=False)

    # Save the mesh to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.glb')
    textured_mesh.export(temp_file.name, file_type='glb')
    temp_file.close()
    return temp_file.name


# Function to visualize texture based on selection criteria
def visualize_dynamic_texture():
    # Load the original mesh
    mesh = trimesh.load('train.glb', force='mesh')
    custom_texture = Image.open('defect.png').convert('RGB')

    # Apply the new texture
    material = trimesh.visual.texture.SimpleMaterial(image=custom_texture)
    color_visuals = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, image=custom_texture, material=material)
    textured_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, visual=color_visuals, validate=True, process=False)

    # Save the mesh to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.glb')
    textured_mesh.export(temp_file.name, file_type='glb')
    temp_file.close()
    return temp_file.name

def convert_to_3d(image_path="defect.png"):
    try:
        image = Image.open(image_path)
        model = image_to_3d(image)
        return model
    except Exception as e:
        return f"An error occurred: {e}"

with gr.Blocks() as app:
    with gr.Tabs():
        with gr.Tab("3D Texture Studio"):
            with gr.Tabs():
                with gr.Tab("Predefined Defect Texture"):
                    with gr.Row():
                        material_input = gr.Dropdown(choices=list(material_defects.keys()), label="Select Material")
                        defect_input = gr.Dropdown(choices=[], label="Select Defect Type")
                        generate_button = gr.Button("Generate Texture")
                    image_output = gr.Image(label="Generated Texture")
                    model_output = gr.Model3D(label="3D Model with Applied Texture")

                    material_input.change(fn=update_defect_options, inputs=[material_input], outputs=[defect_input])
                    generate_button.click(
                        fn=lambda material, defect: generate_images(ask_rail_defect_question(f"Describe the texture of {defect} on {material}")),
                        inputs=[material_input, defect_input],
                        outputs=[image_output]
                    )

                    visualize_button = gr.Button("Visualize 3D Model")
                    visualize_button.click(
                        fn=visualize_dynamic_texture,
                        inputs=[],
                        outputs=[model_output]

                    )
                    
                    

                with gr.Tab("Custom Defect Texture"):
                    with gr.Row():
                        custom_prompt_input = gr.Textbox(label="Enter Custom Prompt for Texture", placeholder="Describe any texture detail you need.")
                        refine_button = gr.Button("Refine Prompt")
                    refined_prompt_output = gr.Textbox(label="Refined Prompt", placeholder="This will show the refined prompt.")

                    with gr.Row():
                        generate_button = gr.Button("Generate Texture")
                    custom_image_output = gr.Image(label="Generated Texture")
                    model_output_custom = gr.Model3D(label="3D Model with Applied Texture")

                    # Refine the input prompt
                    refine_button.click(
                        fn=lambda prompt: ask_rail_defect_question(prompt),
                        inputs=[custom_prompt_input],
                        outputs=[refined_prompt_output]
                    )

                    # Use the refined prompt to generate the texture image
                    generate_button.click(
                        fn=lambda prompt: generate_images(prompt),
                        inputs=[custom_prompt_input],
                        outputs=[custom_image_output]
                    )

                    visualize_button_custom = gr.Button("Visualize 3D Model")
                    visualize_button_custom.click(
                        fn=visualize_dynamic_texture,
                        inputs=[],
                        outputs=[model_output_custom]
                    )

                with gr.Tab("Inpaint Defect Texture"):
                    with gr.Row():
                        image_input = gr.Image(label="Upload Image for Inpainting")
                        inpaint_prompt_input = gr.Textbox(label="Enter Prompt for Texture Inpainting")
                        inpaint_button = gr.Button("Generate Inpainted Texture")
                    inpaint_image_output = gr.Image(label="Generated Inpainted Texture")

                    inpaint_button.click(
                        fn=lambda img, prompt: inpaint_texture(img, prompt, 1),
                        inputs=[image_input, inpaint_prompt_input],
                        outputs=[inpaint_image_output]
                    )

                    visualize_button = gr.Button("Visualize 3D Model")
                    model_output = gr.Model3D(label="3D Model with Applied Texture")

                    visualize_button.click(
                        fn=visualize_dynamic_texture,
                        inputs=[],
                        outputs=[model_output]
                    )

        with gr.Tab("2D Defect Studio"):
            with gr.Tabs():
                with gr.Tab("Current Defects"):
                    with gr.Row():
                        prompt_input = gr.Dropdown(choices=predefined_defects, label="Select a prompt")
                        submit_button_dropdown = gr.Button("Generate")
                    image_output_dropdown = gr.Image(label="Generated Image")

                    def on_submit_click_dropdown(prompt):
                        image = process_railway_defect(prompt)
                        return image

                    submit_button_dropdown.click(
                        fn=on_submit_click_dropdown,
                        inputs=[prompt_input],
                        outputs=image_output_dropdown
                    )

                    convert_3d_button = gr.Button("Convert to 3D")
                    model_output_dropdown = gr.Model3D(label="3D Modele")

                    convert_3d_button.click(
                        fn=convert_to_3d,
                        inputs=[],
                        outputs=model_output_dropdown
                    )

                with gr.Tab("Custom Defect"):
                    with gr.Row():
                        custom_prompt_input = gr.Textbox(label="Custom Defect")
                        submit_button_custom = gr.Button("Generate")
                    image_output_custom = gr.Image(label="Generated Image")

                    def on_submit_click_custom(custom_prompt):
                        image = process_railway_defect(custom_prompt)
                        return image

                    submit_button_custom.click(
                        fn=on_submit_click_custom,
                        inputs=[custom_prompt_input],
                        outputs=image_output_custom
                    )

                    convert_3d_button_custom = gr.Button("Convert to 3D")
                    model_output_custom = gr.Model3D(label="3D Model")

                    convert_3d_button_custom.click(
                        fn=convert_to_3d,
                        inputs=[],
                        outputs=model_output_custom
                    )

                with gr.Tab("Inpaint Defect"):
                    with gr.Row():
                        image_input = gr.Image(label="Upload Image for Inpainting")
                        inpaint_prompt_input = gr.Textbox(label="Enter Prompt for Defect Inpainting")
                        inpaint_button = gr.Button("Generate Inpainted Defect")
                    inpaint_image_output = gr.Image(label="Generated Inpainted Defect")

                    # Use the images and prompt to generate the inpainted defect image
                    inpaint_button.click(
                        fn=lambda img, prompt: inpaint_texture(img, prompt, 1),
                        inputs=[image_input, inpaint_prompt_input],
                        outputs=[inpaint_image_output]
                    )

                    convert_3d_button_custom = gr.Button("Convert to 3D")
                    model_output_custom = gr.Model3D(label="3D Model")
              
                    convert_3d_button_custom.click(
                        fn=convert_to_3d,
                        inputs=[],
                        outputs=model_output_custom
                    )

        with gr.Tab("Chat with LLM-Inspector"):
            chat_interface = gr.ChatInterface(
                fn=generate, 
                chatbot=mychatbot,
                retry_btn=None,
                undo_btn=None
            )

app.launch(share=True)
