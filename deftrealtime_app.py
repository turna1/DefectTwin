import cv2
import gradio as gr
import google.generativeai as genai
import os
import PIL.Image
import openai

# Configure the API key for Google Generative AI
genai.configure(api_key= "YOUR  GEMINI API KEY")


# Define the Generative AI model
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Function to capture frames from a video
def frame_capture(video_path, num_frames=5):
    vidObj = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    total_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = total_frames // num_frames
    while True:
        success, image = vidObj.read()
        if not success:
            break
        if count % frame_step == 0 and len(frames) < num_frames:
            frames.append(image)
        count += 1
    vidObj.release()
    return frames

# Function to generate text descriptions for frames
def generate_descriptions_for_frames(video_path):
    # Capture frames from the video
    frames = frame_capture(video_path)
    
    # Prepare images for input to the model
    images = [PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    
    # Prepare the prompt with images and instructions
    instructions = "Instructions: Consider the following frames:"
    prompt = "What is shown in each of the frames, is there any defect?"
    images_with_prompt = [prompt] + [instructions] + images
    
    # Generate content using the model
    responses = model.generate_content(images_with_prompt)
    
    # Extract and return the text descriptions from the responses
    descriptions = [response.text for response in responses]
    formatted_description = format_descriptions(descriptions)
    
    # Analyze for rail defects based on descriptions
    defect_analysis = analyze_rail_defects(formatted_description)
    
    return defect_analysis

# Function to analyze rail defects based on descriptions
def analyze_rail_defects(defect_analysis):
    question = f"Based on this: '{defect_analysis}', identify any potential rail defects or incidents, suggest specific actions, and describe vital risk management components concisely."
    return ask_rail_defect_question(question)

# Function to ask about rail defects
def ask_rail_defect_question(question, model_name='ft:gpt-3.5-turbo-0125:personal::99NsSAeQ'):
    openai.api_key = "YOUR  OPENAI API KEY"
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "The assistant is knowledgeable about any railway asset related defects and can answer questions related to them.",
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        max_tokens=1000  # Limit the response to be concise
    )
    return response.choices[0].message['content']

# Helper function to format descriptions
def format_descriptions(descriptions):
    # Join the descriptions into a single string
    formatted_description = ' '.join(descriptions)
    # Remove any leading or trailing whitespace
    formatted_description = formatted_description.strip()
    # Replace any occurrences of special characters with a space
    formatted_description = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in formatted_description)
    return formatted_description

# Define Gradio interface
video_input = gr.Video(label="Upload Video", autoplay=True)
output_text = gr.Textbox(label="Analysis of Rail Defects")

# Create Gradio app
gr.Interface(fn= generate_descriptions_for_frames, inputs=video_input, outputs=output_text, title="Rail Defect Analysis System").launch()

