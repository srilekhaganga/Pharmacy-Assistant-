import os
import json
import gradio as gr
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from rag_agent import run_agent  # This is your LangGraph agent
import re

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("Missing GOOGLE_API_KEY in environment variables.")

genai.configure(api_key=api_key)
vision_model = genai.GenerativeModel("gemini-1.5-flash")

# Vision prompt
vision_prompt = """
You are a pharmacy assistant AI. From this handwritten prescription image, extract structured information.

Return JSON with the following format:

{
  "medicines": [
    {
      "name": "Paracetamol",
      "dosage": "500mg",
      "frequency": "2" ( For instance, 1(Morning)-0(Noon)-1(Night) means 2 times a day ),
      "duration": "5 days",
      "required_quantity": 10 (fequency * duration) ,
      "Availability": Yes (Yes/No)
    }
  ]
}

Don't explain anything. Return only valid JSON. Use tools to check the availabiltity of drug, quantity required.
"""

def process_prescription(image: Image.Image):
    try:
        # Gemini Vision extraction
        response = vision_model.generate_content([vision_prompt, image])
        raw_text = response.text.strip()

        # Clean Markdown-style formatting
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)

       # Parse JSON
        try:
            parsed_json = json.loads(raw_text)
            print(parsed_json)

        except json.JSONDecodeError as e:
            return f"[Vision JSON Parse Error] {e}\n\n{raw_text}"

        # Run LangGraph agent on parsed prescription
        final_output = run_agent(parsed_json)
        #final_output = run_agent({"parsed": parsed_json})
        return json.dumps(final_output, indent=2)


    except Exception as e:
        return f"[Pipeline Error] {str(e)}"

# Gradio Interface
app = gr.Interface(
    fn=process_prescription,
    inputs=gr.Image(type="pil", label="Upload Prescription Image"),
    outputs=gr.Textbox(label="Assistant Output"),
    title="Pharmacy Assistant (Vision + LangGraph)",
    description="Upload a handwritten prescription. Gemini Vision will check prescription availability"
)

if __name__ == "__main__":
    app.launch(share=True)
