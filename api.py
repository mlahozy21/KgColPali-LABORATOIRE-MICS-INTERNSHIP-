import os
import google.generativeai as genai
import torch

from template import *
# Configure the API key from an environment variable
os.environ['GOOGLE_API_KEY'] = "YOUR_GOOGLE_API_KEY"  # Replace with your Google AI Studio key
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Initialize the Gemini 2.5 Flash model
gemini = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash-lite",
    generation_config={
        "temperature": 0.2,         # Lower temperature for more focused responses
        "top_p": 0.2,              # Restricts token selection for more concise responses
        "max_output_tokens": 30,  # Limits the maximum response length
    }
)
apitemplates = {
    "cot_system": general_cot_system,
    "cot_prompt": general_cot,
    "medrag_system": general_medrag_system,
    "medrag_prompt": general_medrag,
    "kgcontext_system": general_kg_context_system,
    "kgcontext_prompt": general_kg_context,
}