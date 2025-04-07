import os
from dotenv import load_dotenv
from google import generativeai 
from google.generativeai import types

foodWaste_Frequency = {
    "Plate Waste" : 50,
    "Kitchen Prep Waste": 35,
    "Spoilage Waste" : 10
}

load_dotenv()

def generate():
    generativeai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    model = generativeai.GenerativeModel("gemini-2.0-flash")

    foodWaste_Frequency = {
    "Plate Waste" : 50,
    "Kitchen Prep Waste": 35,
    "Spoilage Waste" : 10
    }

    prompt = f"""
    {foodWaste_Frequency}

    the dictionary above is data about food waste where the key is the food category and the value is the frequency

    Can you give possible reasons why a certain food category has higher waste frequency and suggest possible solutions
    only do so for the highest frequency Do so in a short and concise way without any unnecessary wordings
    """

    response = model.generate_content(prompt)
    print(response.text)

if __name__ == "__main__":
    generate()
