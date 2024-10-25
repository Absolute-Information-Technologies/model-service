from flask import Flask, request, jsonify
import requests
import logging
import json
import re

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

OLLAMA_URL = "http://localhost:11434/api/generate"

def post_process_json(json_data):
    # Ensure all required keys are present
    required_keys = ["name", "address", "phone", "email", "summary/objective", "skills", "certifications", "education", "experience", "languages", "social_media", "undefined"]
    for key in required_keys:
        if key not in json_data:
            json_data[key] = [] if key in ["skills", "certifications", "education", "experience", "languages", "social_media", "undefined"] else ""
    
    # Format dates
    date_pattern = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}\b'
    for exp in json_data.get("experience", []):
        for date_field in ["start_date", "end_date"]:
            if exp.get(date_field) and not re.match(date_pattern, exp[date_field]):
                exp[date_field] = ""
    
    for edu in json_data.get("education", []):
        for date_field in ["start_date", "end_date"]:
            if edu.get(date_field) and not re.match(date_pattern, edu[date_field]):
                edu[date_field] = ""
    
    # Ensure lists are actually lists
    list_fields = ["skills", "certifications", "education", "experience", "languages", "social_media", "undefined"]
    for field in list_fields:
        if not isinstance(json_data.get(field), list):
            json_data[field] = []
    
    return json_data

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        model = data.get('model', 'llama3.2:3b')
        resume_text = data.get('resume_text', '')

        if not resume_text:
            return jsonify({"error": "resume_text is required"}), 400

        prompt = f"""
Analyze the following resume content and extract information into the defined categories. Provide the output in JSON format without any additional text or explanations. Each piece of text in the resume must be assigned to the appropriate section. If the text does not clearly belong to a specific section, place it under "Undefined" and suggest the closest possible section. Pay attention to any typos and alternative spellings and attempt to correct them based on context.

Instructions:

- Ensure that all text in the resume is included in one of the sections.
- Handle common typos and spelling mistakes by matching similar words or phrases to the most appropriate section (e.g., "skilz" to "skills").
- Do **not** use ellipses ("...") or any placeholders to indicate missing or incomplete information. If any section is incomplete or unknown, return an empty string or an empty list.
- Always format the response as a valid JSON object, using empty strings or empty lists where data is missing.
- All dates must be formatted as "MMM YYYY" (e.g., "Jan 2020"). If a precise date is not available, use an empty string.
- If a section or category is not clear, classify it as "Undefined" and suggest the most likely section.
- Do not create new categories. Use only the categories specified in the JSON format below.
- For the "experience" section, include all job positions mentioned in the resume, even if some details are missing.
- In the "skills" section, list all technical skills, tools, and technologies mentioned throughout the resume.
- For "education", include all degrees and certifications mentioned, even if some details are missing.
- If social media profiles or links are mentioned (e.g., LinkedIn), include them in the "social_media" section.
- Any additional information that doesn't fit into the predefined categories should be placed in the "undefined" section with an appropriate section name suggestion.

JSON Format:
{{
    "name": "",
    "address": "",
    "phone": "",
    "email": "",
    "summary/objective": "",
    "skills": [],
    "certifications": [],
    "education": [
        {{
            "degree": "",
            "institution": "",
            "start_date": "",
            "end_date": "",
            "location": ""
        }}
    ],
    "experience": [
        {{
            "role": "",
            "company": "",
            "start_date": "",
            "end_date": "",
            "location": "",
            "responsibilities": []
        }}
    ],
    "languages": [],
    "social_media": [],
    "undefined": [
        {{
            "section": "",
            "body": []
        }}
    ]
}}

Resume content:
{resume_text}

Remember, your entire response must be only the JSON object specified above, with no additional text before or after. Ensure that all sections are filled with the provided information, and any missing data should be represented as empty strings or lists.
"""

        ollama_data = {
            "model": model,
            "prompt": prompt,
            "temperature": 0.1,  # Lower temperature for more deterministic output
            "max_tokens": 50000,  # Adjust max tokens as needed
            "top_p": 0.95,       # Adjust top_p for more focused sampling
            "stream": False
        }

        logging.info(f"Sending request to Ollama: {ollama_data}")

        response = requests.post(OLLAMA_URL, json=ollama_data, timeout=300)  # Increased timeout

        if response.status_code == 200:
            result = response.json()
            ollama_response = result.get("response", "")
            
            try:
                # Parse the string response into a JSON object
                parsed_json = json.loads(ollama_response)
                # Post-process the JSON to ensure it meets our requirements
                processed_json = post_process_json(parsed_json)
                return jsonify(processed_json)
            except json.JSONDecodeError as json_error:
                logging.error(f"Failed to parse Ollama response as JSON: {json_error}")
                logging.error(f"Raw response: {ollama_response}")
                return jsonify({"error": "Failed to parse Ollama response as JSON"}), 500
        else:
            logging.error(f"Ollama API error. Status: {response.status_code}, Content: {response.text}")
            return jsonify({"error": f"Failed to get response from Ollama. Status: {response.status_code}"}), 500

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
