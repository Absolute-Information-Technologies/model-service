from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json.get('prompt', '')
    
    data = {
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 1
        }
    }
    
    response = requests.post(OLLAMA_URL, json=data)
    
    if response.status_code == 200:
        result = response.json()
        return jsonify({"response": result.get("response", "")})
    else:
        return jsonify({"error": "Failed to get response from Ollama"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
