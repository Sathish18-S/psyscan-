from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key='AIzaSyD9ahj9-kV-ZH9z4qVoBKejMYsJXW9c2MY')

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config)

@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.json.get('input')
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        response = model.generate_content(user_input)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)