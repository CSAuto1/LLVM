import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ThaiLLM API Configuration
THAILLM_API_KEY = os.getenv("THAILLM_API_KEY", "9AMZiZ6HMJj05JlwUge1miy8d4u9IOHz")
THAILLM_API_URL = "http://thaillm.or.th/api/v1/chat/completions"
DEFAULT_MODEL = "pathumma-thaillm-qwen3-8b-think-3.0.0"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Message is required"}), 400

    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty"}), 400

    # Support full conversation history if provided, otherwise single message
    messages = data.get("messages")
    if not messages:
        messages = [{"role": "user", "content": user_message}]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {THAILLM_API_KEY}"
    }

    payload = {
        "model": data.get("model", DEFAULT_MODEL),
        "messages": messages,
        "max_tokens": data.get("max_tokens", 2048),
        "temperature": data.get("temperature", 0.3)
    }

    try:
        response = requests.post(THAILLM_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        res_data = response.json()
        
        # Extract reply from OpenAI-compatible response format
        reply = ""
        if "choices" in res_data and len(res_data["choices"]) > 0:
            choice = res_data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                reply = choice["message"]["content"]
            elif "text" in choice:
                reply = choice["text"]
        else:
            reply = str(res_data)

        return jsonify({"reply": reply, "raw": res_data})

    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if hasattr(e, "response") and e.response is not None:
            try:
                error_msg = e.response.json().get("error", e.response.text)
            except Exception:
                error_msg = e.response.text
        return jsonify({"error": f"ThaiLLM API Error: {error_msg}"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
