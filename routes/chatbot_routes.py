from flask import Blueprint, request, jsonify
from services.chatbot_service import generate_medical_reply

chatbot_bp = Blueprint("chatbot", __name__)

@chatbot_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Question is required"}), 400

    question = data["question"]

    reply = generate_medical_reply(question)

    return jsonify({
        "question": question,
        "answer": reply
    })
