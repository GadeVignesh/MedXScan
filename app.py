from flask import Flask, jsonify
from routes.xray_routes import xray_bp
from routes.chatbot_routes import chatbot_bp

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status": "MedXScan AI Backend Running"})

app.register_blueprint(xray_bp)
app.register_blueprint(chatbot_bp)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
