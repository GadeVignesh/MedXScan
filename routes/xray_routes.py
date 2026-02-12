from flask import Blueprint, request, jsonify
from services.inference_service import predict_xray
import os

xray_bp = Blueprint("xray", __name__)

@xray_bp.route("/predict-xray", methods=["POST"])
def predict_xray_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    filename = image.filename

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    image_path = os.path.join(upload_dir, filename)
    image.save(image_path)

    result = predict_xray(image_path, filename)
    return jsonify(result)
