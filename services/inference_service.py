import torch
import os
from models.xray_model import load_xray_model, CLASS_NAMES, DEVICE
from utils.image_utils import preprocess_image
from utils.gradcam import GradCAM, overlay_heatmap
from services.report_service import generate_medical_report

MODEL = load_xray_model()
TARGET_LAYER = MODEL.features[-1]
GRADCAM = GradCAM(MODEL, TARGET_LAYER)

HEATMAP_DIR = "reports"
os.makedirs(HEATMAP_DIR, exist_ok=True)

def predict_xray(image_path, filename):
    image_tensor = preprocess_image(image_path).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)

    confidence, class_idx = torch.max(probs, 1)

    # ---------- GRAD-CAM ----------
    cam = GRADCAM.generate(image_tensor, class_idx.item())
    heatmap_path = os.path.join(HEATMAP_DIR, f"heatmap_{filename}")
    overlay_heatmap(image_path, cam, heatmap_path)

    # ---------- PDF REPORT ----------
    report_path = generate_medical_report(
        image_name=filename,
        prediction=CLASS_NAMES[class_idx.item()],
        confidence=confidence.item(),
        heatmap_path=heatmap_path
    )

    return {
        "prediction": CLASS_NAMES[class_idx.item()],
        "confidence": round(confidence.item(), 4),
        "heatmap_path": heatmap_path,
        "report_path": report_path
    }
