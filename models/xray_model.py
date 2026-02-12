import torch
import torchvision.models as models
from torchvision.models import DenseNet121_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example disease labels (update as per your dataset)
CLASS_NAMES = [
    "Normal",
    "Pneumonia",
    "Covid-19",
    "Tuberculosis"
]

def load_xray_model():
    weights = DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)

    # Replace classifier for our number of classes
    model.classifier = torch.nn.Linear(
        model.classifier.in_features,
        len(CLASS_NAMES)
    )

    model.to(DEVICE)
    model.eval()

    return model
