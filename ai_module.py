from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
LABELS_PATH = BASE_DIR / "labels_map.pth"
MODEL_PATH = BASE_DIR / "my_wardrobe_multitask.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelLoadError(RuntimeError):
    pass


class MultiTaskResNet(nn.Module):
    def __init__(self, num_categories, num_styles):
        super(MultiTaskResNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc_category = nn.Linear(num_ftrs, num_categories)
        self.fc_style = nn.Linear(num_ftrs, num_styles)

    def forward(self, x):
        features = self.resnet(x)
        return self.fc_category(features), self.fc_style(features)


@st.cache_resource(show_spinner="Đang tải mô hình AI...")
def load_model():
    missing_files = [
        str(path.name) for path in (LABELS_PATH, MODEL_PATH) if not path.is_file()
    ]
    if missing_files:
        raise ModelLoadError(
            "Không tìm thấy artifact AI: " + ", ".join(missing_files)
        )

    try:
        labels_map = torch.load(
            LABELS_PATH,
            weights_only=True,
            map_location="cpu",
        )
        idx_to_cat = labels_map["cat"]
        idx_to_style = labels_map["style"]

        model = MultiTaskResNet(len(idx_to_cat), len(idx_to_style))
        state_dict = torch.load(
            MODEL_PATH,
            weights_only=True,
            map_location=DEVICE,
        )
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        model.eval()
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ModelLoadError(
            f"Không thể tải mô hình từ file .pth: {exc}"
        ) from exc

    return model, idx_to_cat, idx_to_style


image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def phan_loai_thong_minh(image_path):
    model, idx_to_cat, idx_to_style = load_model()

    with Image.open(image_path) as image:
        img_goc = image.convert('RGB')

    # Canvas vuông trắng giữ nguyên tỷ lệ của ảnh gốc.
    max_size = max(img_goc.size)
    anh_vuong = Image.new("RGB", (max_size, max_size), (255, 255, 255))
    x = (max_size - img_goc.size[0]) // 2
    y = (max_size - img_goc.size[1]) // 2
    anh_vuong.paste(img_goc, (x, y))

    input_tensor = image_transforms(anh_vuong).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        cat_out, style_out = model(input_tensor)
        cat_probs = torch.softmax(cat_out, dim=1)
        style_probs = torch.softmax(style_out, dim=1)
        cat_confidence, cat_pred = torch.max(cat_probs, 1)
        style_confidence, style_pred = torch.max(style_probs, 1)

    category = idx_to_cat[cat_pred.item()]
    style = idx_to_style[style_pred.item()]

    return (
        category,
        style,
        anh_vuong,
        float(cat_confidence.item()),
        float(style_confidence.item()),
    )
