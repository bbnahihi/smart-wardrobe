import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
# ĐÃ XÓA: from rembg import remove

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    labels_map = torch.load('labels_map.pth', weights_only=False, map_location=device)
    IDX_TO_CAT = labels_map['cat']
    IDX_TO_STYLE = labels_map['style']
except FileNotFoundError:
    print("Lỗi: Chưa có labels_map.pth")
    IDX_TO_CAT, IDX_TO_STYLE = {}, {}

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

if IDX_TO_CAT and IDX_TO_STYLE:
    model = MultiTaskResNet(len(IDX_TO_CAT), len(IDX_TO_STYLE))
    model.load_state_dict(torch.load('my_wardrobe_multitask.pth', map_location=device))
    model = model.to(device)
    model.eval()

# Bộ lọc ảnh giữ nguyên
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def phan_loai_thong_minh(image_path):
    # 1. Đọc ảnh gốc
    img_goc = Image.open(image_path).convert('RGB')
    
    # 2. Tạo khung Canvas hình vuông màu trắng (dựa trên cạnh dài nhất)
    max_size = max(img_goc.size)
    anh_vuong = Image.new("RGB", (max_size, max_size), (255, 255, 255))
    
    # 3. DÁN TRỰC TIẾP ẢNH GỐC VÀO (Không xóa nền, giữ nguyên bối cảnh)
    x = (max_size - img_goc.size[0]) // 2
    y = (max_size - img_goc.size[1]) // 2
    anh_vuong.paste(img_goc, (x, y))

    # 4. Đưa vào AI nhận diện
    input_tensor = image_transforms(anh_vuong).unsqueeze(0).to(device)

    with torch.no_grad():
        cat_out, style_out = model(input_tensor)
        cat_probs = torch.softmax(cat_out, dim=1)
        style_probs = torch.softmax(style_out, dim=1)
        cat_confidence, cat_pred = torch.max(cat_probs, 1)
        style_confidence, style_pred = torch.max(style_probs, 1)

    category = IDX_TO_CAT[cat_pred.item()]
    style = IDX_TO_STYLE[style_pred.item()]
    category_confidence = float(cat_confidence.item())
    style_confidence = float(style_confidence.item())

    # Trả về ảnh vuông (vẫn còn phông nền gốc) để Check VAR
    return category, style, anh_vuong, category_confidence, style_confidence
