import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from rembg import remove

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

# Đổi bộ lọc: Đã là hình vuông sẵn thì chỉ cần thu nhỏ về 224, KHÔNG CẮT NỮA
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def phan_loai_thong_minh(image_path):
    img_goc = Image.open(image_path).convert('RGB')
    
    # 1. Cắt phông nền lấy vật thể
    img_trong_suot = remove(img_goc)
    
    # 2. Tạo khung Canvas hình vuông màu trắng (dựa trên cạnh dài nhất)
    max_size = max(img_trong_suot.size)
    anh_vuong = Image.new("RGB", (max_size, max_size), (255, 255, 255))
    
    # 3. Dán vật thể vào chính giữa Canvas trắng
    x = (max_size - img_trong_suot.size[0]) // 2
    y = (max_size - img_trong_suot.size[1]) // 2
    anh_vuong.paste(img_trong_suot, (x, y), mask=img_trong_suot.split()[3])

    # 4. Đưa vào AI nhận diện
    input_tensor = image_transforms(anh_vuong).unsqueeze(0).to(device)

    with torch.no_grad():
        cat_out, style_out = model(input_tensor)
        _, cat_pred = torch.max(cat_out, 1)
        _, style_pred = torch.max(style_out, 1)

    category = IDX_TO_CAT[cat_pred.item()]
    style = IDX_TO_STYLE[style_pred.item()]

    # Trả về thêm bức ảnh vuông để web hiển thị
    return category, style, anh_vuong