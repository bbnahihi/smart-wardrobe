# ==========================================
# FILE: train_model.py (Multi-task Learning)
# CHỨC NĂNG: Dạy AI nhận diện cùng lúc Loại đồ và Phong cách
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import copy
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[*] Đang sử dụng thiết bị: {device}")

# ==========================================
# 1. ĐỌC VÀ LỌC DỮ LIỆU TỪ FILE CSV
# ==========================================
print("[*] Đang đọc file styles.csv...")
df = pd.read_csv('styles.csv', on_bad_lines='skip')
image_dir = 'anh_da_giai_nen/images'

# Bước 1: Lọc lấy nhóm Quần áo/Giày dép có nhãn phong cách
df_apparel = df[(df['masterCategory'].isin(['Apparel', 'Footwear'])) & (df['usage'].notna())].copy()

# Bước 2: Lọc đích danh 17 loại đồ mục tiêu
target_categories = [
    'Tshirts', 'Shirts', 'Top', 'Tops', 'Sweaters', 'Jackets',  # Nhóm Áo
    'Jeans', 'Trousers', 'Shorts', 'Skirts', 'Track Pants',     # Nhóm Quần/Chân váy
    'Casual Shoes', 'Formal Shoes', 'Sports Shoes', 'Heels', 'Flats', # Nhóm Giày
    'Dresses'                                                   # Nhóm Váy liền thân
]
df_filtered = df_apparel[df_apparel['articleType'].isin(target_categories)].copy()

# Bước 3: Giới hạn tối đa 600 ảnh mỗi loại (KHỞI TẠO df_final TẠI ĐÂY)
df_final = df_filtered.groupby('articleType').head(600)

# Bước 4: Kiểm tra đối chiếu file vật lý (SAU KHI ĐÃ CÓ df_final)
print("[*] Đang kiểm tra đối chiếu file vật lý...")
def check_file_exists(row):
    return os.path.exists(os.path.join(image_dir, str(row['id']) + ".jpg"))

df_final = df_final[df_final.apply(check_file_exists, axis=1)]
print(f"[*] Dữ liệu sạch sẽ hoàn toàn. Số lượng hợp lệ: {len(df_final)} ảnh")

# Bước 5: Tạo từ điển ánh xạ
cat_to_idx = {cat: i for i, cat in enumerate(df_final['articleType'].unique())}
style_to_idx = {style: i for i, style in enumerate(df_final['usage'].unique())}
idx_to_cat = {i: cat for cat, i in cat_to_idx.items()}
idx_to_style = {i: style for style, i in style_to_idx.items()}

# Lưu từ điển
torch.save({'cat': idx_to_cat, 'style': idx_to_style}, 'labels_map.pth')
print(f"[*] AI sẽ học {len(cat_to_idx)} Loại đồ và {len(style_to_idx)} Phong cách.")

# Bước 6: Chia tập Train/Val
train_df, val_df = train_test_split(
    df_final,
    test_size=0.2,
    random_state=42,
    stratify=df_final['articleType'],
)

# ==========================================
# 2. XÂY DỰNG CLASS ĐỌC ẢNH TỰ ĐỘNG (ĐỒNG BỘ INFERENCE)
# ==========================================
class FashionDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self._validate_image_paths()

    def _validate_image_paths(self):
        missing_paths = [
            os.path.join(self.img_dir, f"{image_id}.jpg")
            for image_id in self.dataframe['id']
            if not os.path.isfile(os.path.join(self.img_dir, f"{image_id}.jpg"))
        ]
        if missing_paths:
            preview = "\n".join(missing_paths[:10])
            raise FileNotFoundError(
                f"Thiếu {len(missing_paths)} ảnh trong dataset. "
                f"Các file đầu tiên:\n{preview}"
            )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = os.path.join(self.img_dir, str(row['id']) + ".jpg")

        # Đọc ảnh theo cơ chế fail-fast. Ảnh thiếu hoặc hỏng phải dừng train.
        with Image.open(img_name) as img:
            img_goc = img.convert('RGB')

        # Tạo canvas vuông trắng để bảo toàn tỷ lệ và hình dáng vật thể.
        max_size = max(img_goc.size)
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255))
        x = (max_size - img_goc.size[0]) // 2
        y = (max_size - img_goc.size[1]) // 2
        image.paste(img_goc, (x, y))

        # 3. Nạp qua bộ biến đổi PyTorch
        if self.transform:
            image = self.transform(image)

        # Lấy nhãn số
        cat_label = cat_to_idx[row['articleType']]
        style_label = style_to_idx[row['usage']]

        return image, cat_label, style_label

# ==========================================
# 3. CHUẨN BỊ DATA LOADER VÀ BỘ LỌC
# ==========================================
data_transforms = {
    'train': transforms.Compose([
        # Ảnh đã vuông sẵn, chỉ ép về 224x224 (KHÔNG dùng CenterCrop/RandomCrop nữa)
        transforms.Resize((224, 224)), 
        
        # Tăng cường dữ liệu (Augmentation) để ResNet18 khôn hơn VGG
        transforms.RandomHorizontalFlip(), # Lật ảnh ngang ngẫu nhiên
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Thay đổi độ sáng/tương phản nhẹ
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)), # Che một mảng nhỏ trên ảnh
        
        # Bộ số vàng của ImageNet (Bắt buộc phải có để Transfer Learning)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # Tập thi: Chỉ thu nhỏ và chuẩn hóa, không thêm nhiễu
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {
    'train': FashionDataset(train_df, image_dir, data_transforms['train']),
    'val': FashionDataset(val_df, image_dir, data_transforms['val'])
}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'val']}

# Tạo Kiến trúc 1 Não - 2 Đầu Ra
class MultiTaskResNet(nn.Module):
    def __init__(self, num_categories, num_styles):
        super(MultiTaskResNet, self).__init__()
        # BẮT BUỘC DÙNG DEFAULT KHI TRAIN ĐỂ CÓ KIẾN THỨC NỀN
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc_category = nn.Linear(num_ftrs, num_categories)
        self.fc_style = nn.Linear(num_ftrs, num_styles)

    def forward(self, x):
        features = self.resnet(x)
        return self.fc_category(features), self.fc_style(features)

model = MultiTaskResNet(len(cat_to_idx), len(style_to_idx)).to(device)

# 1. Đóng băng TOÀN BỘ não bộ lúc đầu
for name, param in model.named_parameters():
    param.requires_grad = False

# 2. TUYỆT CHIÊU MỞ KHÓA: Chỉ đánh thức Block cuối cùng (layer4) và 2 đầu ra
for name, param in model.named_parameters():
    if 'resnet.layer4' in name or 'fc_category' in name or 'fc_style' in name:
        param.requires_grad = True


def set_frozen_batchnorm_eval(module):
    """Giữ running statistics của các BatchNorm đã đóng băng."""
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            parameters = list(child.parameters())
            if parameters and not any(param.requires_grad for param in parameters):
                child.eval()

# In ra để kiểm tra xem đã mở khóa đúng chưa
print("\n[*] Các lớp đang được huấn luyện:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(" -", name)
        
criterion = nn.CrossEntropyLoss()
# 3. Cấu hình Optimizer: Học cực chậm (lr=0.0001) để "nắn nót" lại kiến thức
params_to_update = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(params_to_update, lr=0.0001)

# 4. Bộ giảm tốc độ học (Scheduler): Cứ sau 5 vòng, đi chậm lại 1 nửa
from torch.optim import lr_scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Đặt số vòng lặp (Khuyên dùng 15-20 vòng cho việc học sâu này)
num_epochs = 20
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            set_frozen_batchnorm_eval(model.resnet)
        else:
            model.eval()
        
        running_loss = 0.0
        running_corrects_cat = 0
        running_corrects_style = 0

        for inputs, labels_cat, labels_style in dataloaders[phase]:
            inputs, labels_cat, labels_style = inputs.to(device), labels_cat.to(device), labels_style.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                out_cat, out_style = model(inputs)
                
                # Trọng số lỗi: 1 Loại đồ + 1 Phong cách
                loss_cat = criterion(out_cat, labels_cat)
                loss_style = criterion(out_style, labels_style)
                loss = loss_cat + loss_style

                _, preds_cat = torch.max(out_cat, 1)
                _, preds_style = torch.max(out_style, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects_cat += torch.sum(preds_cat == labels_cat.data)
            running_corrects_style += torch.sum(preds_style == labels_style.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        acc_cat = running_corrects_cat.double() / len(image_datasets[phase])
        acc_style = running_corrects_style.double() / len(image_datasets[phase])
        
        # Đánh giá độ tốt bằng trung bình cộng 2 độ chính xác
        epoch_acc = (acc_cat + acc_style) / 2

        print(f'{phase.upper()} Loss: {epoch_loss:.4f} | Acc Loại đồ: {acc_cat:.4f} | Acc Phong cách: {acc_style:.4f}')

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    exp_lr_scheduler.step()
    print()

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'my_wardrobe_multitask.pth')
print("\n[*] Đã lưu mô hình KÉP vào file 'my_wardrobe_multitask.pth'")
