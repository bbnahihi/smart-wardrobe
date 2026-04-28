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
import time
import copy
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[*] Đang sử dụng thiết bị: {device}")

# ==========================================
# 1. ĐỌC VÀ LỌC DỮ LIỆU TỪ FILE CSV
# ==========================================
print("[*] Đang đọc file styles.csv...")
df = pd.read_csv('styles.csv', on_bad_lines='skip')

# Lọc chỉ lấy Quần áo và bỏ qua các dòng bị thiếu dữ liệu Phong cách (usage)
df_apparel = df[(df['masterCategory'] == 'Apparel') & (df['usage'].notna())].copy()

# Lấy 15 loại quần áo phổ biến nhất
top_15_cat = df_apparel['articleType'].value_counts().nlargest(15).index
df_filtered = df_apparel[df_apparel['articleType'].isin(top_15_cat)].copy()

# Giới hạn 600 ảnh mỗi loại để máy không bị quá tải
df_final = df_filtered.groupby('articleType').head(600)

# Tạo từ điển dịch Tên (Chữ) sang Số (để AI hiểu được)
cat_to_idx = {cat: i for i, cat in enumerate(df_final['articleType'].unique())}
style_to_idx = {style: i for i, style in enumerate(df_final['usage'].unique())}
idx_to_cat = {i: cat for cat, i in cat_to_idx.items()}
idx_to_style = {i: style for style, i in style_to_idx.items()}

# Lưu từ điển ra file để ai_module.py đọc lúc chạy app
torch.save({'cat': idx_to_cat, 'style': idx_to_style}, 'labels_map.pth')
print(f"[*] AI sẽ học {len(cat_to_idx)} Loại đồ và {len(style_to_idx)} Phong cách.")

# Chia tập Train và Val
train_df, val_df = train_test_split(df_final, test_size=0.2, random_state=42)

# ==========================================
# 2. XÂY DỰNG CLASS ĐỌC ẢNH TỰ ĐỘNG
# ==========================================
class FashionDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = os.path.join(self.img_dir, str(row['id']) + ".jpg")

        # Đọc ảnh (nếu file ảnh bị lỗi hoặc thiếu thì tạo 1 ảnh trắng để không bị crash code)
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        # Lấy nhãn số cho Loại đồ và Phong cách
        cat_label = cat_to_idx[row['articleType']]
        style_label = style_to_idx[row['usage']]

        return image, cat_label, style_label

# ==========================================
# 3. CHUẨN BỊ DATA LOADER VÀ MODEL KÉP
# ==========================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),             # Thu nhỏ giữ tỷ lệ
        transforms.RandomCrop(224),         # Cắt ngẫu nhiên 224x224 (Giúp AI học chống nhiễu)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),             # Thu nhỏ giữ tỷ lệ
        transforms.CenterCrop(224),         # Cắt chính giữa 224x224 (Chuẩn mực để chấm điểm)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {
    'train': FashionDataset(train_df, 'images', data_transforms['train']),
    'val': FashionDataset(val_df, 'images', data_transforms['val'])
}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'val']}

# Tạo Kiến trúc 1 Não - 2 Đầu Ra
class MultiTaskResNet(nn.Module):
    def __init__(self, num_categories, num_styles):
        super(MultiTaskResNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() # Cắt bỏ cái đầu mặc định

        # Mọc ra 2 cái đầu mới
        self.fc_category = nn.Linear(num_ftrs, num_categories)
        self.fc_style = nn.Linear(num_ftrs, num_styles)

    def forward(self, x):
        features = self.resnet(x)
        return self.fc_category(features), self.fc_style(features)

model = MultiTaskResNet(len(cat_to_idx), len(style_to_idx)).to(device)

# Đóng băng xương sống để chạy cho nhanh trên CPU
for name, param in model.named_parameters():
    if 'fc_category' not in name and 'fc_style' not in name:
        param.requires_grad = False

# Tính toán lỗi cho cả 2 nhánh
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# ==========================================
# 4. VÒNG LẶP HUẤN LUYỆN
# ==========================================
print("\n========== BẮT ĐẦU HUẤN LUYỆN ==========")
num_epochs = 15
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    for phase in ['train', 'val']:
        model.train() if phase == 'train' else model.eval()
        
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
    print()

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'my_wardrobe_multitask.pth')
print("\n[*] Đã lưu mô hình KÉP vào file 'my_wardrobe_multitask.pth'")