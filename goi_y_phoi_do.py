import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision import models, transforms
from logic_module import get_wardrobe_items

MODEL_PATH = "my_wardrobe_multitask.pth"
LABELS_PATH = "labels_map.pth"

TOPS_LIST = ['Tshirts', 'Shirts', 'Top', 'Tops', 'Sweaters', 'Jackets']
BOTTOMS_LIST = ['Jeans', 'Trousers', 'Shorts', 'Skirts', 'Track Pants']
SHOES_LIST = ['Casual Shoes', 'Formal Shoes', 'Sports Shoes', 'Heels', 'Flats',
              'Sandals', 'Flip Flops']
DRESS_LIST = ['Dresses', 'Kurtas']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================================
# BƯỚC 1: ĐỊNH NGHĨA LẠI MODEL — Y NGUYÊN KIẾN TRÚC CỦA ai_module.py
# ============================================================================
# Class này phải giống 100% với class trong ai_module.py, vì cùng load
# 1 file my_wardrobe_multitask.pth. Khác duy nhất: forward() trả thêm
# "features" (embedding) ra ngoài, thay vì chỉ trả nhãn cuối.

class MultiTaskResNet(nn.Module):
    def __init__(self, num_categories, num_styles):
        super(MultiTaskResNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # bỏ lớp ép-về-1-nhãn, giữ vector thô
        self.fc_category = nn.Linear(num_ftrs, num_categories)
        self.fc_style = nn.Linear(num_ftrs, num_styles)

    def forward(self, x):
        features = self.resnet(x)  # ← chính là embedding 512 chiều
        cat_out = self.fc_category(features)
        style_out = self.fc_style(features)
        return cat_out, style_out, features  # trả thêm features ra ngoài


# ============================================================================
# BƯỚC 2: LOAD MODEL ĐÃ TRAIN (dùng lại đúng file .pth, không train lại)
# ============================================================================

labels_map = torch.load(LABELS_PATH, map_location=device, weights_only=False)
IDX_TO_CAT = labels_map['cat']
IDX_TO_STYLE = labels_map['style']

model = MultiTaskResNet(len(IDX_TO_CAT), len(IDX_TO_STYLE))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()  # chế độ suy luận — không học, không cập nhật trọng số

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ============================================================================
# BƯỚC 3: TRÍCH EMBEDDING TỪ 1 ẢNH
# ============================================================================

def get_embedding(image_path):
    """
    Đưa ảnh qua model, lấy vector 512 chiều ngay TRƯỚC lớp phân loại cuối.
    Đây là "đặc trưng tổng thể" mà ResNet18 đã tự học để phân biệt các loại
    quần áo — gồm cả màu, hình dáng, kiểu dáng trộn lẫn vào nhau.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError):
        return None

    # Xử lý canvas vuông giống đúng cách ai_module.py đã làm lúc train/predict
    # — bắt buộc phải làm giống, vì model "quen" nhìn ảnh theo dạng này
    max_size = max(img.size)
    square_img = Image.new("RGB", (max_size, max_size), (255, 255, 255))
    x = (max_size - img.size[0]) // 2
    y = (max_size - img.size[1]) // 2
    square_img.paste(img, (x, y))

    input_tensor = image_transforms(square_img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, features = model(input_tensor)

    return features.squeeze(0).cpu().numpy()  # tensor -> numpy, bỏ chiều batch


# ============================================================================
# BƯỚC 4: KNN THẬT TRÊN KHÔNG GIAN EMBEDDING 512 CHIỀU
# ============================================================================

def find_best_by_embedding(target_embedding, candidates_df, embedding_cache):
    """
    Bản rút gọn của find_top_k_by_embedding(): chỉ trả về 1 món duy nhất —
    món có embedding GẦN NHẤT với target_embedding (k=1, trường hợp đặc biệt
    của KNN gọi là 1-Nearest-Neighbor).

    Trả về image_path của món gần nhất, hoặc None nếu không có ứng viên nào.
    """
    if candidates_df.empty or target_embedding is None:
        return None

    paths, vectors = [], []
    for path in candidates_df['image_path']:
        if path not in embedding_cache:
            embedding_cache[path] = get_embedding(path)
        emb = embedding_cache[path]
        if emb is None:
            continue
        paths.append(path)
        vectors.append(emb)

    if not vectors:
        return None

    vectors = np.array(vectors)

    knn = NearestNeighbors(n_neighbors=1, metric="cosine")
    knn.fit(vectors)

    target_vector = target_embedding.reshape(1, -1)
    distances, indices = knn.kneighbors(target_vector)

    best_index = indices[0][0]  # chỉ có đúng 1 kết quả vì n_neighbors=1
    return paths[best_index]


# ============================================================================
# HÀM CHÍNH — thay thế suggest_outfit_hybrid() ở color_matcher.py
# ============================================================================

def suggest_outfit_by_embedding(user_chosen_item_path, embedding_cache=None):
    """
    Vẫn giữ nguyên luật CỨNG: lọc theo style trước (Casual đi với Casual...).
    Phần "chọn món nào trong cùng style" giờ dựa trên khoảng cách embedding
    512 chiều — phản ánh độ giống tổng thể do chính AI học được, không phải
    luật phối màu lý thuyết do con người viết tay.

    Chỉ trả về ĐÚNG 1 món cho mỗi vị trí (Top/Bottom/Shoes) — món có embedding
    gần nhất, dùng 1-Nearest-Neighbor (k=1).
    """
    if embedding_cache is None:
        embedding_cache = {}

    wardrobe_df = pd.DataFrame(
        get_wardrobe_items(),
        columns=['id', 'image_path', 'category', 'style'],
    )
    if wardrobe_df.empty:
        return {'Top': None, 'Bottom': None, 'Shoes': None}, None

    chosen_rows = wardrobe_df[wardrobe_df['image_path'] == user_chosen_item_path]
    if chosen_rows.empty:
        return {'Top': None, 'Bottom': None, 'Shoes': None}, None

    chosen_item = chosen_rows.iloc[0]
    target_style = chosen_item['style']
    target_cat = chosen_item['category']
    target_embedding = get_embedding(user_chosen_item_path)
    embedding_cache[user_chosen_item_path] = target_embedding

    matching_items = wardrobe_df[wardrobe_df['style'] == target_style]
    outfit = {'Top': None, 'Bottom': None, 'Shoes': None}

    def best_match(cat_list):
        subset = matching_items[matching_items['category'].isin(cat_list)]
        return find_best_by_embedding(target_embedding, subset, embedding_cache)

    if target_cat in TOPS_LIST:
        outfit['Top'] = user_chosen_item_path
        outfit['Bottom'] = best_match(BOTTOMS_LIST)
        outfit['Shoes'] = best_match(SHOES_LIST)
    elif target_cat in DRESS_LIST:
        outfit['Top'] = user_chosen_item_path
        outfit['Shoes'] = best_match(SHOES_LIST)
    elif target_cat in BOTTOMS_LIST:
        outfit['Bottom'] = user_chosen_item_path
        outfit['Top'] = best_match(TOPS_LIST)
        outfit['Shoes'] = best_match(SHOES_LIST)
    elif target_cat in SHOES_LIST:
        outfit['Shoes'] = user_chosen_item_path
        outfit['Top'] = best_match(TOPS_LIST)
        outfit['Bottom'] = best_match(BOTTOMS_LIST)
    else:
        outfit['Top'] = user_chosen_item_path

    return outfit, target_style
