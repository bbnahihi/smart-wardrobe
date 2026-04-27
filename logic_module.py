import pandas as pd

# Giả lập database Tủ đồ hiện có bằng Pandas DataFrame
tu_do_db = pd.DataFrame([
    {"id": 1, "loai": "jeans", "mau": "xanh", "style": "casual"},
    {"id": 2, "loai": "t-shirt", "mau": "trắng", "style": "casual"},
    {"id": 3, "loai": "shirt", "mau": "đen", "style": "formal"},
    {"id": 4, "loai": "sneakers", "mau": "trắng", "style": "sport"}
])

def goi_y_outfit(item_moi):
    loai_item = item_moi['loai']
    
    # Luật phối đồ đơn giản
    if loai_item in ["t-shirt", "shirt"]:
        # Nếu input là áo -> Gợi ý quần hoặc giày có cùng style
        goi_y = tu_do_db[(tu_do_db['loai'].isin(["jeans", "shorts"])) | (tu_do_db['loai'] == "sneakers")]
    elif loai_item in ["jeans", "shorts"]:
        # Nếu input là quần -> Gợi ý áo
        goi_y = tu_do_db[tu_do_db['loai'].isin(["t-shirt", "shirt"])]
    else:
        goi_y = tu_do_db.sample(1) # Gợi ý ngẫu nhiên nếu không khớp luật
        
    return goi_y