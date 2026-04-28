import pandas as pd

# Giả lập Database Tủ đồ ảo của người dùng (Có thêm cột 'phong_cach')
tu_do_db = pd.DataFrame([
    {"id": 1, "loai": "Jeans", "phong_cach": "Casual", "ten_mon": "Quần Jeans xanh nhạt", "icon": "👖"},
    {"id": 2, "loai": "Shirts", "phong_cach": "Formal", "ten_mon": "Áo sơ mi trắng công sở", "icon": "👔"},
    {"id": 3, "loai": "Tshirts", "phong_cach": "Casual", "ten_mon": "Áo thun basic đen", "icon": "👕"},
    {"id": 4, "loai": "Track Pants", "phong_cach": "Sports", "ten_mon": "Quần nỉ thể thao", "icon": "🩳"},
    {"id": 5, "loai": "Casual Shoes", "phong_cach": "Casual", "ten_mon": "Giày Sneaker trắng", "icon": "👟"},
    {"id": 6, "loai": "Formal Shoes", "phong_cach": "Formal", "ten_mon": "Giày da Oxford đen", "icon": "👞"},
    {"id": 7, "loai": "Trousers", "phong_cach": "Formal", "ten_mon": "Quần âu sẫm màu", "icon": "👖"},
    {"id": 8, "loai": "Sports Shoes", "phong_cach": "Sports", "ten_mon": "Giày chạy bộ", "icon": "👟"},
    {"id": 9, "loai": "Dresses", "phong_cach": "Party", "ten_mon": "Váy liền đi tiệc", "icon": "👗"},
    {"id": 10, "loai": "Heels", "phong_cach": "Party", "ten_mon": "Giày cao gót", "icon": "👠"},
])

def goi_y_outfit_thong_minh(loai_hien_tai, phong_cach_hien_tai):
    """
    Hàm gợi ý dựa trên 2 tiêu chí: Trái ngược về vị trí (Áo đi với Quần) nhưng ĐỒNG NHẤT về phong cách.
    """
    # BƯỚC 1: Lọc một nhát dứt khoát -> Chỉ giữ lại đồ cùng phong cách
    do_cung_style = tu_do_db[tu_do_db['phong_cach'] == phong_cach_hien_tai]
    
    # BƯỚC 2: Định nghĩa các nhóm đồ
    nhom_ao = ["Tshirts", "Shirts", "Sweaters", "Jackets", "Tops"]
    nhom_quan_vay = ["Jeans", "Track Pants", "Trousers", "Shorts", "Skirts"]
    nhom_giay = ["Casual Shoes", "Formal Shoes", "Sports Shoes", "Heels", "Flats"]

    goi_y = pd.DataFrame()

    # BƯỚC 3: Luật phối chéo
    if loai_hien_tai in nhom_ao:
        # Nếu đang có áo -> Gợi ý quần/váy và giày (cùng style)
        goi_y = do_cung_style[do_cung_style['loai'].isin(nhom_quan_vay + nhom_giay)]
    elif loai_hien_tai in nhom_quan_vay:
        # Nếu đang có quần -> Gợi ý áo và giày (cùng style)
        goi_y = do_cung_style[do_cung_style['loai'].isin(nhom_ao + nhom_giay)]
    elif loai_hien_tai in nhom_giay:
        # Nếu đang có giày -> Gợi ý cả bộ quần + áo (cùng style)
        goi_y = do_cung_style[do_cung_style['loai'].isin(nhom_ao + nhom_quan_vay)]
    else:
        # Nếu đồ nguyên bộ (như Dresses) hoặc không khớp luật -> Gợi ý ngẫu nhiên phụ kiện cùng style
        goi_y = do_cung_style[do_cung_style['loai'] != loai_hien_tai]

    return goi_y