# ==========================================
# FILE: app.py
# CHỨC NĂNG: Giao diện người dùng (Frontend) bằng Streamlit
# ==========================================

import streamlit as st
import pandas as pd
import os
from PIL import Image
from ai_module import phan_loai_thong_minh

# ==========================================
# 0. KHỞI TẠO HỆ THỐNG LƯU TRỮ VĨNH VIỄN
# ==========================================
# Tạo thư mục chứa ảnh quần áo thực tế của bạn (nếu chưa có)
if not os.path.exists('my_closet'):
    os.makedirs('my_closet')

# Khởi tạo hoặc đọc CSDL tủ đồ
DB_PATH = 'my_wardrobe_db.csv'
if os.path.exists(DB_PATH):
    my_wardrobe = pd.read_csv(DB_PATH)
else:
    my_wardrobe = pd.DataFrame(columns=['image_path', 'category', 'style'])
    my_wardrobe.to_csv(DB_PATH, index=False)

# Danh mục phân nhóm để thuật toán biết đâu là Áo, Quần, Giày
TOPS_LIST = ['Tshirts', 'Shirts', 'Top', 'Sweaters', 'Jackets', 'Dresses']
BOTTOMS_LIST = ['Jeans', 'Trousers', 'Shorts', 'Skirts']
SHOES_LIST = ['Casual Shoes', 'Sports Shoes', 'Heels', 'Flats']

# ==========================================
# 1. THUẬT TOÁN STYLIST ẢO
# ==========================================
def suggest_outfit(user_chosen_item_path):
    wardrobe_df = pd.read_csv(DB_PATH)
    
    # Tìm thông tin món đồ gốc
    chosen_item = wardrobe_df[wardrobe_df['image_path'] == user_chosen_item_path].iloc[0]
    target_style = chosen_item['style']
    target_cat = chosen_item['category']
    
    # Chỉ lấy những món đồ CÙNG STYLE trong tủ
    matching_items = wardrobe_df[wardrobe_df['style'] == target_style]
    
    outfit = {'Top': None, 'Bottom': None, 'Shoes': None}
    
    def get_random_item(cat_list):
        subset = matching_items[matching_items['category'].isin(cat_list)]
        if not subset.empty:
            return subset.sample(1)['image_path'].values[0]
        return None

    # Logic: Nếu chọn Áo -> đi tìm Quần + Giày
    if target_cat in TOPS_LIST:
        outfit['Top'] = user_chosen_item_path
        outfit['Bottom'] = get_random_item(BOTTOMS_LIST)
        outfit['Shoes'] = get_random_item(SHOES_LIST)
        
    # Nếu chọn Quần -> đi tìm Áo + Giày
    elif target_cat in BOTTOMS_LIST:
        outfit['Bottom'] = user_chosen_item_path
        outfit['Top'] = get_random_item(TOPS_LIST)
        outfit['Shoes'] = get_random_item(SHOES_LIST)
        
    # Nếu chọn Giày -> đi tìm Áo + Quần
    elif target_cat in SHOES_LIST:
        outfit['Shoes'] = user_chosen_item_path
        outfit['Top'] = get_random_item(TOPS_LIST)
        outfit['Bottom'] = get_random_item(BOTTOMS_LIST)
    else:
        outfit['Top'] = user_chosen_item_path # Nếu là phụ kiện thì cứ coi như áo

    return outfit, target_style

# ==========================================
# 2. XÂY DỰNG GIAO DIỆN WEB
# ==========================================
st.set_page_config(page_title="Smart Wardrobe AI", page_icon="👕", layout="wide")
st.title("👕 Smart Wardrobe AI - Trợ lý Thời trang Cá nhân")
st.markdown("Hệ thống Hybrid: AI Đa nhiệm nhận diện hình ảnh kết hợp cùng Thuật toán gợi ý Rule-based.")
st.markdown("---")

# CHIA LÀM 2 TAB GIAO DIỆN
tab1, tab2 = st.tabs(["➕ Thêm Đồ Vào Tủ (AI Tagging)", "✨ Tủ Đồ & Phối Đồ (Stylist)"])

# ------------------------------------------
# TAB 1: NHẬP KHO BẰNG AI
# ------------------------------------------
with tab1:
    st.header("📸 Nhập kho quần áo mới")
    uploaded_file = st.file_uploader("Tải lên hình ảnh món đồ thực tế của bạn", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded_file, caption="Ảnh gốc tải lên", use_container_width=True)
            
        with col2:
            st.subheader("⚙️ Xử lý bởi AI")
            if st.button("🚀 Phân Tích & Lưu Vào Tủ Đồ", type="primary"):
                with st.spinner("AI đang nhận diện và phân loại..."):
                    
                    # LƯU ẢNH VĨNH VIỄN VÀO THƯ MỤC MY_CLOSET (Không dùng temp file nữa)
                    save_path = os.path.join('my_closet', uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Gọi AI nhận diện từ file vừa lưu
                    loai_do, phong_cach, anh_ai_nhin = phan_loai_thong_minh(save_path)
                    
                    # Lưu thông tin vào CSDL CSV
                    new_row = pd.DataFrame([{'image_path': save_path, 'category': loai_do, 'style': phong_cach}])
                    my_wardrobe = pd.concat([my_wardrobe, new_row], ignore_index=True)
                    my_wardrobe.to_csv(DB_PATH, index=False)

                    st.success("Đã thêm thành công vào Tủ đồ cá nhân!")
                    
                    metric1, metric2 = st.columns(2)
                    metric1.metric(label="🧥 Nhãn Loại đồ", value=loai_do)
                    metric2.metric(label="🎨 Nhãn Phong cách", value=phong_cach)
                    
                    with st.expander("Xem ảnh AI đã bóc tách"):
                        st.image(anh_ai_nhin, caption="Ảnh AI đã xử lý")

# ------------------------------------------
# TAB 2: HIỂN THỊ TỦ ĐỒ VÀ GỢI Ý
# ------------------------------------------
with tab2:
    st.header("✨ Tủ đồ của tôi")
    
    # Đọc lại CSDL mới nhất
    current_wardrobe = pd.read_csv(DB_PATH)
    
    if current_wardrobe.empty:
        st.info("Tủ đồ của bạn đang trống! Hãy sang Tab 'Thêm Đồ' để nhập dữ liệu nhé.")
    else:
        # Hiển thị Selectbox để người dùng chọn 1 món đồ làm Gốc
        st.subheader("B1: Chọn một món đồ bạn muốn mặc hôm nay")
        
        # Tạo danh sách hiển thị dễ nhìn: "Tên file - Category - Style"
        options = current_wardrobe['image_path'].tolist()
        def format_func(path):
            row = current_wardrobe[current_wardrobe['image_path'] == path].iloc[0]
            name = path.split('\\')[-1].split('/')[-1]
            return f"[{row['style']}] {row['category']} - {name}"
            
        chosen_item_path = st.selectbox("Lựa chọn trang phục:", options, format_func=format_func)
        
        # Hiển thị ảnh món đồ vừa chọn
        st.image(chosen_item_path, width=150, caption="Món đồ bạn chọn")
        
        st.markdown("---")
        st.subheader("B2: Xem gợi ý phối đồ từ Stylist Ảo")
        
        if st.button("🪄 Phối đồ cho tôi!", type="primary"):
            # Gọi thuật toán gợi ý
            outfit_dict, outfit_style = suggest_outfit(chosen_item_path)
            
            st.success(f"Đã tìm thấy Set đồ chuẩn phong cách **{outfit_style}** dành cho bạn!")
            
            # Hiển thị kết quả ra 3 cột: Áo - Quần - Giày
            col_top, col_bot, col_shoe = st.columns(3)
            
            with col_top:
                st.markdown("### 👕 ÁO")
                if outfit_dict['Top']:
                    st.image(outfit_dict['Top'], use_container_width=True)
                else:
                    st.warning("Chưa có áo phù hợp")
                    
            with col_bot:
                st.markdown("### 👖 QUẦN / VÁY")
                if outfit_dict['Bottom']:
                    st.image(outfit_dict['Bottom'], use_container_width=True)
                else:
                    st.warning("Chưa có quần phù hợp")
                    
            with col_shoe:
                st.markdown("### 👟 GIÀY DÉP")
                if outfit_dict['Shoes']:
                    st.image(outfit_dict['Shoes'], use_container_width=True)
                else:
                    st.warning("Chưa có giày phù hợp")