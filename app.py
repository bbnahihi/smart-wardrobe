# ==========================================
# FILE: app.py
# CHỨC NĂNG: Giao diện người dùng (Frontend) bằng Streamlit
# ==========================================

import streamlit as st
import os
import sqlite3
import uuid
from pathlib import Path

from ai_module import ModelLoadError, phan_loai_thong_minh
from logic_module import (
    add_wardrobe_item,
    get_wardrobe_items,
    initialize_database,
    suggest_outfit,
)

st.set_page_config(page_title="Smart Wardrobe AI", page_icon="👕", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
CLOSET_DIR = BASE_DIR / "my_closet"
CLOSET_DIR.mkdir(exist_ok=True)
initialize_database()

# Danh mục phân nhóm để thuật toán biết đâu là Áo, Quần, Giày
# ==========================================
# 1. THUẬT TOÁN STYLIST ẢO
# ==========================================
# ==========================================
# 2. XÂY DỰNG GIAO DIỆN WEB
# ==========================================
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
                    _, file_ext = os.path.splitext(uploaded_file.name)
                    unique_filename = f"{uuid.uuid4().hex}{file_ext.lower()}"
                    save_path = CLOSET_DIR / unique_filename
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        (
                            loai_do,
                            phong_cach,
                            anh_ai_nhin,
                            do_tin_cay_loai,
                            do_tin_cay_phong_cach,
                        ) = phan_loai_thong_minh(save_path)
                        add_wardrobe_item(
                            str(save_path),
                            loai_do,
                            phong_cach,
                        )
                    except (ModelLoadError, OSError, sqlite3.Error) as exc:
                        save_path.unlink(missing_ok=True)
                        st.error(f"Không thể phân tích hoặc lưu ảnh: {exc}")
                        st.stop()

                    st.success("Đã thêm thành công vào Tủ đồ cá nhân!")
                    
                    metric1, metric2 = st.columns(2)
                    metric1.metric(label="🧥 Nhãn Loại đồ", value=f"{loai_do} - {do_tin_cay_loai:.2%}")
                    metric2.metric(label="🎨 Nhãn Phong cách", value=f"{phong_cach} - {do_tin_cay_phong_cach:.2%}")
                    
                    with st.expander("Xem ảnh AI đã xử lý"):
                        st.image(anh_ai_nhin, caption="Ảnh AI đã xử lý")

# ------------------------------------------
# TAB 2: HIỂN THỊ TỦ ĐỒ VÀ GỢI Ý
# ------------------------------------------
with tab2:
    st.header("✨ Tủ đồ của tôi")
    
    # Đọc dữ liệu mới nhất từ SQLite ở mỗi lần Streamlit rerun.
    current_wardrobe = get_wardrobe_items()
    
    if not current_wardrobe:
        st.info("Tủ đồ của bạn đang trống! Hãy sang Tab 'Thêm Đồ' để nhập dữ liệu nhé.")
    else:
        # Hiển thị Selectbox để người dùng chọn 1 món đồ làm Gốc
        st.subheader("B1: Chọn một món đồ bạn muốn mặc hôm nay")
        
        # Tạo danh sách hiển thị dễ nhìn: "Tên file - Category - Style"
        options = [item['image_path'] for item in current_wardrobe]
        items_by_path = {
            item['image_path']: item for item in current_wardrobe
        }

        def format_func(path):
            item = items_by_path[path]
            name = Path(path).name
            return f"[{item['style']}] {item['category']} - {name}"
            
        chosen_item_path = st.selectbox("Lựa chọn trang phục:", options, format_func=format_func)
        
        # Hiển thị ảnh món đồ vừa chọn
        st.image(chosen_item_path, width=150, caption="Món đồ bạn chọn")
        
        st.markdown("---")
        st.subheader("B2: Xem gợi ý phối đồ từ Stylist Ảo")
        
        if st.button("🪄 Phối đồ cho tôi!", type="primary"):
            # Gọi thuật toán gợi ý
            outfit_dict, outfit_style = suggest_outfit(chosen_item_path)

            if outfit_style is None:
                st.warning("Không tìm thấy món đồ đã chọn trong tủ đồ.")
                st.stop()
            
            st.success(f"Đã tìm thấy Set đồ chuẩn phong cách **{outfit_style}** dành cho bạn!")
            
            # Hiển thị kết quả ra 3 cột: Áo - Quần - Giày
            col_top, col_bot, col_shoe = st.columns(3)
            
            with col_top:
                st.markdown("### 👕 ÁO / VÁY")
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
