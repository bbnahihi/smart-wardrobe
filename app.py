# ==========================================
# FILE: app.py
# CHỨC NĂNG: Giao diện người dùng (Frontend) bằng Streamlit
# ==========================================

import streamlit as st
import tempfile
from ai_module import phan_loai_thong_minh
from logic_module import goi_y_outfit_thong_minh

# 1. Cấu hình trang web (Mở rộng toàn màn hình)
st.set_page_config(page_title="Smart Wardrobe AI", page_icon="👕", layout="wide")

# 2. Tiêu đề và Mô tả
st.title("👕 Smart Wardrobe AI - Phiên bản Tối thượng")
st.markdown("Hệ thống AI Đa nhiệm: Nhận diện **Loại trang phục** & **Phong cách**, tự động xóa phông nền và kết hợp thuật toán phối đồ thông minh.")
st.markdown("---")

# 3. Nút tải ảnh lên
uploaded_file = st.file_uploader("Tải lên hình ảnh trang phục của bạn (Ảnh tự chụp hoặc ảnh mạng đều được)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Chia giao diện làm 2 cột: Cột trái (Ảnh) tỉ lệ 1, Cột phải (Kết quả) tỉ lệ 2
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📸 Hình ảnh đầu vào")
        st.image(uploaded_file, caption="Ảnh gốc bạn tải lên", use_container_width=True)
        
    with col2:
        # Lưu file ảnh tạm thời để đưa đường dẫn cho AI đọc
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        st.subheader("⚙️ Bảng Điều Khiển")
        if st.button("✨ Phân Tích & Gợi Ý Outfit", type="primary", use_container_width=True):
            with st.spinner("AI ResNet18 đang bóc tách phông nền và trích xuất đặc trưng..."):
                
                # Gọi bộ não AI Kép (Hứng 3 biến: Loại đồ, Phong cách, Bức ảnh đã xóa nền)
                loai_do, phong_cach, anh_ai_nhin = phan_loai_thong_minh(tmp_path)
                
                # IN BỨC ẢNH MÀ AI ĐÃ XỬ LÝ RA CỘT TRÁI (TÍNH NĂNG CHECK VAR)
                with col1:
                    st.divider()
                    st.image(anh_ai_nhin, caption="👀 Ảnh thực tế AI dùng để phân tích (Đã xóa phông & Căn giữa)", use_container_width=True)
                
                st.success("Hoàn tất xử lý!")
                
                # PHẦN 1: KẾT QUẢ AI NHẬN DIỆN
                st.subheader("1. Kết quả Nhận diện AI")
                metric1, metric2 = st.columns(2)
                metric1.metric(label="🧥 Phân loại", value=loai_do)
                metric2.metric(label="🎨 Phong cách", value=phong_cach)

                st.divider()

                # PHẦN 2: LOGIC GỢI Ý PHỐI ĐỒ
                st.subheader(f"2. Đề xuất phối đồ (Chuẩn Style: {phong_cach})")
                
                # Gọi hàm từ logic_module
                ket_qua_goi_y = goi_y_outfit_thong_minh(loai_do, phong_cach)
                
                if not ket_qua_goi_y.empty:
                    st.write("Dựa trên tủ đồ hiện tại, đây là những món đồ tone-sur-tone dành cho bạn:")
                    # Hiển thị bảng kết quả đẹp mắt, ẩn cột index
                    st.dataframe(
                        ket_qua_goi_y[['icon', 'ten_mon', 'loai', 'phong_cach']],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("Rất tiếc! Tủ đồ ảo hiện tại chưa có món đồ nào phù hợp để phối cùng. Cần đi shopping thêm thôi!")