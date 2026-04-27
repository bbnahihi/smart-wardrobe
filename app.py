import streamlit as st
import tempfile
from ai_module import phan_loai_quan_ao, lay_mau_chu_dao
from logic_module import goi_y_outfit

st.title("👕 Smart Wardrobe AI")
st.write("Upload một món đồ để AI phân loại và gợi ý cách phối!")

# Tạo nút upload file
uploaded_file = st.file_uploader("Chọn ảnh quần áo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Hiển thị ảnh đã tải lên
    st.image(uploaded_file, caption="Ảnh bạn vừa tải lên", width=300)
    
    # Lưu file tạm thời để AI đọc
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if st.button("Phân tích bằng AI"):
        with st.spinner("AI đang nhìn..."):
            # Gọi hàm từ các module
            loai_do = phan_loai_quan_ao(tmp_path)
            mau_sac_rgb = lay_mau_chu_dao(tmp_path)
            
            st.success("Phân tích hoàn tất!")
            
            # Hiển thị kết quả Phân loại
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Kết quả AI:")
                st.write(f"**Loại đồ:** {loai_do.capitalize()}")
                st.write(f"**Mã màu RGB:** {mau_sac_rgb}")
            
            # Hiển thị kết quả Gợi ý
            with col2:
                st.subheader("Gợi ý phối đồ:")
                item_hien_tai = {"loai": loai_do}
                ket_qua_goi_y = goi_y_outfit(item_hien_tai)
                st.dataframe(ket_qua_goi_y)