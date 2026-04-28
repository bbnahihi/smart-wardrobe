import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

print("1. Đang đọc file CSV...")
df = pd.read_csv('styles.csv', on_bad_lines='skip')

print("2. Đang lọc dữ liệu (Chỉ lấy Quần áo, loại bỏ Phụ kiện)...")
# Chỉ giữ lại đồ Apparel (Quần áo)
df_apparel = df[df['masterCategory'] == 'Apparel']

# Lấy 15 loại quần áo phổ biến nhất để AI học (bạn có thể đổi số 15 này)
top_15_loai = df_apparel['articleType'].value_counts().nlargest(15).index
df_filtered = df_apparel[df_apparel['articleType'].isin(top_15_loai)]

# Lấy tối đa 600 ảnh cho mỗi loại để laptop cá nhân train không bị treo
df_final = df_filtered.groupby('articleType').head(600)
print(f"-> Tổng số ảnh AI sẽ học: {len(df_final)} ảnh")

print("3. Chia tập Train (80%) và Val (20%)...")
train_df, val_df = train_test_split(df_final, test_size=0.2, random_state=42, stratify=df_final['articleType'])

def tao_thu_muc_va_copy(dataframe, folder_name):
    base_dir = f"./dataset_phuc_tap/{folder_name}"
    
    # Lặp qua từng dòng trong Excel
    for _, row in dataframe.iterrows():
        img_id = str(row['id']) + ".jpg"
        
        # Đổi tên nhãn cho chuẩn (VD: 'Track Pants' thành 'Track_Pants')
        nhan_ai = str(row['articleType']).replace(" ", "_") 
        
        target_dir = os.path.join(base_dir, nhan_ai)
        os.makedirs(target_dir, exist_ok=True)
        
        src_path = os.path.join("images", img_id)
        dst_path = os.path.join(target_dir, img_id)
        
        # Copy ảnh nếu file tồn tại
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

print("4. Đang copy ảnh vào thư mục Train...")
tao_thu_muc_va_copy(train_df, "train")

print("5. Đang copy ảnh vào thư mục Val...")
tao_thu_muc_va_copy(val_df, "val")

print("\n[THÀNH CÔNG] Thư mục 'dataset_phuc_tap' đã sẵn sàng!")