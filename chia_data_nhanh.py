import splitfolders

print("Đang chia dữ liệu thành Train và Val...")
# Lấy thư mục gốc, tự động chia 80% train, 20% val và lưu ra thư mục mới tên là 'dataset_quan_ao'
splitfolders.ratio("Apparel_images", output="dataset_quan_ao", seed=42, ratio=(0.8, 0.2))
print("Hoàn tất! Bật code train AI lên thôi.")