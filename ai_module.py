from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Tải mô hình nhận diện (lần đầu chạy sẽ hơi lâu để tải model)
classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

def phan_loai_quan_ao(image_path):
    image = Image.open(image_path)
    # Định nghĩa các nhãn bạn muốn AI nhận diện
    labels_to_search = ["t-shirt", "jeans", "dress", "sneakers", "shirt", "shorts"]
    
    results = classifier(image, candidate_labels=labels_to_search)
    
    # Trả về nhãn có độ tin cậy cao nhất
    return results[0]['label']

def lay_mau_chu_dao(image_path):
    # Đọc ảnh và chuyển sang hệ màu RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Thu nhỏ ảnh để tính toán cho nhanh
    image = cv2.resize(image, (50, 50))
    pixels = image.reshape((-1, 3))
    
    # Dùng K-Means gom cụm thành 2 màu, lấy màu chiếm diện tích lớn nhất
    kmeans = KMeans(n_clusters=2, n_init=10)
    kmeans.fit(pixels)
    
    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
    
    # Trả về mã màu RGB (bạn có thể viết thêm hàm chuyển RGB sang text như "Đỏ", "Xanh")
    return [int(c) for c in dominant_color]