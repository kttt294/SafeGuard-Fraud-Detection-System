import pandas as pd
import pickle
import os
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Đường dẫn thư mục
DATA_PATH = '../data/processed/data_splits.pkl'
MODEL_PATH = 'trained_model.pkl'

print("--- Quick Train: Preparing Model for Web App ---")

# 1. Load data
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Không tìm thấy file {DATA_PATH}. Vui lòng chạy 1_eda_preprocessing.py trước.")
    exit()

with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

print(f"Dữ liệu huấn luyện: {X_train.shape}")

# 2. Xây dựng Pipeline với SMOTE và Random Forest
# Chúng ta dùng tham số nhẹ (n_estimators=50) để train nhanh
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))
])

print("\n--- Training Model (với SMOTE)... Vui lòng đợi ---")
pipeline.fit(X_train, y_train)

# 3. Đánh giá nhanh
print("\n--- Đánh giá nhanh trên tập Test ---")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 4. Lưu mô hình (Pipeline bao gồm cả logic dự đoán)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(pipeline, f)

print(f"\n[SUCCESS] Mô hình đã được lưu tại '{MODEL_PATH}'")
print("Bạn có thể sử dụng file này để bắt đầu xây dựng Web App bằng Streamlit.")
