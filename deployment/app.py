import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Cấu hình trang
st.set_page_config(
    page_title="SafeGuard Banking | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# Đường dẫn file
MODEL_PATH = '../modeling/trained_model.pkl'

# Tải mô hình
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

# --- GIAO DIỆN CHÍNH ---
st.title("🛡️ SafeGuard Banking | Fraud Analyst Operations Dashboard")
st.markdown("""
Welcome back, **Analyst**. Current System Status: <span style='color:green; font-weight:bold'>ACTIVE</span> | 
Model Version: `v1.0.4-stable` | Last Retraining: `2026-03-15`
""", unsafe_allow_html=True)

# --- QUY TRÌNH KIỂM TRA CỘT (Feature Alignment) ---
# Danh sách cột bắt buộc và đúng thứ tự mà mô hình đã học
FEATURE_COLUMNS = ['scaled_amount', 'scaled_time'] + [f'V{i}' for i in range(1, 29)]

if model is None:
    st.error(f"❌ Không tìm thấy mô hình tại `{MODEL_PATH}`. Vui lòng chạy `quick_train_for_web.py` trước.")
else:
    # --- Sidebar: Chọn dịch vụ ---
    st.sidebar.image("https://img.icons8.com/color/96/000000/shield.png")
    st.sidebar.header("Operations Menu")
    menu = st.sidebar.selectbox(
        "Select Operation Mode:", 
        ["🔍 Transaction Investigation", "📊 Periodic Batch Audit"]
    )

    if menu == "🔍 Transaction Investigation":
        st.subheader("🕵️ Deep Investigation Mode")
        st.info("Dành cho chuyên viên điều tra các giao dịch bị hệ thống Core-Banking đánh dấu nghi vấn.")
        
        col1, col2, col3 = st.columns(3)
        
        # Chúng ta cho phép nhập 10 biến quan trọng nhất, các biến khác mặc định = 0
        with col1:
            v17 = st.number_input("V17 (Tương quan mạnh nhất)", value=0.0)
            v14 = st.number_input("V14", value=0.0)
            v12 = st.number_input("V12", value=0.0)
            v10 = st.number_input("V10", value=0.0)
        
        with col2:
            v16 = st.number_input("V16", value=0.0)
            v3 = st.number_input("V3", value=0.0)
            v7 = st.number_input("V7", value=0.0)
            v11 = st.number_input("V11", value=0.0)
            
        with col3:
            amount = st.number_input("Amount (Số tiền giao dịch)", value=100.0)
            time = st.number_input("Time (Giây sau giao dịch đầu)", value=1000.0)
            
        if st.button("🚀 Kiểm tra ngay"):
            # Tạo DataFrame 1 dòng với đầy đủ tên cột để tránh sai thứ tự
            input_df = pd.DataFrame(np.zeros((1, 30)), columns=FEATURE_COLUMNS)
            input_df['scaled_amount'] = amount / 100 
            input_df['scaled_time'] = time / 1000
            input_df['V17'], input_df['V14'], input_df['V12'] = v17, v14, v12
            input_df['V10'], input_df['V16'], input_df['V11'] = v10, v16, v11
            
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            if prediction == 1:
                st.error(f"⚠️ **CẢNH BÁO:** Giao dịch có dấu hiệu GIAN LẬN! (Xác suất: {probability:.2%})")
                st.image("https://img.icons8.com/color/96/000000/high-priority.png")
            else:
                st.success(f"✅ Giao dịch HỢP LỆ. (Xác suất gian lận: {probability:.2%})")
                st.image("https://img.icons8.com/color/96/000000/verified-badge.png")

    elif menu == "📊 Periodic Batch Audit":
        st.subheader("📂 Batch Audit Interface")
        st.info("💡 Mẹo: Hệ thống tự động căn chỉnh cột. Bạn chỉ cần đảm bảo file có đủ các tên cột cần thiết.")
        
        # --- TÍNH NĂNG TẢI FILE MẪU ---
        # Tạo file mẫu với 5 dòng dữ liệu trắng
        template_df = pd.DataFrame(np.zeros((5, 30)), columns=FEATURE_COLUMNS)
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📄 Tải File mẫu (CSV Template)",
            data=template_csv,
            file_name='transaction_template.csv',
            mime='text/csv',
            help="Tải file mẫu để biết định dạng các cột mà hệ thống yêu cầu."
        )
        st.divider()

        uploaded_file = st.file_uploader("Chọn file CSV giao dịch của bạn", type="csv")
        
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Xem trước dữ liệu:", df_upload.head())
            
            # KIỂM TRA CỘT
            missing_cols = [c for c in FEATURE_COLUMNS if c not in df_upload.columns]
            
            if missing_cols:
                st.error(f"❌ File thiếu các cột bắt buộc: {', '.join(missing_cols)}")
            else:
                st.success("✅ File hợp lệ! Đã tìm thấy đầy đủ các đặc trưng.")
                # TỰ ĐỘNG SẮP XẾP LẠI CỘT CHO ĐÚNG THỨ TỰ MÔ HÌNH CẦN
                df_ready = df_upload[FEATURE_COLUMNS]
                
                if st.button("🔍 Tiến hành phân tích File"):
                    results = model.predict(df_ready)
                    df_upload['Dự đoán'] = results
                    df_upload['Kết quả'] = df_upload['Dự đoán'].apply(lambda x: '‼️ GIAN LẬN' if x == 1 else '✅ Hợp lệ')
                    
                    # Hiển thị thống kê
                    fraud_count = (results == 1).sum()
                    st.warning(f"Phát hiện **{fraud_count}** giao dịch nghi vấn trong tổng số {len(df_upload)} dòng.")
                    
                    st.dataframe(df_upload[df_upload['Dự đoán'] == 1])
                    
                    # Cho phép download kết quả
                    csv = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Tải file kết quả báo cáo",
                        data=csv,
                        file_name='fraud_detection_results.csv',
                        mime='text/csv',
                    )

    # --- MỚI: THEO DÕI LIVE TỪ API (Database) ---
    st.sidebar.divider()
    st.sidebar.subheader("Live Monitor (from API)")
    log_file = '../data/outputs/transaction_history.json'
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            live_logs = json.load(f)
        
        # Hiển thị 5 giao dịch mới nhất lên Sidebar để tạo cảm giác Live
        for log in reversed(live_logs[-5:]):
            color = "red" if log['decision'] == "BLOCK" else "green"
            st.sidebar.markdown(f"**{log['timestamp']}**")
            st.sidebar.markdown(f"ID: `{log['transaction_id']}`")
            st.sidebar.markdown(f"Status: <span style='color:{color}'>{log['decision']}</span> (${log['amount']})", unsafe_allow_html=True)
            st.sidebar.divider()
    else:
        st.sidebar.info("Đang đợi dữ liệu từ API...")

# --- Footer ---
st.divider()
st.caption("© 2024 SafeGuard Banking Intelligence | Đội ngũ Phân tích Rủi ro")
