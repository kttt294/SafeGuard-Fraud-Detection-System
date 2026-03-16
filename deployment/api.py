from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
import os
import time
import json

# 1. Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="SafeGuard Banking Real-time API",
    description="Hệ thống lõi xử lý giao dịch tự động của ngân hàng",
    version="1.0.0"
)

# 2. Cấu hình mô hình
MODEL_PATH = '../modeling/model.pkl'
FEATURE_COLUMNS = ['scaled_amount', 'scaled_time'] + [f'V{i}' for i in range(1, 29)]

# 3. Load model khi khởi động
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# 4. Định nghĩa cấu trúc dữ liệu gửi đến (Schema)
class Transaction(BaseModel):
    amount: float = Field(..., example=150.75)
    time_offset: float = Field(..., example=1200.0)
    v_features: list[float] = Field(..., min_items=28, max_items=28, description="Danh sách 28 đặc trưng PCA từ V1 đến V28")

@app.get("/")
def home():
    return {"status": "ONLINE", "message": "Ngân hàng SafeGuard - Hệ thống lõi đã sẵn sàng."}

@app.post("/verify")
async def verify_transaction(tx: Transaction):
    """
    API tiếp nhận giao dịch trực tiếp từ Máy POS, Web, App.
    Trả về kết quả Chấp thuận hoặc Từ chối trong mili giây.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được tải lên máy chủ.")
    
    start_time = time.time()
    
    try:
        # Chuẩn bị DataFrame đúng thứ tự mô hình yêu cầu
        input_data = [tx.amount/100, tx.time_offset/1000] + tx.v_features
        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        
        # Dự đoán
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        process_time = time.time() - start_time
        
        # Logic nghiệp vụ: Nếu gian lận xác suất cao -> Từ chối ngay
        decision = "BLOCK" if prediction == 1 else "APPROVE"
        
        result = {
            "transaction_id": f"TXN-{int(time.time()*1000)}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": tx.amount,
            "decision": decision,
            "fraud_probability": f"{probability:.4%}",
            "processing_time_ms": f"{process_time*1000:.2f}ms"
        }

        # --- GHI VÀO DATABASE (Mô phỏng bằng file JSON) ---
        log_file = '../data/outputs/transaction_history.json'
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                try:
                    logs = json.load(f)
                except:
                    logs = []
        
        logs.append(result)
        with open(log_file, 'w') as f:
            json.dump(logs[-100:], f, indent=4) # Lưu 100 giao dịch gần nhất
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý dữ liệu: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
