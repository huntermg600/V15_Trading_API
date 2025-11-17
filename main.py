import fastapi
import uvicorn
import joblib
import numpy as np
import pandas as pd
import os
import tempfile
from pydantic import BaseModel
from typing import List
from google.cloud import storage

# --- 1. الإعدادات ---
# ‼️ تأكد من أن هذه الأسماء تطابق ما لديك في Google Cloud
BUCKET_NAME = "v15-model-storage-hunter" # ‼️ (الاسم الذي أنشأته لـ Bucket)
MODEL_FILE_NAME = "random_forest_eurusd_v15_upgraded_scalper.joblib" # ‼️ (اسم ملف النموذج الضخم)

# ‼️ تأكد من أن هذه الـ 21 ميزة بالترتيب الصحيح
FEATURE_COLUMNS = [
    'DayOfWeek', 'HourOfDay', 'RSI_m15', 'ATR_m15', 'MACD_m15', 
    'MACD_signal_m15', 'Momentum_m15_0', 'Momentum_m15_1', 'SMA50_h1', 
    'Momentum_h1_0', 'SMA50_h4', 'SMA200_h4', 'Dist_from_High_m15', 
    'Dist_from_Low_m15', 'Dist_from_High_h1', 'Dist_from_Low_h1', 
    'Dist_from_High_h4', 'Dist_from_Low_h4', 'Volume', 'Volume_h1', 'Volume_h4'
]

model = None
app = fastapi.FastAPI()

# --- 2. تحميل النموذج عند بدء التشغيل (FastAPI) ---
@app.on_event("startup")
def load_model_on_startup():
    global model
    if model is not None:
        print("✅ النموذج محمل مسبقاً.")
        return
        
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_FILE_NAME)
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            print(f"🔄 [1/2] جاري تحميل {MODEL_FILE_NAME} من Google Storage...")
            blob.download_to_filename(temp_file.name)
            print("✅ تم التحميل بنجاح.")
            
            print(f"🔄 [2/2] جاري تحميل النموذج إلى الذاكرة...")
            model = joblib.load(temp_file.name)
            print("✅✅✅ نجاح! تم تحميل نموذج v15 (Random Forest).")
        
        os.remove(temp_file.name) # حذف الملف المؤقت
        
    except Exception as e:
        print(f"❌ خطأ فادح: فشل تحميل النموذج من GCS: {e}")
        model = None

# --- 3. تحديد هيكل البيانات (FastAPI) ---
class FeaturesInput(BaseModel):
    features: List[float] # (يتطابق مع الإكسبيرت الذي يرسل قائمة)

# --- 4. نقطة نهاية التنبؤ (FastAPI) ---
@app.post("/predict")
async def predict(data: FeaturesInput):
    if model is None:
        print("🔴 خطأ 500: النموذج غير محمل.")
        raise fastapi.HTTPException(status_code=500, detail="Model is not loaded. Check startup logs.")
    
    try:
        features_list = data.features
        
        # تحويلها إلى Pandas DataFrame 
        features_df = pd.DataFrame([features_list], columns=FEATURE_COLUMNS)
        
        # طلب التنبؤ (0 أو 1)
        prediction = model.predict(features_df)
        signal = int(prediction[0])
        
        print(f"🟢 [v15 Server] تم استلام الميزات. الإشارة = {signal}")
        
        # إرسال 0 أو 1 (يتطابق مع الإكسبيرت)
        return {"prediction": signal}
        
    except Exception as e:
        error_message = str(e)
        print(f"🔴 [v15 Server] حدث خطأ أثناء التنبؤ: {error_message}")
        raise fastapi.HTTPException(status_code=500, detail=error_message)

@app.get("/")
def root():
    if model is None:
        return {"message": "❌ خادم v15: فشل تحميل النموذج. راجع السجلات."}
    return {"message": "🧠 خادم v15 (Random Forest) يعمل وجاهز!"}


