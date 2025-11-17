import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from google.cloud import storage
import tempfile

# --- 1. الإعدادات (غيّر هذا ليطابق v15) ---
BUCKET_NAME = "v15-model-storage-hunter" # ‼️ (اسم الخزنة v15)
MODEL_FILE_NAME = "random_forest_eurusd_v15_upgraded_scalper.joblib" # ‼️ (اسم نموذج v15)

# --- 2. قائمة الميزات (للتأكد فقط) ---
FEATURE_COLUMNS = [
    'DayOfWeek', 'HourOfDay', 'RSI_m15', 'ATR_m15', 'MACD_m15', 
    'MACD_signal_m15', 'Momentum_m15_0', 'Momentum_m15_1', 'SMA50_h1', 
    'Momentum_h1_0', 'SMA50_h4', 'SMA200_h4', 'Dist_from_High_m15', 
    'Dist_from_Low_m15', 'Dist_from_High_h1', 'Dist_from_Low_h1', 
    'Dist_from_High_h4', 'Dist_from_Low_h4', 'Volume', 'Volume_h1', 'Volume_h4'
]

model = None
app = Flask(__name__)

# --- 3. دالة تحميل النموذج (من V10) ---
def download_model_from_gcs():
    global model
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_FILE_NAME)
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            print(f"🔄 [v15] جاري تحميل النموذج {MODEL_FILE_NAME} من GCS...")
            blob.download_to_filename(temp_file.name)
            print("✅ [v15] تم التحميل بنجاح.")
            
            print(f"🔄 [v15] جاري تحميل النموذج إلى الذاكرة...")
            model = joblib.load(temp_file.name)
            print("✅✅✅ [v15] نجاح! تم تحميل النموذج.")
        
        os.remove(temp_file.name)
        
    except Exception as e:
        print(f"❌ [v15] خطأ فادح أثناء تحميل النموذج: {e}")
        model = None

# --- 4. التحميل عند بدء التشغيل (Flask) ---
with app.app_context():
    if model is None:
        print("‼️ [v15] النموذج غير موجود، جاري التحميل...")
        download_model_from_gcs()

# --- 5. المسارات (Routes) ---
@app.route("/")
def home():
    if model is None:
        return "<h1>❌ خطأ: فشل تحميل نموذج v15.</h1><p>راجع السجلات.</p>", 500
    return "<h1>🧠 V15 Random Forest API (Flask)</h1><p>النموذج جاهز للعمل.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        print("‼️ فشل التنبؤ: النموذج v15 غير محمل.")
        return jsonify({"error": "Model is not loaded"}), 500

    try:
        data = request.json
        features_list = data.get('features') # (نتوقع قائمة)
        
        if not isinstance(features_list, list) or len(features_list) != 21:
             return jsonify({"error": f"Expected a list of 21 features"}), 400
        
        # تحويلها إلى NumPy Array ثم Pandas DataFrame (لأن v10 يتوقع هذا)
        features_np = np.array(features_list).reshape(1, -1)
        features_df = pd.DataFrame(features_np, columns=FEATURE_COLUMNS)

        # *** هام جداً: نحن نرسل predict_proba (الاحتمالية) ***
        prediction_prob = model.predict_proba(features_df)
        buy_probability = prediction_prob[0][1] # (احتمالية الشراء 0.xx)
        
        print(f"🟢 [v15 Server] تم استلام الميزات. الاحتمالية = {buy_probability}")
        return jsonify({"prediction": buy_probability})

    except Exception as e:
        print(f"‼️ [v15] خطأ أثناء التنبؤ: {e}")
        return jsonify({"error": str(e)}), 500

# --- 6. التشغيل ---
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
