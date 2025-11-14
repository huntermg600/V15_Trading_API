import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from google.cloud import storage # ‼️ مكتبة Google Cloud
import tempfile

# --- الإعدادات ---
# ‼️ تأكد من أن هذه الأسماء تطابق ما ستنشئه في Google Cloud
BUCKET_NAME = "v10-model-storage-hunter" # ‼️ (استخدم هذا الاسم بالضبط في الخطوة 3)
MODEL_FILE_NAME = "random_forest_eurusd_v10_full_SR.joblib" # ‼️ (اسم ملفك الضخم)

# متغير عالمي لحفظ النموذج بعد تحميله
model = None

def download_model_from_gcs():
    """
    يقوم بتحميل ملف النموذج من Google Cloud Storage إلى ملف مؤقت
    """
    global model
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_FILE_NAME)
        
        # إنشاء ملف مؤقت آمن
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            print(f"🔄 [1/2] جاري تحميل النموذج {MODEL_FILE_NAME} من GCS...")
            blob.download_to_filename(temp_file.name)
            print("✅ تم التحميل بنجاح.")
            
            print(f"🔄 [2/2] جاري تحميل النموذج إلى الذاكرة...")
            model = joblib.load(temp_file.name)
            print(f"✅✅✅ نجاح! تم تحميل النموذج ({len(model.estimators_)} شجرة).")
        
        # حذف الملف المؤقت بعد التحميل
        os.remove(temp_file.name)
        
    except Exception as e:
        print(f"❌ خطأ فادح أثناء تحميل النموذج من GCS: {e}")
        model = None # التأكد من أن النموذج فارغ في حالة الفشل

# ===============================================
# تهيئة الخادم (Flask)
# ===============================================
app = Flask(__name__)

# ---------------------------
# تحميل النموذج عند بدء تشغيل الخادم
# ---------------------------
@app.before_request
def load_model():
    global model
    if model is None:
        print("‼️ النموذج غير موجود، جاري التحميل من GCS...")
        download_model_from_gcs()

# ---------------------------
# نقطة النهاية (Endpoint) الرئيسية
# ---------------------------
@app.route("/")
def home():
    if model is None:
        return "<h1>❌ خطأ: فشل تحميل النموذج.</h1><p>الرجاء مراجعة سجلات Cloud Run.</p>", 500
    return f"<h1>🧠 V10 Random Forest API (Cloud Run)</h1><p>تم تحميل النموذج ({len(model.estimators_)} شجرة) وجاهز للعمل.</p>"

# ---------------------------
# نقطة نهاية التنبؤ (لـ MQL5)
# ---------------------------
@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        print("‼️ فشل التنبؤ: النموذج غير محمل.")
        return jsonify({"error": "Model is not loaded"}), 500

    try:
        data = request.json
        features_str = data.get('features')
        if not features_str:
            return jsonify({"error": "No 'features' key found"}), 400
            
        features_list = [float(f) for f in features_str.split(',')]
        if len(features_list) != 21:
            return jsonify({"error": f"Expected 21 features, received {len(features_list)}"}), 400
            
        features_np = np.array(features_list).reshape(1, -1)
        prediction_prob = model.predict_proba(features_np)
        buy_probability = prediction_prob[0][1]
        
        return jsonify({"prediction": buy_probability})
    except Exception as e:
        print(f"‼️ خطأ أثناء التنبؤ: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------
# تشغيل الخادم
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))