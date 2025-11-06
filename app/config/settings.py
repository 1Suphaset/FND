import os

# BASE_DIR ชี้ไปที่ root ของโปรเจกต์ (โฟลเดอร์ FND)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # -> app/config/settings.py
BASE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../"))  # ขึ้นไปสองระดับจาก app/config/

MODEL_PATH = os.path.join(BASE_DIR,"assets", "model_fold_5_new.h5")
VECTORIZER_PATH = os.path.join(BASE_DIR, "assets", "vectorizer-new.pkl")

print("MODEL_PATH:", MODEL_PATH)
print("VECTORIZER_PATH:", VECTORIZER_PATH)
