from tensorflow import keras
import os

# Just point directly to the file
old_model_path = "plant_model.h5"
new_model_path = "plant_model.keras"

if os.path.exists(old_model_path):
    print(f"📂 Found: {old_model_path}")
    model = keras.models.load_model(old_model_path, compile=False)
    model.save(new_model_path)
    print(f"✅ Converted {old_model_path} → {new_model_path}")
else:
    print(f"❌ File not found: {old_model_path}. Please check your folder.")
