import streamlit as st
import numpy as np
import joblib
from keras.models import load_model
from catboost import CatBoostRegressor

# Başlık
st.title("🚗 Mercedes-Benz Test Süresi Tahmini")
st.write("Autoencoder + CatBoost + Ridge ile tahmin yapılır.")

# Model ve encoder yükle
stack_model = joblib.load("stacking_model.pkl")
encoder = load_model("autoencoder_encoder.keras")
cat_model = CatBoostRegressor()
cat_model.load_model("catboost_model.cbm")  # CatBoost modelini .cbm uzantısıyla kaydettiysen

st.subheader("🔧 Araç Özelliklerini Girin")

st.caption("Bu 10 özellik, yüksek boyutlu araç konfigürasyonlarının Autoencoder ile sıkıştırılmış halidir. Gelişmiş modeller bu veriyi kullanır.")


# Örnek başlangıç değerleri (10 tane), kalan 40 sıfır
default_values = [0.01, 0.03, 0.02, 0.00, 0.04, 0.03, 0.04, 0.01, 0.00, 0.00] + [0.0]*40

features = []
for i in range(50):
    val = st.number_input(f"Feature {i}", value=default_values[i])
    features.append(val)



# Tahmin butonu
if st.button("Tahmin Et"):
    encoded_input = np.array(features).reshape(1, -1)

    # CatBoost tahmini (1 boyut)
    cat_pred = cat_model.predict(encoded_input).reshape(-1, 1)

    # Final stacking input (50 + 1 = 51 boyut)
    full_input = np.hstack([encoded_input, cat_pred])

    # Tahmin
    y_pred = stack_model.predict(full_input)
    st.success(f"Tahmini Test Süresi: {y_pred[0]:.2f} saniye")
