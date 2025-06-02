# Mercedes-Benz Greener Manufacturing - Test Süresi Tahmini

## Proje Özeti
Bu projede, Mercedes-Benz araçlarının üretim sürecindeki test tezgahında geçirdiği süre tahmin edilmiştir.  
Amacımız, üretim verimliliğini artırmak ve çevresel sürdürülebilirliğe katkı sağlamak için test sürelerini doğru şekilde tahmin etmektir.

## Kullanılan Veri Seti
- Kaggle’dan sağlanan anonimleştirilmiş ve yüksek boyutlu özelliklere sahip veri seti,
- 4209 örnek ve 378 özellik (kategorik + binary),
- Hedef değişken: test süresi (saniye cinsinden).

## Yöntem ve Teknolojiler
- **Veri Ön İşleme:** Sabit sütunların kaldırılması, kategorik değişkenlerin one-hot encoding ile işlenmesi,
- **Boyut İndirgeme:** Keras ile derin autoencoder kullanarak 356 özellik 50 boyuta indirildi,
- **Modelleme:** CatBoost, XGBoost ve LightGBM modellerinin stacking (ensemble) yöntemi,
- **Model Değerlendirme:** R² ve RMSE metrikleri ile performans ölçümü,
- **Model Açıklanabilirlik:** SHAP analizi ile önemli özelliklerin görselleştirilmesi,
- **Arayüz:** Streamlit ile kullanıcı dostu tahmin uygulaması.

## Performans Sonuçları
- Autoencoder + CatBoost + Ridge stacking modeli ile:
  - R² Skoru: 0.8938 (yaklaşık %89,4 açıklama),
  - RMSE: 4.13 saniye.

## Öğrendiklerim ve Katkılar
- Yüksek boyutlu ve anonim verilerle çalışmanın zorlukları,
- Derin öğrenme ile boyut indirgeme teknikleri,
- Güçlü ensemble modellerin oluşturulması,
- Model açıklanabilirliği için SHAP kullanımı,
- Projeyi uçtan uca Streamlit ile deploy etme tecrübesi.

## Geliştirmeye Açık Alanlar
- Daha gelişmiş hiperparametre optimizasyonu (Optuna, Hyperopt),
- Kategorik özelliklerin daha etkin kullanımı,
- Streamlit arayüzünün gerçek veri girişine uyarlanması,
- Modelin canlı veri ile güncellenmesi ve otomatik eğitimi.

## Dosya Listesi
- `app.py`: Streamlit uygulama kodları,
- `stacking_model.pkl`: Eğitilmiş final stacking modeli,
- `autoencoder_encoder.keras`: Autoencoder encoder modeli,
- `catboost_model.cbm`: CatBoost modeli,
- `requirements.txt`: Projede kullanılan Python paketleri,
- `README.md`: Proje açıklaması.

---

## Kullanım

```bash
pip install -r requirements.txt
streamlit run app.py
Arayüz üzerinden araç özelliklerini girerek test süresi tahmini yapabilirsiniz.


Lisans
MIT License


İletişim
Her türlü soru ve öneriler için [kaburgada@gmail.com] adresinden bana ulaşabilirsiniz.