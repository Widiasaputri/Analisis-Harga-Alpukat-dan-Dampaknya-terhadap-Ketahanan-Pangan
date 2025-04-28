# 🥑 Avocado Price Prediction API

Selamat datang!  
Ini adalah project sederhana untuk memprediksi **harga alpukat** di pasar Amerika Serikat, dikembangkan menggunakan **FastAPI** dan model **XGBoost**. Project ini dibuat sebagai bagian dari tahapan Deployment dalam analisis data harga alpukat.

## 📁 Struktur File
- `main.py` → Script utama API (FastAPI app)
- `xgb_best_model.pkl` → File model Machine Learning yang sudah dilatih
- `scaler.pkl` → File StandardScaler untuk preprocessing input
- `requirements.txt` → Daftar dependensi Python yang dibutuhkan

## 🚀 Fitur API
- Menerima data penjualan alpukat melalui POST request
- Memberikan prediksi harga rata-rata dalam satuan dolar
- Mudah digunakan untuk integrasi ke aplikasi, dashboard, atau sekadar eksperimen

## 🛠️ Cara Menjalankan API
1. **Clone repositori**
   ```bash
   git clone https://github.com/Widiasaputri/Analisis-Harga-Alpukat-dan-Dampaknya-terhadap-Ketahanan-Pangan.git
   cd Analisis-Harga-Alpukat-dan-Dampaknya-terhadap-Ketahanan-Pangan
   ```

2. **Buat dan aktifkan virtual environment**
   ```bash
   python -m venv env
   .\env\Scripts\activate   
   ```

3. **Install semua dependensi**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan API**
   ```bash
   uvicorn main:app --reload
   ```

5. **Buka dokumentasi interaktif**
   - Akses melalui browser: 👉 [http://127.0.0.1:8000/docs].

---

## 📩 Contoh Input
```
{
  "total_volume": 3500,
  "_4046": 700,
  "_4225": 1200,
  "_4770": 600,
  "total_bags": 200,
  "small_bags": 100,
  "large_bags": 80,
  "xlarge_bags": 20,
  "type": "organic",
  "year": 2025,
  "month": 7,
  "season": "summer",
  "region": "California"
}
```

## 🎯 Contoh Output
```
{
  "predicted_price": 2.178629
}
