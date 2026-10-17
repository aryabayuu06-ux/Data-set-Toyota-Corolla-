# Prediksi Harga Mobil Toyota Corolla dengan Machine Learning

Proyek ini menggunakan dataset **Toyota Corolla** untuk memprediksi harga mobil berdasarkan karakteristik kendaraan. Notebook mencakup eksplorasi data (*Exploratory Data Analysis/EDA*), preprocessing, pelatihan dan evaluasi model regresi, visualisasi hasil, penyimpanan model, pembuatan API, dan monitoring prediksi.

## Tujuan Proyek

- Memahami karakteristik dataset mobil Toyota Corolla.
- Menyiapkan data agar dapat digunakan oleh model machine learning.
- Membangun model regresi untuk memperkirakan harga mobil.
- Mengevaluasi prediksi menggunakan MAE, MSE, RMSE, dan R².
- Menyediakan prediksi melalui API menggunakan FastAPI.
- Mendemonstrasikan monitoring data drift dan perubahan performa model.

## Struktur Repository

Pastikan berkas utama berada dalam satu folder:

```text
.
├── README.md
├── Toyota_Corolla_ML_Rapi.ipynb
└── ToyotaCorolla.csv
```

Notebook juga dapat menghasilkan berkas tambahan untuk deployment dan monitoring setelah seluruh sel terkait dijalankan.

## Dataset

Dataset yang digunakan adalah `ToyotaCorolla.csv`, berisi **1.436 baris dan 10 kolom**. Target prediksi adalah `Price`; sembilan kolom lainnya berperan sebagai fitur kendaraan.

### Deskripsi kolom

| Kolom | Deskripsi |
|---|---|
| `Price` | Harga mobil dan target prediksi |
| `Age` | Umur mobil (bulan) |
| `KM` | Jarak tempuh mobil (kilometer) |
| `FuelType` | Jenis bahan bakar: Petrol, Diesel, atau CNG |
| `HP` | Tenaga mesin (horsepower) |
| `MetColor` | Indikator warna metalik (0 atau 1) |
| `Automatic` | Indikator transmisi otomatis (0 atau 1) |
| `CC` | Kapasitas mesin (cc) |
| `Doors` | Jumlah pintu |
| `Weight` | Berat mobil |

### Ringkasan data mentah

| Informasi | Hasil |
|---|---:|
| Jumlah baris | 1.436 |
| Jumlah kolom | 10 |
| Missing value | 0 |
| Baris duplikat | 1 |
| Harga minimum (`Price`) | 4.350 |
| Harga median (`Price`) | 9.900 |
| Harga rata-rata (`Price`) | 10.730,82 |
| Harga maksimum (`Price`) | 32.500 |

Satuan atau mata uang pada kolom `Price` tidak dinyatakan secara eksplisit dalam berkas CSV, sehingga angka di atas mengikuti nilai asli dataset.

### Distribusi beberapa kolom

| Fitur | Kategori | Jumlah |
|---|---|---:|
| `FuelType` | Petrol | 1.264 |
| `FuelType` | Diesel | 155 |
| `FuelType` | CNG | 17 |
| `Automatic` | 0 (bukan otomatis) | 1.356 |
| `Automatic` | 1 (otomatis) | 80 |
| `MetColor` | 0 | 467 |
| `MetColor` | 1 | 969 |

## Alur Pengolahan Data

1. Membaca `ToyotaCorolla.csv` dan memeriksa informasi serta statistik deskriptif data.
2. Memeriksa missing value, baris duplikat, distribusi fitur, box plot, dan korelasi.
3. Menyaring outlier menggunakan metode *Interquartile Range* (IQR).
4. Melakukan standardisasi pada kolom numerik dengan `StandardScaler`.
5. Memisahkan fitur (`X`) dan target (`y`), serta mengubah fitur kategorikal menjadi numerik menggunakan `pd.get_dummies()`.
6. Membagi data menjadi 80% data training dan 20% data testing dengan `random_state=42`.
7. Melatih dan mengevaluasi model regresi.
8. Membuat visualisasi aktual vs prediksi, residual, dan *feature importance*.
9. Menyimpan model terpilih dan menyiapkan artefak deployment.
10. Menjalankan contoh API FastAPI serta monitoring data drift dan performa model.

### Perhatian pada filter IQR

Filter IQR pada notebook memeriksa semua kolom numerik, termasuk `Price` dan fitur biner `Automatic`. Dengan dataset ini, filter menghasilkan **1.133 baris dari 1.436 baris awal**. Semua baris dengan `Automatic = 1` ikut terhapus; setelah filter, 1.133 baris tersisa semuanya memiliki `Automatic = 0`.

Hal ini terjadi karena nilai `1` pada fitur `Automatic` dianggap berada di luar rentang IQR ketika nilai tersebut jarang muncul. Akibatnya, model tidak mempelajari variasi kendaraan bertransmisi otomatis. Perlu dipertimbangkan untuk memperbaiki strategi penanganan outlier sebelum menggunakan model untuk aplikasi nyata.

## Metode dan Hasil Evaluasi

Notebook saat ini melatih dan membandingkan tiga algoritma regresi: **Gradient Boosting Regressor**, **Linear Regression**, dan **LARS (Least Angle Regression)**. Berdasarkan hasil evaluasi yang tersimpan di notebook, Gradient Boosting Regressor memperoleh R² tertinggi dan dipilih sebagai model utama.

### Hasil pada data testing

Karena kolom `Price` ikut distandardisasi sebelum pembagian data, metrik MAE, MSE, dan RMSE pada tabel berikut berada pada **skala standar**, bukan pada skala harga asli.

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Gradient Boosting Regressor | 0.3196 | 0.1708 | 0.4132 | 0.8332 |
| Linear Regression | 0.3397 | 0.1927 | 0.4390 | 0.8118 |
| LARS | 0.4307 | 0.3240 | 0.5692 | 0.6836 |

Nilai R² sebesar `0.8332` menunjukkan bahwa model menjelaskan sekitar 83,32% variasi target pada data testing dalam eksperimen tersebut. Hasil ini tidak menjamin tingkat performa yang sama untuk semua mobil atau data baru.

### Evaluasi prediksi pada skala harga asli

Bagian deployment notebook menguji ulang prediksi dalam satuan harga asli dataset dan mencatat baseline berikut:

| Metrik | Nilai |
|---|---:|
| MAE | 740.1718 |
| RMSE | 956.8953 |
| R² | 0.8332 |

Nilai tersebut merupakan hasil yang tersimpan pada output notebook. Nilai dapat berubah jika dataset, proses preprocessing, pembagian data, atau versi library berubah.

## Teknologi yang Digunakan

- Python
- pandas dan NumPy
- Matplotlib dan Seaborn
- scikit-learn
- Joblib
- SciPy
- FastAPI dan Uvicorn
- HTTPX untuk pengujian API

## Instalasi dan Menjalankan Notebook

### 1. Siapkan berkas

Letakkan `Toyota_Corolla_ML_Rapi.ipynb` dan `ToyotaCorolla.csv` di folder yang sama.

### 2. Instal library

Di terminal, jalankan:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib scipy fastapi httpx uvicorn jupyter
```

### 3. Jalankan notebook

Buka notebook di Jupyter Notebook, JupyterLab, atau lingkungan yang mendukung Jupyter. Pilih **Restart & Run All** agar sel dijalankan secara berurutan dari awal. Beberapa sel menghasilkan berkas kode dan artefak deployment yang digunakan oleh sel berikutnya.

## Deployment API dengan FastAPI

Notebook menghasilkan berkas yang dibutuhkan untuk menjalankan API, antara lain `model_bundle.joblib`, `model_metadata.json`, `serving.py`, `app.py`, dan `requirements.txt`. Jalankan sel deployment terlebih dahulu untuk membuat berkas tersebut.

Jalankan API dari terminal pada folder proyek:

```bash
uvicorn app:app --reload
```

Dokumentasi API interaktif tersedia di:

```text
http://127.0.0.1:8000/docs
```

### Endpoint API

| Method | Endpoint | Fungsi |
|---|---|---|
| `GET` | `/health` | Memeriksa status API dan informasi model |
| `POST` | `/predict` | Menghasilkan prediksi harga mobil |

Contoh request untuk endpoint `/predict`:

```json
{
  "Age": 30,
  "KM": 60000,
  "FuelType": "Petrol",
  "HP": 110,
  "MetColor": 1,
  "Automatic": 0,
  "CC": 1600,
  "Doors": 4,
  "Weight": 1050
}
```

`FuelType` menerima `CNG`, `Diesel`, atau `Petrol`. Nilai masukan lainnya divalidasi berdasarkan batas yang didefinisikan dalam `app.py`. API mengembalikan prediksi harga dan informasi model. Request prediksi juga dicatat pada `prediction_log.csv`.

## Monitoring Model

Notebook mendemonstrasikan monitoring dalam dua aspek.

### 1. Data drift

- Menghitung *Population Stability Index* (PSI) untuk membandingkan distribusi fitur data acuan dan data baru.
- Menggunakan uji Kolmogorov–Smirnov (KS) pada fitur numerik.
- Menampilkan laporan serta visualisasi drift antarbatch.

Ambang PSI yang digunakan:

| Nilai PSI | Status |
|---|---|
| `< 0.10` | AMAN |
| `0.10` sampai `< 0.25` | WASPADA |
| `>= 0.25` | DRIFT |

### 2. Perubahan performa

Notebook menghitung MAE, RMSE, dan R² pada batch dengan harga aktual yang tersedia. Status `ALERT` diberikan jika RMSE meningkat lebih dari 25% dibanding baseline atau R² turun lebih dari 0,10.

Beberapa batch monitoring adalah simulasi, termasuk skenario pergeseran distribusi data dan penurunan harga pasar 20%. Hasil simulasi bukan bukti bahwa perubahan tersebut telah terjadi di pasar sebenarnya.

## Berkas yang Dihasilkan

Berkas berikut dibuat saat bagian terkait pada notebook dijalankan:

| Berkas | Kegunaan |
|---|---|
| `best_model.joblib` | Model dengan nilai R² tertinggi |
| `gbr_model.joblib` | Model Gradient Boosting Regressor |
| `model_bundle.joblib` | Paket model, preprocessing, urutan fitur, dan metadata untuk deployment |
| `model_metadata.json` | Metadata model dalam format JSON |
| `serving.py` | Fungsi preprocessing dan prediksi untuk API |
| `app.py` | Implementasi API FastAPI |
| `requirements.txt` | Daftar dependensi untuk deployment |
| `prediction_log.csv` | Log request dan prediksi API |
| `monitoring_log.csv` | Log evaluasi performa monitoring |

## Catatan dan Keterbatasan

- **Filter IQR menghapus seluruh baris mobil otomatis.** Model hasil preprocessing tidak dapat dianggap tervalidasi untuk mobil otomatis.
- **Standardisasi dilakukan sebelum pembagian training/testing.** Ini berisiko menimbulkan *data leakage*. Untuk evaluasi yang lebih ketat, pembagian data sebaiknya dilakukan terlebih dahulu, kemudian preprocessing di-*fit* hanya pada data training melalui pipeline.
- Karena `Price` distandardisasi, MAE/MSE/RMSE dari evaluasi utama notebook bukan dalam satuan harga asli. Untuk laporan bisnis, gunakan metrik hasil prediksi yang sudah dikembalikan ke skala harga asli.
- Ambang drift dan batch bergeser dalam notebook adalah contoh implementasi dan simulasi, bukan aturan universal.
- Berkas `.joblib` menggunakan format berbasis pickle. Muat berkas model hanya dari sumber yang dipercaya dan upayakan menggunakan versi library yang konsisten.

## Sumber Data dan Lisensi

File `ToyotaCorolla.csv` digunakan sebagai dataset masukan proyek. Informasi sumber publik dan lisensi dataset tidak dicantumkan secara eksplisit pada file CSV atau notebook ini. Tambahkan tautan sumber asli dan ketentuan lisensinya sebelum repository dipublikasikan secara luas.
