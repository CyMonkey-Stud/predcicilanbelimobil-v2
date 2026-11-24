# Simulasi Cicilan dan Kemampuan Finansial Mobil

Proyek data science untuk membandingkan algoritma regresi dan klasifikasi, menghitung simulasi cicilan, serta merekomendasikan kendaraan berdasarkan kemampuan finansial. Aplikasi menggunakan model hasil notebook, bukan model lama.

## Hasil penelitian

- 75.020 transaksi kredit valid dari 100.000 transaksi awal.
- 60.016 data development, 15.004 final test; tuning pada 12.000 sampel development dengan 5-fold CV.
- Model regresi terpilih: Linear Regression dengan fitur finansial (MAE final test Rp0,251).
- Model klasifikasi terpilih: Decision Tree dengan fitur finansial (macro-F1 final test 1,0).
- Random Forest, XGBoost dan algoritma lain tetap dibandingkan di notebook dan laporan CV.
- Target cicilan dataset mengikuti rumus tanpa bunga; label klasifikasi berasal dari rasio cicilan terhadap pendapatan ≤35%. Skor tinggi merefleksikan target deterministik, bukan kemampuan memprediksi persetujuan kredit atau gagal bayar nyata.

## Fitur aplikasi

1. Simulasi tanpa bunga sesuai dataset atau bunga flat berdasarkan asumsi pengguna.
2. Rasio seluruh cicilan, batas kemampuan bayar, anggaran DP, total pembayaran dan sisa penghasilan.
3. Prediksi ML ditampilkan terpisah dari keputusan simulasi; peringatan input di luar rentang training.
4. Rekomendasi maksimal lima mobil unik berdasarkan batas rasio 35%, anggaran DP, skema pembiayaan dan preferensi ranking.
5. Tab evaluasi model, tabel cross-validation, uji generalisasi, dan grafik penelitian.

Harga katalog bukan harga pasar terkini. Simulasi tidak mencakup biaya hidup, biaya administrasi, asuransi, pajak atau perawatan. Tidak ada keputusan persetujuan kredit yang dibuat aplikasi.

## Menjalankan aplikasi

Gunakan Python 3.13 (training pada 3.13.5). Dari direktori root repository:

```bash
python -m venv .venv
```

Aktifkan virtual environment (Windows PowerShell: `.venv\Scripts\Activate.ps1`; macOS/Linux: `source .venv/bin/activate`), lalu:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy melalui Streamlit Community Cloud

1. Buat repository GitHub baru, tambahkan README awal jika menggunakan web upload.
2. Ekstrak ZIP dan unggah isi direktori `predcicilanbelimobil-v2/` (bukan ZIP atau direktori pembungkus) ke root repo. Pertahankan susunan folder. Jangan unggah dataset mentah.
3. Buka https://share.streamlit.io/ dan hubungkan akun GitHub. Pilih **Create app** dan opsi deploy dari repo yang sudah ada.
4. Isi repository `USERNAME/predcicilanbelimobil-v2`, branch `main`, main file path `app.py`.
5. Di Advanced settings pilih Python **3.13**, simpan lalu deploy. Tidak membutuhkan secrets.
6. Periksa tiga tab dan lakukan simulasi dengan berbagai input. Commit perubahan ke repo untuk memperbarui aplikasi.

## Training ulang (opsional, tidak diperlukan untuk deploy)

Dataset mentah tidak disertakan karena mengandung identitas/kontak. Gunakan salinan asli secara lokal dan pastikan hak penggunaan data. Jalankan notebook `notebooks/car_financing_analysis.ipynb` dari root repository, dengan dataset `indonesia_car_sales_bigdata.csv` di current working directory. Untuk Colab, unggah notebook dan CSV ke `/content`. Instal versi library sesuai `notebooks/requirements-training.txt` bila diperlukan. Notebook menulis `outputs_car_financing/`.

Setelah training ulang, salin `model_cicilan.pkl`, `model_kelayakan.pkl`, `model_metadata.json` ke `models/`, `katalog_mobil.csv` ke `data/`, `features.py` ke `src/`, seluruh laporan CSV/JSON evaluasi dan folder `figures/` ke `reports/`. Sinkronkan versi dependency dengan metadata model. Jangan mencampur model, metadata, dan feature engineering dari training berbeda. Muat PKL hanya dari sumber yang dipercaya.

## Pengujian

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py src tests
```

Sembilan unit test mencakup perhitungan, ambang rasio, validasi input, monotonisitas seluruh kombinasi katalog, model dan rekomendasi. Pengujian UI Streamlit dan deployment cloud perlu dijalankan pada lingkungan yang memiliki Streamlit.

## Struktur

```text
app.py                         # UI Streamlit
requirements.txt               # dependensi aplikasi
models/                        # PKL dan metadata training
data/katalog_mobil.csv          # katalog tanpa identitas pribadi
src/                           # feature engineering, inferensi, kalkulator, rekomendasi
notebooks/                     # penelitian dan dependensi training
reports/                       # metrik, audit data, grafik
tests/test_system.py            # pengujian otomatis
```
