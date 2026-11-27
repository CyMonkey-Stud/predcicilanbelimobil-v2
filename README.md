# Simulasi Cicilan dan Kemampuan Finansial Mobil

Proyek data science untuk membandingkan algoritma regresi dan klasifikasi, menghitung simulasi cicilan, serta merekomendasikan kendaraan berdasarkan kemampuan finansial. Aplikasi menggunakan model hasil notebook, bukan model lama.

## Hasil penelitian

- 75.020 transaksi kredit valid dari 100.000 transaksi awal.
- 60.016 data development, 15.004 final test; tuning pada 12.000 sampel development dengan 5-fold CV.
- Model regresi terpilih: Linear Regression dengan fitur finansial (MAE final test Rp0,251).
- Model klasifikasi terpilih: Decision Tree dengan fitur finansial (macro-F1 final test 1,0).
- Random Forest, XGBoost dan algoritma lain tetap dibandingkan di notebook dan laporan CV.
- Target cicilan dataset mengikuti rumus tanpa bunga; label klasifikasi berasal dari rasio cicilan terhadap pendapatan ≤35%. Skor tinggi merefleksikan target deterministik, bukan kemampuan memprediksi persetujuan kredit atau gagal bayar nyata.

## Pembaruan V4: keputusan simulasi yang dapat dijelaskan

- Setelah menekan **Jalankan simulasi**, status **LAYAK DALAM SIMULASI** muncul hanya jika seluruh cicilan maksimal 35% penghasilan dan DP tidak melebihi anggaran. Jika salah satu gagal, status **TIDAK LAYAK DALAM SIMULASI** muncul dengan alasan nominal dan kondisi kedua syarat.
- Panel alasan memperlihatkan total cicilan, batas, selisih nominal, kebutuhan DP dan sisa/kekurangan anggaran. Keputusan menggunakan perbandingan rupiah tanpa pembulatan persentase tampilan.
- Jika belum layak, tampilkan penghasilan minimum menurut aturan rasio untuk skema yang sama, kekurangan anggaran, serta tiga alternatif DP/tenor terdekat yang sudah lolos kedua syarat. Ranking alternatif mengutamakan jumlah perubahan langkah DP/tenor, kemudian total pembayaran, cicilan, dan DP. Bila tidak ada alternatif, arahkan ke tab Rekomendasi.
- Prediksi ML tetap terpisah dari keputusan akhir dan memakai definisi dataset tanpa bunga, cicilan lain, dan anggaran DP. Status akhir tidak berubah jika model gagal dimuat.
- CSV hasil simulasi memuat status, alasan, batas, selisih, dan penghasilan minimum. Total 24 unit test lulus; rendering browser Streamlit belum diuji karena paket Streamlit tidak tersedia di lingkungan pengujian.

## Pembaruan V3: UI dan transparansi algoritma

- Formulir utama di samping hasil. Simulasi hanya berjalan setelah tombol **Jalankan simulasi** ditekan. Hasil tetap mengikuti input terakhir yang dikirim; mengedit formulir belum mengubah hasil.
- Rekomendasi hanya berjalan setelah **Cari rekomendasi**. Simulasi baru menghapus hasil pencarian sebelumnya. Reset menghapus hasil dan input sesi.
- Tab Proses & diagnostik menampilkan nilai fitur, transformasi RobustScaler, koefisien dan kontribusi Linear Regression, jalur node Decision Tree aktual, pembandingan ML terhadap rumus dataset, serta peringatan rentang training dan detail error. Kalkulator tetap berjalan apabila model gagal dimuat.
- Tabel rekomendasi dengan kolom numerik yang dapat diurutkan, grafik, peringkat dan jumlah kandidat yang lolos tiap batas.
- Tab penelitian memiliki filter algoritma/skenario serta EDA kategorikal interaktif berdasarkan 60.016 data development dan grafik dari notebook terbaru. Model dan hasil training V2 tidak diganti.
- Tema Streamlit di `.streamlit/config.toml`. Pastikan folder tersembunyi ikut diunggah jika menggunakan upload browser.

## Fitur aplikasi

1. Simulasi tanpa bunga sesuai dataset atau bunga flat berdasarkan asumsi pengguna.
2. Rasio seluruh cicilan, batas kemampuan bayar, anggaran DP, total pembayaran dan sisa penghasilan.
3. Prediksi ML ditampilkan terpisah dari keputusan simulasi; peringatan input di luar rentang training.
4. Rekomendasi maksimal lima mobil unik berdasarkan batas rasio 35%, anggaran DP, skema pembiayaan dan preferensi ranking.
5. Tab proses dan diagnostik serta hasil penelitian, tabel cross-validation, uji generalisasi, dan grafik EDA.

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
2. Ekstrak ZIP dan unggah isi direktori `predcicilanbelimobil-v3/` (bukan ZIP atau direktori pembungkus) ke root repo. Pertahankan susunan folder. Jangan unggah dataset mentah.
3. Buka https://share.streamlit.io/ dan hubungkan akun GitHub. Pilih **Create app** dan opsi deploy dari repo yang sudah ada.
4. Isi repository `USERNAME/predcicilanbelimobil-v3`, branch `main`, main file path `app.py`.
5. Di Advanced settings pilih Python **3.13**, simpan lalu deploy. Tidak membutuhkan secrets.
6. Periksa empat tab dan lakukan simulasi dengan berbagai input. Commit perubahan ke repo untuk memperbarui aplikasi.

## Training ulang (opsional, tidak diperlukan untuk deploy)

Dataset mentah tidak disertakan karena mengandung identitas/kontak. Gunakan salinan asli secara lokal dan pastikan hak penggunaan data. Jalankan notebook `notebooks/car_financing_analysis.ipynb` dari root repository, dengan dataset `indonesia_car_sales_bigdata.csv` di current working directory. Untuk Colab, unggah notebook dan CSV ke `/content`. Instal versi library sesuai `notebooks/requirements-training.txt` bila diperlukan. Notebook menulis `outputs_car_financing/`.

Setelah training ulang, salin `model_cicilan.pkl`, `model_kelayakan.pkl`, `model_metadata.json` ke `models/`, `katalog_mobil.csv` ke `data/`, `features.py` ke `src/`, seluruh laporan CSV/JSON evaluasi dan folder `figures/` ke `reports/`. Sinkronkan versi dependency dengan metadata model. Jangan mencampur model, metadata, dan feature engineering dari training berbeda. Muat PKL hanya dari sumber yang dipercaya.

## Pengujian

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py src tests
```

17 unit test lulus pada Python 3.13.5, termasuk 9 pengujian sebelumnya dan 8 pengujian workflow baru. Pengujian mencakup rekonstruksi prediksi linear, node tree, fallback ketika model gagal, konsistensi mode bunga, dan statistik rekomendasi. Sintaks Python diperiksa. Paket Streamlit tidak tersedia pada lingkungan pembuatan ZIP sehingga UI browser dan deployment cloud belum diuji langsung.

## Struktur

```text
app.py                         # UI Streamlit
requirements.txt               # dependensi aplikasi
models/                        # PKL dan metadata training
data/katalog_mobil.csv          # katalog tanpa identitas pribadi
src/                           # feature engineering, inferensi, kalkulator, rekomendasi
notebooks/                     # penelitian dan dependensi training
reports/                       # metrik, audit data, grafik
tests/                         # test_system dan test_workflow
.streamlit/config.toml         # tema aplikasi
```

## Pemeriksaan UI setelah deploy

1. Buka aplikasi: hasil harus kosong sebelum klik Jalankan simulasi.
2. Gunakan nilai default Ayla, DP 25%, tenor 48 bulan, penghasilan Rp15 juta: cicilan Rp2.656.250, rasio 17,71%, sesuai batas simulasi.
3. Ubah penghasilan menjadi Rp5 juta tanpa submit: hasil lama tetap tampil. Klik Jalankan simulasi: rasio 53,13%, melebihi batas. Hasil rekomendasi lama harus hilang.
4. Buka Proses & diagnostik: nilai fitur, kontribusi regresi dan jalur node tree harus muncul. Untuk input Rp1 juta, peringatan rentang training harus muncul.
5. Buka Rekomendasi, tekan Cari rekomendasi; tabel atau pesan hasil kosong muncul bersama jumlah filter. Anggaran DP Rp0 menghasilkan nol alternatif.
6. Pilih bunga flat 5% dengan input default: cicilan Rp3.187.500; prediksi ML tanpa bunga tetap sama.
7. Buka Hasil penelitian, ganti skenario dan kategori; grafik dan tabel harus diperbarui. Periksa unduhan CSV dan tombol reset.
