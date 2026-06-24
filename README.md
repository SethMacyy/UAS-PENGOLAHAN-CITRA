# 🖼️ Aplikasi Pengolahan Citra Digital
### Project UAS - Mata Kuliah Pengolahan Citra Digital
**Universitas Pelita Bangsa | Fakultas Teknik | Teknik Informatika**

---

## 📋 Deskripsi Project

Aplikasi pengolahan citra digital berbasis Python dengan GUI Tkinter yang mampu melakukan berbagai teknik pemrosesan gambar. Project ini menggunakan dataset 20 gambar buah-buahan tropis sebagai bahan uji coba implementasi algoritma pengolahan citra.

---

## 👨‍💻 Identitas

| Keterangan | Detail |
|---|---|
| Nama | Surya Putra Darma Jaya |
| NIM | 312410405 |
| Mata Kuliah | Pengolahan Citra Digital |
| Dosen Pengampu | Dr. Muhamad Fatchan, S.Kom., M.Kom |
| Kelas | I241C |
| Program Studi | Teknik Informatika |
| Universitas | Universitas Pelita Bangsa |

---

## 🗂️ Struktur Project

```
UAS/
├── pengolahan_citra.py     # Aplikasi utama dengan GUI Tkinter
├── batch_processor.py      # Script batch untuk proses semua dataset
├── requirements.txt        # Daftar library yang dibutuhkan
├── foto/                   # Dataset 20 gambar buah-buahan
│   ├── alpukat.jpg
│   ├── anggur.png
│   ├── apel.jpg
│   ├── apel hijau.webp
│   ├── buah naga.jpg
│   ├── delima.jpg
│   ├── durian.jpg
│   ├── jambu biji.jpg
│   ├── jeruk.jpg
│   ├── kiwi.jpg
│   ├── mangga.jpg
│   ├── manggis.jpg
│   ├── melon.jpeg
│   ├── nanas.jpg
│   ├── pepaya.jpg
│   ├── pir.jpg
│   ├── pisang.webp
│   ├── rambutan.jpg
│   ├── semangka.jpg
│   └── stroberi.jpg
└── hasil/                  # Output hasil pengolahan (auto-generated)
    ├── 1_grayscale/
    ├── 2_biner/
    ├── 3_histogram_eq/
    ├── 4_contrast_stretching/
    ├── 5_brightness/
    ├── 6_sharpening/
    ├── 7_mean_filter/
    ├── 8_median_filter/
    ├── 9_gaussian_filter/
    ├── 10_sobel/
    ├── 11_canny/
    ├── 12_prewitt/
    ├── 13_kmeans/
    └── 14_thresholding/
```

---

## 🍎 Dataset

Dataset terdiri dari **20 gambar buah-buahan tropis** dengan format JPG, PNG, dan WEBP:

| No | Nama Buah | No | Nama Buah |
|---|---|---|---|
| 1 | Alpukat | 11 | Mangga |
| 2 | Anggur | 12 | Manggis |
| 3 | Apel Merah | 13 | Melon |
| 4 | Apel Hijau | 14 | Nanas |
| 5 | Buah Naga | 15 | Pepaya |
| 6 | Delima | 16 | Pir |
| 7 | Durian | 17 | Pisang |
| 8 | Jambu Biji | 18 | Rambutan |
| 9 | Jeruk | 19 | Semangka |
| 10 | Kiwi | 20 | Stroberi |

---

## ⚙️ Fitur Pengolahan Citra

### 1. 🎨 Konversi Citra
- **Grayscale** — Konversi RGB ke skala abu-abu
- **Biner (Otsu)** — Konversi ke citra hitam-putih dengan thresholding otomatis

### 2. ✨ Perbaikan Kualitas
- **Histogram Equalization** — Pemerataan distribusi intensitas piksel
- **Contrast Stretching** — Peregangan kontras menggunakan percentile 2-98
- **Brightness Adjustment** — Penambahan kecerahan (+50)
- **Sharpening** — Penajaman citra dengan Unsharp Masking

### 3. 🔍 Filtering
- **Mean Filter** — Filter rata-rata dengan kernel 5×5
- **Median Filter** — Filter median untuk reduksi noise
- **Gaussian Filter** — Filter Gaussian untuk pelembutan citra

### 4. 📐 Deteksi Tepi
- **Sobel** — Deteksi tepi menggunakan operator Sobel
- **Canny** — Deteksi tepi dengan algoritma Canny
- **Prewitt** — Deteksi tepi menggunakan operator Prewitt

### 5. 🧩 Segmentasi Citra
- **Thresholding** — Segmentasi berbasis nilai ambang Otsu
- **K-Means Segmentation** — Pengelompokan piksel dengan K-Means (k=3)
- **Watershed Segmentation** — Segmentasi berbasis watershed

### ⭐ Bonus
- **Deteksi Wajah Haar Cascade** — Deteksi wajah otomatis menggunakan OpenCV
- **GUI Tkinter** — Antarmuka grafis interaktif

---

## 🚀 Cara Instalasi & Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/username/pengolahan-citra-digital.git
cd pengolahan-citra-digital
```

### 2. Buat Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi GUI
```bash
python pengolahan_citra.py
```

### 5. Jalankan Batch Processor (opsional)
```bash
python batch_processor.py
```
> Akan otomatis memproses semua 20 foto di folder `foto/` dan menyimpan hasilnya ke folder `hasil/`

---

## 📦 Dependencies

```
opencv-python>=4.8.0
Pillow>=10.0.0
scikit-image>=0.21.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.11.0
```

---

## 🖥️ Tampilan Aplikasi

Aplikasi memiliki tampilan GUI dengan tema gelap yang terdiri dari:
- **Panel kiri** — Menu operasi pengolahan citra
- **Panel tengah** — Tampilan gambar asli
- **Panel kanan** — Tampilan hasil pengolahan
- **Status bar** — Informasi proses yang sedang berjalan

---

## 📊 Hasil Pengolahan

Batch processor menghasilkan **280 gambar output** (20 foto × 14 metode) yang tersimpan otomatis ke dalam subfolder `hasil/` berdasarkan jenis operasinya.

---

## 🛠️ Teknologi yang Digunakan

| Library | Kegunaan |
|---|---|
| OpenCV | Pemrosesan citra utama |
| Pillow (PIL) | Tampilan gambar di GUI |
| Scikit-image | Algoritma pengolahan citra lanjutan |
| NumPy | Operasi matriks dan array |
| Matplotlib | Visualisasi histogram |
| SciPy | Konvolusi filter Prewitt |
| Tkinter | Antarmuka grafis (GUI) |

---

## 📄 Lisensi

Project ini dibuat untuk keperluan akademik — UAS Mata Kuliah Pengolahan Citra Digital, Universitas Pelita Bangsa.
