"""
=============================================================
BATCH PROCESSOR - Pengolahan Citra Digital
Otomatis proses semua foto di folder 'foto/'
dan simpan hasilnya ke folder 'hasil/'
=============================================================
Cara pakai:
  python batch_processor.py
=============================================================
"""

import cv2
import numpy as np
import os
from scipy import ndimage
from skimage.filters import threshold_otsu

# ── Folder input & output ──────────────────────────────────
INPUT_FOLDER  = "foto"
OUTPUT_FOLDER = "hasil"

EKSTENSI = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")

# ── Fungsi-fungsi pengolahan ───────────────────────────────

def convert_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def convert_binary(img):
    gray = convert_grayscale(img)
    _, biner = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return biner

def histogram_equalization(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def contrast_stretching(img):
    f = img.astype(np.float64)
    p2, p98 = np.percentile(f, (2, 98))
    stretched = np.clip((f - p2) / (p98 - p2 + 1e-6) * 255, 0, 255)
    return stretched.astype(np.uint8)

def brightness_adjustment(img, beta=50):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)

def sharpening(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def mean_filter(img):
    return cv2.blur(img, (5, 5))

def median_filter(img):
    return cv2.medianBlur(img, 5)

def gaussian_filter(img):
    return cv2.GaussianBlur(img, (5, 5), 1.0)

def sobel_edge(img):
    gray = convert_grayscale(img)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def canny_edge(img):
    gray = convert_grayscale(img)
    return cv2.Canny(gray, 50, 150)

def prewitt_edge(img):
    gray = convert_grayscale(img).astype(np.float64)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    ex = ndimage.convolve(gray, kx)
    ey = ndimage.convolve(gray, ky)
    mag = np.sqrt(ex**2 + ey**2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def kmeans_segmentation(img, k=3):
    data = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(img.shape)

def threshold_segmentation(img):
    gray = convert_grayscale(img)
    thresh_val = threshold_otsu(gray)
    binary = (gray > thresh_val).astype(np.uint8) * 255
    return binary

# ── Buat subfolder output ──────────────────────────────────

SUBFOLDER = [
    "1_grayscale", "2_biner", "3_histogram_eq", "4_contrast_stretching",
    "5_brightness", "6_sharpening", "7_mean_filter", "8_median_filter",
    "9_gaussian_filter", "10_sobel", "11_canny", "12_prewitt",
    "13_kmeans", "14_thresholding"
]

def buat_folder_output():
    for sub in SUBFOLDER:
        os.makedirs(os.path.join(OUTPUT_FOLDER, sub), exist_ok=True)

# ── Proses satu gambar ─────────────────────────────────────

def proses_gambar(path, nama_file):
    img = cv2.imread(path)
    if img is None:
        print(f"  [SKIP] Gagal baca: {nama_file}")
        return

    nama = os.path.splitext(nama_file)[0]

    def simpan(subfolder, hasil):
        out_path = os.path.join(OUTPUT_FOLDER, subfolder, nama + ".jpg")
        cv2.imwrite(out_path, hasil)

    simpan("1_grayscale",           convert_grayscale(img))
    simpan("2_biner",               convert_binary(img))
    simpan("3_histogram_eq",        histogram_equalization(img))
    simpan("4_contrast_stretching", contrast_stretching(img))
    simpan("5_brightness",          brightness_adjustment(img))
    simpan("6_sharpening",          sharpening(img))
    simpan("7_mean_filter",         mean_filter(img))
    simpan("8_median_filter",       median_filter(img))
    simpan("9_gaussian_filter",     gaussian_filter(img))
    simpan("10_sobel",              sobel_edge(img))
    simpan("11_canny",              canny_edge(img))
    simpan("12_prewitt",            prewitt_edge(img))
    simpan("13_kmeans",             kmeans_segmentation(img))
    simpan("14_thresholding",       threshold_segmentation(img))

    print(f"  [OK] {nama_file}")

# ── Main ───────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("BATCH PROCESSOR - Pengolahan Citra Digital")
    print("=" * 50)

    if not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] Folder '{INPUT_FOLDER}' tidak ditemukan!")
        return

    buat_folder_output()

    daftar_foto = [f for f in os.listdir(INPUT_FOLDER)
                   if f.lower().endswith(EKSTENSI)]

    if not daftar_foto:
        print(f"[ERROR] Tidak ada gambar di folder '{INPUT_FOLDER}'!")
        return

    print(f"\nDitemukan {len(daftar_foto)} foto. Memproses...\n")

    for i, nama_file in enumerate(daftar_foto, 1):
        print(f"[{i}/{len(daftar_foto)}] Memproses {nama_file}...")
        path = os.path.join(INPUT_FOLDER, nama_file)
        proses_gambar(path, nama_file)

    print("\n" + "=" * 50)
    print(f"SELESAI! Hasil tersimpan di folder '{OUTPUT_FOLDER}/'")
    print("=" * 50)

if __name__ == "__main__":
    main()