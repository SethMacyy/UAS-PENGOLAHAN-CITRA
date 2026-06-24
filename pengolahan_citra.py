"""
pip install opencv-python pillow scikit-image numpy matplotlib scipy
python pengolahan_citra.py
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import exposure, filters, feature, segmentation, color
from skimage.filters import threshold_otsu
from scipy import ndimage


# ─────────────────────────────────────────
# 1. KONVERSI CITRA
# ─────────────────────────────────────────

def convert_grayscale(image_bgr):
    """Konversi RGB ke Grayscale menggunakan OpenCV."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def convert_binary(image_bgr, threshold=127):
    """Konversi RGB ke Biner (Otsu thresholding)."""
    gray = convert_grayscale(image_bgr)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ─────────────────────────────────────────
# 2. PERBAIKAN KUALITAS CITRA
# ─────────────────────────────────────────

def histogram_equalization(image_bgr):
    """Histogram Equalization pada channel Y (YCrCb)."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def contrast_stretching(image_bgr):
    """Contrast Stretching dengan percentile 2-98."""
    img_float = image_bgr.astype(np.float64)
    p2, p98 = np.percentile(img_float, (2, 98))
    stretched = np.clip((img_float - p2) / (p98 - p2 + 1e-6) * 255, 0, 255)
    return stretched.astype(np.uint8)


def brightness_adjustment(image_bgr, beta=50):
    """Brightness Adjustment (additive)."""
    return cv2.convertScaleAbs(image_bgr, alpha=1.0, beta=beta)


def sharpening(image_bgr):
    """Sharpening menggunakan Unsharp Masking."""
    kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
    return cv2.filter2D(image_bgr, -1, kernel)


# ─────────────────────────────────────────
# 3. FILTERING
# ─────────────────────────────────────────

def mean_filter(image_bgr, ksize=5):
    """Mean Filter (Box Filter)."""
    return cv2.blur(image_bgr, (ksize, ksize))


def median_filter(image_bgr, ksize=5):
    """Median Filter."""
    return cv2.medianBlur(image_bgr, ksize)


def gaussian_filter(image_bgr, ksize=5, sigma=1.0):
    """Gaussian Filter."""
    return cv2.GaussianBlur(image_bgr, (ksize, ksize), sigma)


# ─────────────────────────────────────────
# 4. DETEKSI TEPI
# ─────────────────────────────────────────

def sobel_edge(image_bgr):
    """Deteksi tepi Sobel."""
    gray = convert_grayscale(image_bgr)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def canny_edge(image_bgr, low=50, high=150):
    """Deteksi tepi Canny."""
    gray = convert_grayscale(image_bgr)
    return cv2.Canny(gray, low, high)


def prewitt_edge(image_bgr):
    """Deteksi tepi Prewitt menggunakan kernel manual."""
    gray = convert_grayscale(image_bgr).astype(np.float64)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    ex = ndimage.convolve(gray, kx)
    ey = ndimage.convolve(gray, ky)
    magnitude = np.sqrt(ex**2 + ey**2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ─────────────────────────────────────────
# 5. SEGMENTASI CITRA
# ─────────────────────────────────────────

def threshold_segmentation(image_bgr):
    """Segmentasi Thresholding (Otsu)."""
    gray = convert_grayscale(image_bgr)
    thresh_val = threshold_otsu(gray)
    binary = (gray > thresh_val).astype(np.uint8) * 255
    return binary


def kmeans_segmentation(image_bgr, k=3):
    """K-Means Segmentation."""
    data = image_bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(image_bgr.shape)
    return segmented


def watershed_segmentation(image_bgr):
    """Watershed Segmentation."""
    gray = convert_grayscale(image_bgr)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    img_copy = image_bgr.copy()
    cv2.watershed(img_copy, markers)
    img_copy[markers == -1] = [0, 0, 255]
    return img_copy


# ─────────────────────────────────────────
# 6. BONUS: DETEKSI WAJAH HAAR CASCADE
# ─────────────────────────────────────────

def detect_faces(image_bgr):
    """Deteksi wajah menggunakan Haar Cascade."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = convert_grayscale(image_bgr)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    result = image_bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result, 'Wajah', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return result, len(faces)


# ─────────────────────────────────────────
# 7. BONUS: GUI TKINTER
# ─────────────────────────────────────────

class AplikasiPengolahanCitra:
    def __init__(self, root):
        self.root = root
        self.root.title("Pengolahan Citra Digital - Universitas Pelita Bangsa")
        self.root.geometry("1280x780")
        self.root.configure(bg="#1e1e2e")

        self.image_bgr = None
        self.result_bgr = None
        self.image_path = None

        self._build_ui()

    # ── Layout ──────────────────────────────

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#313244", pady=8)
        header.pack(fill=tk.X)
        tk.Label(header, text="🖼  Aplikasi Pengolahan Citra Digital",
                 font=("Helvetica", 16, "bold"), fg="#cdd6f4", bg="#313244").pack()
        tk.Label(header, text="Universitas Pelita Bangsa | Teknik Informatika",
                 font=("Helvetica", 9), fg="#a6adc8", bg="#313244").pack()

        # Main area
        main = tk.Frame(self.root, bg="#1e1e2e")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # Left: controls
        ctrl = tk.Frame(main, bg="#181825", width=260)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        ctrl.pack_propagate(False)
        self._build_controls(ctrl)

        # Right: image panels
        display = tk.Frame(main, bg="#1e1e2e")
        display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_display(display)

        # Status bar
        self.status_var = tk.StringVar(value="Selamat datang! Buka gambar untuk memulai.")
        status = tk.Label(self.root, textvariable=self.status_var,
                          bg="#313244", fg="#a6e3a1", anchor=tk.W, padx=8)
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_controls(self, parent):
        def section(text):
            tk.Label(parent, text=text, font=("Helvetica", 9, "bold"),
                     fg="#89b4fa", bg="#181825").pack(fill=tk.X, padx=8, pady=(12, 2))
            tk.Frame(parent, bg="#313244", height=1).pack(fill=tk.X, padx=8)

        btn_style = dict(bg="#313244", fg="#cdd6f4", relief=tk.FLAT,
                         activebackground="#45475a", activeforeground="#cdd6f4",
                         cursor="hand2", pady=4)

        # File
        section("📁 File")
        tk.Button(parent, text="Buka Gambar", command=self.open_image, **btn_style).pack(fill=tk.X, padx=8, pady=2)
        tk.Button(parent, text="Simpan Hasil", command=self.save_result, **btn_style).pack(fill=tk.X, padx=8, pady=2)
        tk.Button(parent, text="Reset", command=self.reset_image, **btn_style).pack(fill=tk.X, padx=8, pady=2)

        # Konversi
        section("🎨 Konversi")
        tk.Button(parent, text="→ Grayscale", command=lambda: self.apply("grayscale"), **btn_style).pack(fill=tk.X, padx=8, pady=2)
        tk.Button(parent, text="→ Biner (Otsu)", command=lambda: self.apply("binary"), **btn_style).pack(fill=tk.X, padx=8, pady=2)

        # Kualitas
        section("✨ Perbaikan Kualitas")
        tk.Button(parent, text="Histogram Equalization", command=lambda: self.apply("histeq"), **btn_style).pack(fill=tk.X, padx=8, pady=2)
        tk.Button(parent, text="Contrast Stretching", command=lambda: self.apply("contrast"), **btn_style).pack(fill=tk.X, padx=8, pady=2)
        tk.Button(parent, text="Brightness +50", command=lambda: self.apply("bright"), **btn_style).pack(fill=tk.X, padx=8, pady=2)
        tk.Button(parent, text="Sharpening", command=lambda: self.apply("sharp"), **btn_style).pack(fill=tk.X, padx=8, pady=2)

        # Filter
        section("🔍 Filtering")
        tk.Button(parent, text="Mean Filter", command=lambda: self.apply("mean"), **btn_style).pack(fill=tk.X, padx=8, pady=2)
        tk.Button(parent, text="Median Filter", command=lambda: self.apply("median"), **btn_style).pack(fill=tk.X, padx=8, pady=2)
        tk.Button(parent, text="Gaussian Filter", command=lambda: self.apply("gaussian"), **btn_style).pack(fill=tk.X, padx=8, pady=2)

        # Deteksi Tepi
        section("📐 Deteksi Tepi")
        tk.Button(parent, text="Sobel", command=lambda: self.apply("sobel"), **btn_style).pack(fill=tk.X, padx=8, pady=2)
        tk.Button(parent, text="Canny", command=lambda: self.apply("canny"), **btn_style).pack(fill=tk.X, padx=8, pady=2)
        tk.Button(parent, text="Prewitt", command=lambda: self.apply("prewitt"), **btn_style).pack(fill=tk.X, padx=8, pady=2)

        # Segmentasi
        section("🧩 Segmentasi")
        tk.Button(parent, text="Thresholding", command=lambda: self.apply("thresh_seg"), **btn_style).pack(fill=tk.X, padx=8, pady=2)
        tk.Button(parent, text="K-Means (k=3)", command=lambda: self.apply("kmeans"), **btn_style).pack(fill=tk.X, padx=8, pady=2)
        tk.Button(parent, text="Watershed", command=lambda: self.apply("watershed"), **btn_style).pack(fill=tk.X, padx=8, pady=2)

        # Bonus
        section("⭐ Bonus: Deteksi Wajah")
        tk.Button(parent, text="Haar Cascade", command=lambda: self.apply("face"), **btn_style).pack(fill=tk.X, padx=8, pady=2)
        tk.Button(parent, text="Tampilkan Histogram", command=self.show_histogram, **btn_style).pack(fill=tk.X, padx=8, pady=(2, 12))

    def _build_display(self, parent):
        panel_frame = tk.Frame(parent, bg="#1e1e2e")
        panel_frame.pack(fill=tk.BOTH, expand=True)

        # Original
        left = tk.Frame(panel_frame, bg="#181825", bd=1, relief=tk.FLAT)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        tk.Label(left, text="Gambar Asli", fg="#89b4fa", bg="#181825",
                 font=("Helvetica", 10, "bold")).pack(pady=4)
        self.canvas_orig = tk.Label(left, bg="#11111b", text="[Belum ada gambar]",
                                    fg="#585b70")
        self.canvas_orig.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Result
        right = tk.Frame(panel_frame, bg="#181825", bd=1, relief=tk.FLAT)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.result_label_var = tk.StringVar(value="Hasil")
        tk.Label(right, textvariable=self.result_label_var, fg="#a6e3a1", bg="#181825",
                 font=("Helvetica", 10, "bold")).pack(pady=4)
        self.canvas_result = tk.Label(right, bg="#11111b", text="[Belum diproses]",
                                      fg="#585b70")
        self.canvas_result.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ── Helpers ─────────────────────────────

    def _display(self, label_widget, img_bgr_or_gray, max_w=560, max_h=600):
        if len(img_bgr_or_gray.shape) == 2:
            img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        pil.thumbnail((max_w, max_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil)
        label_widget.configure(image=tk_img, text="")
        label_widget.image = tk_img

    def _require_image(self):
        if self.image_bgr is None:
            messagebox.showwarning("Peringatan", "Buka gambar terlebih dahulu!")
            return False
        return True

    # ── Actions ─────────────────────────────

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All", "*.*")]
        )
        if not path:
            return
        self.image_path = path
        self.image_bgr = cv2.imread(path)
        if self.image_bgr is None:
            messagebox.showerror("Error", "Gagal membaca gambar!")
            return
        h, w = self.image_bgr.shape[:2]
        self._display(self.canvas_orig, self.image_bgr)
        self.canvas_result.configure(image="", text="[Belum diproses]")
        self.result_bgr = None
        self.status_var.set(f"✔ Gambar dibuka: {os.path.basename(path)}  |  Ukuran: {w}×{h} px")

    def save_result(self):
        if self.result_bgr is None:
            messagebox.showwarning("Peringatan", "Belum ada hasil untuk disimpan!")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")]
        )
        if path:
            cv2.imwrite(path, self.result_bgr)
            self.status_var.set(f"✔ Hasil disimpan: {os.path.basename(path)}")

    def reset_image(self):
        if not self._require_image():
            return
        self._display(self.canvas_orig, self.image_bgr)
        self.canvas_result.configure(image="", text="[Belum diproses]")
        self.result_bgr = None
        self.status_var.set("↩ Reset ke gambar asli.")

    def apply(self, method):
        if not self._require_image():
            return

        img = self.image_bgr
        label = method.upper()

        ops = {
            "grayscale":  (lambda: convert_grayscale(img),          "Grayscale"),
            "binary":     (lambda: convert_binary(img),              "Biner (Otsu)"),
            "histeq":     (lambda: histogram_equalization(img),      "Histogram Equalization"),
            "contrast":   (lambda: contrast_stretching(img),         "Contrast Stretching"),
            "bright":     (lambda: brightness_adjustment(img, 50),   "Brightness +50"),
            "sharp":      (lambda: sharpening(img),                  "Sharpening"),
            "mean":       (lambda: mean_filter(img),                 "Mean Filter"),
            "median":     (lambda: median_filter(img),               "Median Filter"),
            "gaussian":   (lambda: gaussian_filter(img),             "Gaussian Filter"),
            "sobel":      (lambda: sobel_edge(img),                  "Deteksi Tepi Sobel"),
            "canny":      (lambda: canny_edge(img),                  "Deteksi Tepi Canny"),
            "prewitt":    (lambda: prewitt_edge(img),                "Deteksi Tepi Prewitt"),
            "thresh_seg": (lambda: threshold_segmentation(img),      "Segmentasi Thresholding"),
            "kmeans":     (lambda: kmeans_segmentation(img),         "K-Means Segmentation"),
            "watershed":  (lambda: watershed_segmentation(img),      "Watershed Segmentation"),
        }

        if method == "face":
            result, n = detect_faces(img)
            self.result_bgr = result
            self.result_label_var.set(f"Deteksi Wajah ({n} wajah ditemukan)")
            self._display(self.canvas_result, result)
            self.status_var.set(f"✔ Deteksi Wajah selesai — {n} wajah ditemukan.")
            return

        if method not in ops:
            return

        fn, label_text = ops[method]
        result = fn()

        # Simpan hasil (pastikan BGR untuk OpenCV)
        if len(result.shape) == 2:
            self.result_bgr = result
        else:
            self.result_bgr = result

        self.result_label_var.set(label_text)
        self._display(self.canvas_result, result)
        self.status_var.set(f"✔ {label_text} berhasil diterapkan.")

    def show_histogram(self):
        if not self._require_image():
            return
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor("#1e1e2e")

        def plot_hist(ax, img, title, color):
            ax.set_facecolor("#181825")
            ax.set_title(title, color="#cdd6f4")
            ax.tick_params(colors="#cdd6f4")
            for spine in ax.spines.values():
                spine.Sset_edgecolor("#313244")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            ax.hist(gray.ravel(), 256, [0, 256], color=color, alpha=0.85)
            ax.set_xlim([0, 256])

        plot_hist(axes[0], self.image_bgr, "Histogram Asli", "#89b4fa")
        if self.result_bgr is not None:
            plot_hist(axes[1], self.result_bgr, "Histogram Hasil", "#a6e3a1")
        else:
            axes[1].set_visible(False)

        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = AplikasiPengolahanCitra(root)
    root.mainloop()
