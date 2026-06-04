import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# BAB 1
from processing.grayscale import (
    convert_to_grayscale,
    red_channel,
    green_channel,
    blue_channel
)
# BAB 2
from processing.arithmetic import (
    adjust_brightness,
    adjust_contrast,
    negative as negative_image
)
# BAB 3
from processing.histogram import (
    calculate_histogram_gray,
    calculate_histogram_rgb,
    equalize_histogram
)
# BAB 4
from processing.filtering import (
    mean_filter,
    gaussian_filter,
    sharpen,
    sobel_edge_detection
)
# BAB 5
from processing.threshold import (
    threshold_manual,
    threshold_otsu
)


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("APLIKASI PENGOLAHAN CITRA DIGITAL")
        self.root.geometry("1200x700")

        self.original_image = None
        self.processed_image = None
        self.original_photo = None
        self.processed_photo = None

        self.display_width = 450
        self.display_height = 450

        self.setup_ui()
        self.root.update()
        self.calculate_display_size()

    def setup_ui(self):
        header_frame = ttk.Frame(self.root)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header_frame, text="APLIKASI PENGOLAHAN CITRA DIGITAL",
                  font=("Arial", 16, "bold")).pack(pady=10)

        self.create_toolbar()
        main_frame = ttk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.create_sidebar(main_frame)
        self.create_image_area(main_frame)
        self.create_statusbar()

    def create_toolbar(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.btn_open = ttk.Button(toolbar, text="Buka Gambar", command=self.open_image)
        self.btn_open.pack(side=tk.LEFT, padx=2)
        self.btn_save = ttk.Button(toolbar, text="Simpan Hasil", command=self.save_result)
        self.btn_save.pack(side=tk.LEFT, padx=2)
        self.btn_reset = ttk.Button(toolbar, text="Reset", command=self.reset)
        self.btn_reset.pack(side=tk.LEFT, padx=2)

    def create_sidebar(self, parent):
        sidebar = ttk.Frame(parent, width=250)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        sidebar.pack_propagate(False)

        # BAB 1
        bab1 = ttk.LabelFrame(sidebar, text="BAB 1")
        bab1.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(bab1, text="Grayscale", command=self.grayscale).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bab1, text="Red Channel", command=self.red_channel).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bab1, text="Green Channel", command=self.green_channel).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bab1, text="Blue Channel", command=self.blue_channel).pack(fill=tk.X, padx=5, pady=2)

        # BAB 2
        bab2 = ttk.LabelFrame(sidebar, text="BAB 2")
        bab2.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(bab2, text="Brightness +", command=self.brightness_plus).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bab2, text="Brightness -", command=self.brightness_minus).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bab2, text="Contrast +", command=self.contrast_plus).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bab2, text="Contrast -", command=self.contrast_minus).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bab2, text="Negative", command=self.negative).pack(fill=tk.X, padx=5, pady=2)

        # BAB 3
        bab3 = ttk.LabelFrame(sidebar, text="BAB 3")
        bab3.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(bab3, text="Histogram Gray", command=self.histogram_gray).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bab3, text="Histogram RGB", command=self.histogram_rgb).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bab3, text="Equalization", command=self.equalization).pack(fill=tk.X, padx=5, pady=2)

        # BAB 4
        bab4 = ttk.LabelFrame(sidebar, text="BAB 4")
        bab4.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(bab4, text="Mean Filter", command=self.mean_filter).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bab4, text="Gaussian Filter", command=self.gaussian_filter).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bab4, text="Sharpen", command=self.sharpen).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bab4, text="Sobel", command=self.sobel).pack(fill=tk.X, padx=5, pady=2)

        # BAB 5
        bab5 = ttk.LabelFrame(sidebar, text="BAB 5")
        bab5.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(bab5, text="Threshold Manual", command=self.threshold).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(bab5, text="Otsu Threshold", command=self.otsu).pack(fill=tk.X, padx=5, pady=2)

    def create_image_area(self, parent):
        self.image_area = ttk.Frame(parent)
        self.image_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.original_frame = ttk.LabelFrame(self.image_area, text="Gambar Asli")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.original_label = tk.Label(self.original_frame, bg="gray90")
        self.original_label.pack(expand=True)

        self.processed_frame = ttk.LabelFrame(self.image_area, text="Hasil Pengolahan")
        self.processed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.processed_label = tk.Label(self.processed_frame, bg="gray90")
        self.processed_label.pack(expand=True)

    def create_statusbar(self):
        statusbar = ttk.Frame(self.root)
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = ttk.Label(statusbar, text="Siap", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X)

    def calculate_display_size(self):
        if hasattr(self, 'image_area'):
            area_width = self.image_area.winfo_width()
            area_height = self.image_area.winfo_height()
            w = area_width // 2 - 30
            h = area_height - 30
            self.display_width = w if w > 100 else 450
            self.display_height = h if h > 100 else 450

    def image_loaded(self):
        if self.original_image is None:
            messagebox.showwarning("Peringatan", "Silakan buka gambar terlebih dahulu.")
            return False
        return True

    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                messagebox.showerror("Error", "Gagal membaca gambar.")
                return
            self.original_image = img_bgr
            self.processed_image = None
            self.processed_photo = None
            self.processed_label.config(image='')
            self.root.update()
            self.calculate_display_size()
            self.display_image(img_bgr, self.original_label)
            self.status_label.config(text=f"Gambar dibuka: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan: {e}")

    def display_image(self, image, label_widget):
        if len(image.shape) == 2:
            image_pil = Image.fromarray(image)
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(img_rgb)

        image_pil.thumbnail((self.display_width, self.display_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image_pil)

        if label_widget == self.original_label:
            self.original_photo = photo
        else:
            self.processed_photo = photo

        label_widget.config(image=photo)

    def show_processed(self, image):
        self.processed_image = image
        self.display_image(image, self.processed_label)

    # ===== BAB 1 =====
    def grayscale(self):
        if not self.image_loaded(): return
        try:
            result = convert_to_grayscale(self.original_image)
            self.show_processed(result)
            self.status_label.config(text="Grayscale berhasil diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Grayscale gagal: {e}")

    def red_channel(self):
        if not self.image_loaded(): return
        try:
            result = red_channel(self.original_image)
            self.show_processed(result)
            self.status_label.config(text="Red Channel diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Red Channel gagal: {e}")

    def green_channel(self):
        if not self.image_loaded(): return
        try:
            result = green_channel(self.original_image)
            self.show_processed(result)
            self.status_label.config(text="Green Channel diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Green Channel gagal: {e}")

    def blue_channel(self):
        if not self.image_loaded(): return
        try:
            result = blue_channel(self.original_image)
            self.show_processed(result)
            self.status_label.config(text="Blue Channel diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Blue Channel gagal: {e}")

    # ===== BAB 2 =====
    def brightness_plus(self):
        if not self.image_loaded(): return
        try:
            result = adjust_brightness(self.original_image, value=50)
            self.show_processed(result)
            self.status_label.config(text="Brightness +50 diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Brightness + gagal: {e}")

    def brightness_minus(self):
        if not self.image_loaded(): return
        try:
            result = adjust_brightness(self.original_image, value=-50)
            self.show_processed(result)
            self.status_label.config(text="Brightness -50 diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Brightness - gagal: {e}")

    def contrast_plus(self):
        if not self.image_loaded(): return
        try:
            result = adjust_contrast(self.original_image, alpha=1.5)
            self.show_processed(result)
            self.status_label.config(text="Contrast + (1.5) diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Contrast + gagal: {e}")

    def contrast_minus(self):
        if not self.image_loaded(): return
        try:
            result = adjust_contrast(self.original_image, alpha=0.5)
            self.show_processed(result)
            self.status_label.config(text="Contrast - (0.5) diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Contrast - gagal: {e}")

    def negative(self):
        if not self.image_loaded(): return
        try:
            result = negative_image(self.original_image)
            self.show_processed(result)
            self.status_label.config(text="Negative diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Negative gagal: {e}")

    # ===== BAB 3 =====
    def histogram_gray(self):
        if not self.image_loaded(): return
        try:
            hist = calculate_histogram_gray(self.original_image)
            plt.figure("Histogram Grayscale")
            plt.bar(range(256), hist, color='gray')
            plt.title("Histogram Grayscale")
            plt.xlabel("Intensitas")
            plt.ylabel("Frekuensi")
            plt.show()
            self.status_label.config(text="Histogram Grayscale ditampilkan")
        except Exception as e:
            messagebox.showerror("Error", f"Histogram Gray gagal: {e}")

    def histogram_rgb(self):
        if not self.image_loaded(): return
        try:
            hist_b, hist_g, hist_r = calculate_histogram_rgb(self.original_image)
            plt.figure("Histogram RGB")
            plt.plot(hist_b, color='blue', label='Blue')
            plt.plot(hist_g, color='green', label='Green')
            plt.plot(hist_r, color='red', label='Red')
            plt.title("Histogram RGB")
            plt.xlabel("Intensitas")
            plt.ylabel("Frekuensi")
            plt.legend()
            plt.show()
            self.status_label.config(text="Histogram RGB ditampilkan")
        except Exception as e:
            messagebox.showerror("Error", f"Histogram RGB gagal: {e}")

    def equalization(self):
        if not self.image_loaded(): return
        try:
            result = equalize_histogram(self.original_image)
            self.show_processed(result)
            self.status_label.config(text="Histogram Equalization diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Equalization gagal: {e}")

    # ===== BAB 4 =====
    def mean_filter(self):
        if not self.image_loaded(): return
        try:
            result = mean_filter(self.original_image, kernel_size=15)
            self.show_processed(result)
            self.status_label.config(text="Mean Filter (15x15) diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Mean Filter gagal: {e}")

    def gaussian_filter(self):
        if not self.image_loaded(): return
        try:
            result = gaussian_filter(self.original_image, kernel_size=15, sigma=5.0)
            self.show_processed(result)
            self.status_label.config(text="Gaussian Filter (15x15, sigma=5) diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Gaussian Filter gagal: {e}")

    def sharpen(self):
        if not self.image_loaded(): return
        try:
            result = sharpen(self.original_image, strength=3.0)
            self.show_processed(result)
            self.status_label.config(text="Sharpen (strength=3) diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Sharpen gagal: {e}")

    def sobel(self):
        if not self.image_loaded(): return
        try:
            result = sobel_edge_detection(self.original_image)
            self.show_processed(result)
            self.status_label.config(text="Sobel Edge Detection diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Sobel gagal: {e}")

    # ===== BAB 5 =====
    def threshold(self):
        """Threshold manual dengan nilai tetap 127."""
        if not self.image_loaded(): return
        try:
            result = threshold_manual(self.original_image, thresh_value=150)
            self.show_processed(result)
            self.status_label.config(text="Threshold manual (127) diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Threshold manual gagal: {e}")

    def otsu(self):
        """Threshold Otsu otomatis."""
        if not self.image_loaded(): return
        try:
            result = threshold_otsu(self.original_image)
            self.show_processed(result)
            self.status_label.config(text="Threshold Otsu otomatis diterapkan")
        except Exception as e:
            messagebox.showerror("Error", f"Otsu threshold gagal: {e}")

    # ===== Toolbar =====
    def save_result(self):
        if self.processed_image is None:
            messagebox.showwarning("Peringatan", "Tidak ada hasil gambar untuk disimpan.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            self.status_label.config(text=f"Gambar disimpan ke {file_path}")

    def reset(self):
        if self.original_image is None:
            return
        self.processed_image = None
        self.processed_photo = None
        self.processed_label.config(image='')
        self.display_image(self.original_image, self.original_label)
        self.status_label.config(text="Reset ke gambar asli")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()