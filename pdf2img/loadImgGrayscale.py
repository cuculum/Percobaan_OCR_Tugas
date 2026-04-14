# Load Img + Grayscale
import cv2

img = cv2.imread("page_0.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Denoising (Hilangkan Noise)
denoised = cv2.fastNlMeansDenoising(gray, h=30)

# adaptive treshold
thresh = cv2.adaptiveThreshold(
    denoised,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

# luruskan gambar
import numpy as np

coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle

(h, w) = thresh.shape[:2]
center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, angle, 1.0)
deskew = cv2.warpAffine(thresh, M, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE)

# Perjelas Gambar
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
processed = cv2.morphologyEx(deskew, cv2.MORPH_CLOSE, kernel)

# Simpan Hasil
cv2.imwrite("processed.png", processed)

# Hilangkan Tabel
# deteksi garis horizontal
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)

# kurangi dari gambar utama
clean = cv2.subtract(thresh, remove_horizontal)