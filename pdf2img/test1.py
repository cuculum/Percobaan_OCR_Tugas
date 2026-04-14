import cv2
import numpy as np
import os

# load image
img = cv2.imread("page_0.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold (binary inverse)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# kernel untuk deteksi garis
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))

# deteksi garis horizontal
horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)

# deteksi garis vertical
vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

# gabungkan garis
table_mask = cv2.add(horizontal_lines, vertical_lines)

# cari contour (area tabel)
contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# folder output
os.makedirs("tables", exist_ok=True)

count = 0

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # filter ukuran kecil (biar bukan noise)
    if w > 300 and h > 200:
        table = img[y:y+h, x:x+w]

        cv2.imwrite(f"tables/table_{count}.png", table)
        count += 1

        # optional: gambar kotak di original image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

# simpan preview hasil deteksi
cv2.imwrite("detected_tables.png", img)

print(f"{count} tabel terdeteksi")