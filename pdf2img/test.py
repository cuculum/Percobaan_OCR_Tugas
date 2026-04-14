import cv2
import numpy as np
import pytesseract
import pandas as pd
import os

# path tesseract (WAJIB di Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\test program\tesseract\tesseract.exe"

# load image
img = cv2.imread("page_0.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# detect garis
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))

horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

table_mask = cv2.add(horizontal, vertical)

# cari kontur tabel
contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

os.makedirs("tables", exist_ok=True)

all_tables = []

count = 0

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    if w > 300 and h > 200:
        table_img = img[y:y+h, x:x+w]

        # =====================
        # PREPROCESSING
        # =====================
        gray_table = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)

        thresh_table = cv2.adaptiveThreshold(
            gray_table, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        # perbesar
        resized = cv2.resize(thresh_table, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # =====================
        # OCR
        # =====================
        data = pytesseract.image_to_data(resized, output_type=pytesseract.Output.DICT)

        rows = {}

        for i in range(len(data["text"])):
            text = data["text"][i].strip()

            if text == "":
                continue

            y_pos = data["top"][i]
            row_key = y_pos // 20

            if row_key not in rows:
                rows[row_key] = []

            rows[row_key].append((data["left"][i], text))

        table = []

        for row in sorted(rows.keys()):
            cols = sorted(rows[row], key=lambda x: x[0])
            table.append([text for _, text in cols])

        df = pd.DataFrame(table)

        # simpan per tabel
        df.to_excel(f"tabel/table_{count}.xlsx", index=False)

        all_tables.append(df)

        count += 1

# gabungkan semua tabel (opsional)
if all_tables:
    final_df = pd.concat(all_tables, ignore_index=True)
    final_df.to_excel("final_output.xlsx", index=False)

print(f"{count} tabel berhasil diproses")