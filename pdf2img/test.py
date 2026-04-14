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

            curr_y = data["top"][i]
            curr_x = data["left"][i]      
            
            # Cari apakah teks ini masuk ke baris yang sudah ada
            found_row = False
            for row in rows:

                #Jika jarak Y teks ini dengan rata-rata Y di baris terebut kecil (misal < 20px)
                if abs(curr_y - row['avg_y']) < 20:
                    row['elements'].append((curr_x, text))

                    # Update rata-rata Y baris agar akurat
                    row['avg_y'] = (row['avg_y'] + curr_y) / 2
                    found_row = True
                    break                   
            
            # Jika tidak masuk baris manapun, buar garis baru
            if not found_row:
                rows.append({'avg_y': curr_y, 'elements': [(curr_x, text)]})

        #Konversi list 'rows' menjadi format tabel untuk DataFrame
        table = []
        for row in rows:

            #Urutkan elemen dalam baris berdasarkan posisi X (kiri ke kanan)
            sorted_cols = sorted(row['elements'], key=lambda x: x[0])
            table.append([text for _, text in sorted_cols])

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