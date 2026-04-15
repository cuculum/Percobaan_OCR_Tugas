import os
import cv2
import re
import numpy as np
import pandas as pd
from paddleocr import PPStructure, save_structure_res

def run_advanced_ocr(image_path, output_dir="hasil_ekstraksi"):
    """
    Fungsi untuk mengekstraksi tabel dari dokumen menggunakan PP-Structure.
    """
    # 1. Loading
    print("[INFO] Memulai Proses: {image}")
    processed_img = processed_image(image)

    if processed_img is None:
        print(f"[ERROR] Gambar tidak ditemukan di: {image_path}")
        return

    # 2. Jalankan Analisis Struktur (Deteksi Tabel, Teks, Judul, dll)
    print("[INFO] Sedang menganalisis struktur dokumen...")
    result = TABLE_ENGINE(processed_img)

    # 3. Siapkan Folder Output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Folder khusus untuk menyimpan gambar hasil deteksi (visualisasi)
    vis_dir = os.path.join(output_dir, "visualisasi_deteksi")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    # 4. Proses Hasil Deteksi
    table_count = 0
    for i, region in enumerate(result):
        # Kita hanya fokus pada region yang terdeteksi sebagai 'table'
        if region['type'] == 'table':
            table_count += 1
            print(f"[INFO] Tabel ke-{table_count} ditemukan!")

            # PaddleOCR memberikan hasil tabel dalam format HTML
            html_str = region['res']['html']
            
            try:
                # Konversi HTML ke DataFrame
                df_list = pd.read_html(html_str)
                
                if df_list:
                    df = df_list[0]

                    # --- POST-PROCESSING (Pembersihan Data) ---
                    df = clean_dataframe(df)

                    # Simpan ke Excel
                    excel_name = f"tabel_ke_{table_count}.xlsx"
                    excel_path = os.path.join(output_dir, excel_name)
                    df.to_excel(excel_path, index=False)
                    print(f"    [SUCCESS] Tersimpan: {excel_name}")
                else:
                    print(f"    [WARNING] Tabel ke-{table_count} tidak memiliki data valid.")

            except Exception as e:
                print(f"    [ERROR] Gagal memproses tabel ke-{table_count}: {e}")

    # 6. Simpan Visualisasi
    # Gunakan gambar yang diproses agar visualisasi kotak deteksi terlihat tajam
    save_structure_res(result, processed_img, output_dir, vis_dir)
    
    print("\n" + "="*30)
    print(f"PROSES SELESAI")
    print(f"Total tabel ditemukan: {table_count}")
    print(f"Hasil Excel ada di folder: {output_dir}")
    print(f"Hasil visualisasi ada di: {os.path.join(output_dir, 'visualisasi_deteksi')}")
    print("="*30)

if __name__ == "__main__":
    # Ganti dengan nama file gambar Anda
    IMAGE_FOLDER = "image" 
    
     # Pastikan folder image ada
    if not os.path.exists(IMAGE_FOLDER):
        print(f"[ERROR] Folder {IMAGE_FOLDER} tidak ditemukan!")
    else:
        # Ambil semua file yang berakhiran .png di dalam folder tersebut
        all_images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png')]
        
        if not all_images:
            print(f"[INFO] Tidak ada file .png di folder {IMAGE_FOLDER}")
        else:
            print(f"[INFO] Ditemukan {len(all_images)} gambar. Mulai memproses...")
            
            # Loop untuk memproses setiap gambar satu per satu
            for filename in all_images:
                image_path = os.path.join(IMAGE_FOLDER, filename)
                
                # Membuat folder output unik untuk setiap gambar agar hasil tidak tertimpa
                # Contoh: image/table_0.png -> hasil_ekstraksi/table_0/
                custom_output_dir = os.path.join("hasil_ekstraksi", os.path.splitext(filename)[0])
                
                print(f"\n>>> Memproses file: {filename}")
                run_advanced_ocr(image_path, output_dir=custom_output_dir)
                
            print("\n" + "="*30)
            print("SEMUA FILE BERHASIL DIPROSES")
            print("="*30)