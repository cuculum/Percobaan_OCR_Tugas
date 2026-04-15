import os
import cv2
import re
import numpy as np
import pandas as pd
from paddleocr import PPStructure, save_structure_res


# 1. GLOBAL INITIALIZATION
# Model AI dimuat satu kali di awal agar eksekusi berikutnya jauh lebih cepat.
print("[INFO] Menginisialisasi AI Engine (PP-Structure)... Mohon tunggu.")
TABLE_ENGINE = PPStructure(show_log=False, image_orientation=True, lang='en')

# 2. SUPPORTING FUNCTIONS (Fungsi Pendukung)
def preprocess_image(image_path):
    """
    Meningkatkan kualitas gambar agar AI lebih akurat dalam mendeteksi tabel.
    - Denoising: Menghapus bintik noise.
    - CLAHE: Mempertajam kontras teks secara adaptif.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Menghilangkan noise 'salt and pepper'
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Konversi ke Grayscale
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    
    # Pertajam kontras secara lokal (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Kembalikan ke format BGR untuk kompatibilitas PaddleOCR
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def clean_dataframe(df):
    """
    Membersihkan data mentah dari AI agar menjadi tabel Excel yang rapi.
    - Menghapus baris/kolom kosong.
    - Menghapus karakter sampah OCR.
    - Mengonversi teks angka menjadi tipe numerik asli.
    """
    # Hapus baris dan kolom yang sepenuhnya kosong
    df = df.dropna(how='all').dropna(axis=1, how='all')

    # Bersihkan karakter sampah (simbol pembatas tabel yang terbaca salah)
    def clean_text(val):
        if isinstance(val, str):
            return re.sub(r'[|~_]', '', val).strip()
        return val

    df = df.applymap(clean_text)

    # Smart Conversion: Ubah string angka menjadi tipe numerik (float/int)
    for col in df.columns:
        converted_col = pd.to_numeric(df[col], errors='coerce')
        if converted_col.notna().sum() > (len(df) * 0.5):
            df[col] = converted_col

    return df

# 3. MAIN PROCESS FUNCTION
def run_png_to_excel(image_path, output_dir="hasil_ekstraksi"):
    """
    Alur Utama: Preprocess -> AI Detection -> Data Cleaning -> Save Excel
    """
    # A. Pre-processing
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        print(f"[ERROR] Gagal memuat gambar: {image_path}")
        return

    # B. AI Analysis
    result = TABLE_ENGINE(processed_img)

    # C. Folder Setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    vis_dir = os.path.join(output_dir, "visualisasi")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    # D. Extraction Process
    table_count = 0
    for region in result:
        if region['type'] == 'table':
            table_count += 1
            try:
                # Konversi HTML hasil AI menjadi DataFrame Pandas
                df_list = pd.read_html(region['res']['html'])
                if df_list:
                    df = clean_dataframe(df_list[0])
                    
                    # Simpan ke file Excel
                    excel_name = f"tabel_{table_count}.xlsx"
                    df.to_excel(os.path.join(output_dir, excel_name), index=False)
                    print(f"    [SUCCESS] Tersimpan: {excel_name}")
            except Exception as e:
                print(f"    [ERROR] Gagal memproses tabel ke-{table_count}: {e}")

    # E. Simpan Visualisasi kotak deteksi
    save_structure_res(result, processed_img, output_dir, vis_dir)
    
    return table_count

# 4. EXECUTION BLOCK (Loop Folder)
if __name__ == "__main__":
    # Ubah ke 'image' untuk memproses semua file di folder tersebut
    IMAGE_FOLDER = "image" 
    
    if not os.path.exists(IMAGE_FOLDER):
        print(f"[ERROR] Folder {IMAGE_FOLDER} tidak ditemukan!")
    else:
        # Ambil semua file .png
        all_images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith('.png')]
        
        if not all_images:
            print(f"[INFO] Tidak ada file .png di folder {IMAGE_FOLDER}")
        else:
            print(f"[INFO] Ditemukan {len(all_images)} gambar. Mulai memproses...")
            
            for filename in all_images:
                image_path = os.path.join(IMAGE_FOLDER, filename)
                # Folder output unik untuk setiap file agar tidak tertimpa
                folder_name = os.path.splitext(filename)[0]
                custom_output_dir = os.path.join("hasil_ekstraksi", folder_name)
                
                print(f"\n>>> Memproses: {filename}")
                count = run_png_to_excel(image_path, custom_output_dir)
                print(f"    Selesai. {count if count else 0} tabel ditemukan.")

            print("\n" + "="*30)
            print("PROSES SEMUA GAMBAR SELESAI")
            print("="*30)