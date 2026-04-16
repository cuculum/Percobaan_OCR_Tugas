import os
import re
from io import StringIO
import cv2
import numpy as np
import pandas as pd
from paddleocr import PPStructure

try:
    import openpyxl
except ImportError:
    print("[ERROR] Install openpyxl: pip install openpyxl")
    exit(1)


# ─────────────────────────────────────────────
# 1. INISIALISASI GLOBAL
# ─────────────────────────────────────────────
print("[INFO] Menginisialisasi AI Engine (PP-Structure)... Mohon tunggu.")
print("[INFO] Download model pertama kali bisa memakan waktu...")

try:
    TABLE_ENGINE = PPStructure(
        layout=True,
        table=True,
        ocr=True,
        show_log=False,
        lang='en',
    )
    print("[INFO] AI Engine siap!")
except Exception as e:
    print(f"[FATAL] Gagal inisialisasi: {e}")
    exit(1)


# ─────────────────────────────────────────────
# 2. FUNGSI PENDUKUNG
# ─────────────────────────────────────────────
def preprocess_image(image_path: str):
    """
    Pra-pemrosesan gambar sebelum analisis AI:
    - Resize jika dimensi melebihi 2000px (hemat VRAM)
    - Denoising dengan median blur
    - CLAHE pada color space LAB (lebih akurat untuk tabel)

    Returns:
        np.ndarray | None: Gambar yang sudah diproses, atau None jika gagal.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Tidak bisa membaca gambar: {image_path}")
        return None

    # Resize jika gambar terlalu besar
    height, width = img.shape[:2]
    max_dim = 2000
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
        print(f"    [INFO] Gambar diresize ke {img.shape[1]}x{img.shape[0]}")

    # Denoising ringan dengan median blur
    denoised = cv2.medianBlur(img, 5)

    # CLAHE pada channel Luminance (color space LAB)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])

    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan DataFrame hasil parsing:
    - Hapus baris/kolom yang seluruhnya kosong
    - Bersihkan karakter noise dari OCR
    - Konversi kolom ke numerik jika >50% nilainya angka

    Returns:
        pd.DataFrame: DataFrame yang sudah dibersihkan.
    """
    if df is None or df.empty:
        return df

    # Hapus baris dan kolom yang seluruhnya kosong
    df = df.dropna(how='all').dropna(axis=1, how='all')

    # Bersihkan karakter noise OCR
    noise_pattern = re.compile(r'[|~_\[\]\{\}\^]')

    def clean_cell(val):
        if isinstance(val, str):
            return noise_pattern.sub('', val).strip()
        return val

    df = df.map(clean_cell)  # Gunakan .map() (applymap() deprecated di pandas 2.1+)

    # Konversi kolom ke numerik jika memungkinkan
    for col in df.columns:
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() > len(df) * 0.5:
                df[col] = converted
        except Exception:
            pass

    return df


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ubah header MultiIndex menjadi 1 level agar aman ditulis ke Excel.
    """
    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        flattened_cols = []
        for col in df.columns:
            parts = [str(part).strip() for part in col if pd.notna(part)]
            parts = [p for p in parts if p and p.lower() != 'nan']
            flattened_cols.append(" | ".join(parts) if parts else "column")
        df.columns = flattened_cols
    else:
        df.columns = [str(col).strip() for col in df.columns]

    # Pastikan nama kolom unik
    seen = {}
    unique_cols = []
    for col in df.columns:
        base = col if col else "column"
        seen[base] = seen.get(base, 0) + 1
        unique_cols.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    df.columns = unique_cols
    return df


def parse_table_html(html_content: str) -> pd.DataFrame | None:
    """
    Parse HTML tabel dengan beberapa strategi agar hasil lebih stabil.
    """
    candidates = []
    parse_variants = [
        {"header": 0},
        {"header": None},
    ]

    for variant in parse_variants:
        try:
            tables = pd.read_html(StringIO(html_content), **variant)
            candidates.extend(tables)
        except Exception:
            continue

    if not candidates:
        return None

    # Pilih kandidat dengan isi non-kosong terbanyak.
    best_df = None
    best_score = -1
    for cand in candidates:
        score = cand.replace("", np.nan).notna().sum().sum()
        if score > best_score:
            best_score = score
            best_df = cand

    if best_df is None:
        return None

    best_df = flatten_columns(best_df)
    best_df = clean_dataframe(best_df)

    # Hapus baris yang identik dengan header (sering muncul dari OCR tabel bertingkat).
    if not best_df.empty:
        header_vals = [str(c).strip().lower() for c in best_df.columns]
        drop_rows = []
        for idx, row in best_df.iterrows():
            row_vals = [str(v).strip().lower() for v in row.tolist()]
            if row_vals == header_vals:
                drop_rows.append(idx)
        if drop_rows:
            best_df = best_df.drop(index=drop_rows)

    return best_df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. FUNGSI UTAMA
# ─────────────────────────────────────────────
def run_png_to_excel(image_path: str, output_dir: str = "hasil_ekstraksi") -> int:
    """
    Memproses satu gambar: deteksi tabel → parsing → simpan ke Excel & CSV.

    Args:
        image_path: Path gambar yang akan diproses.
        output_dir: Direktori output untuk file hasil.

    Returns:
        int: Jumlah tabel yang berhasil diekstrak.
    """
    # Pra-pemrosesan gambar
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return 0

    # Analisis AI
    try:
        result = TABLE_ENGINE(processed_img)
        if not result:
            print("    [INFO] Tidak ada struktur yang terdeteksi")
            return 0
    except Exception as e:
        print(f"    [ERROR] AI gagal memproses gambar: {e}")
        return 0

    # Buat direktori output
    vis_dir = os.path.join(output_dir, "visualisasi")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Ekstraksi tabel dari hasil analisis
    table_count = 0
    for region in result:
        if region.get('type') != 'table':
            continue

        table_count += 1

        try:
            # Ambil konten HTML (kompatibel PaddleOCR <2.7 dan >=2.8)
            html_content = None
            if 'res' in region and isinstance(region['res'], dict):
                html_content = region['res'].get('html')       # PaddleOCR <2.7
            elif 'html' in region:
                html_content = region['html']                  # PaddleOCR >=2.8

            if not html_content:
                print(f"    [WARN] Tidak ada data HTML di tabel {table_count}")
                continue

            # Parse HTML → DataFrame (dengan fallback strategy)
            df = parse_table_html(html_content)
            if df is None or df.empty:
                print(f"    [WARN] Gagal parsing HTML tabel {table_count}")
                continue

            # Validasi ukuran tabel minimal 2 baris × 1 kolom
            if df.shape[0] < 2 or df.shape[1] < 1:
                print(f"    [WARN] Tabel {table_count} terlalu kecil, dilewati")
                continue

            # Simpan ke Excel
            excel_path = os.path.join(output_dir, f"tabel_{table_count}.xlsx")
            df.to_excel(excel_path, index=False, engine='openpyxl')

            # Simpan ke CSV sebagai cadangan
            csv_path = os.path.join(output_dir, f"tabel_{table_count}.csv")
            df.to_csv(csv_path, index=False)

            print(f"    [SUCCESS] tabel_{table_count}.xlsx tersimpan "
                  f"({df.shape[0]} baris × {df.shape[1]} kolom)")

        except Exception as e:
            print(f"    [ERROR] Gagal memproses tabel {table_count}: {str(e)[:100]}")

    return table_count


# ─────────────────────────────────────────────
# 4. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    IMAGE_FOLDER = "image"
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    if not os.path.exists(IMAGE_FOLDER):
        print(f"[ERROR] Folder '{IMAGE_FOLDER}' tidak ditemukan!")
    else:
        all_images = [
            f for f in os.listdir(IMAGE_FOLDER)
            if f.lower().endswith(SUPPORTED_FORMATS)
        ]

        if not all_images:
            print(f"[INFO] Tidak ada gambar di folder '{IMAGE_FOLDER}'")
        else:
            print(f"[INFO] Ditemukan {len(all_images)} gambar. Mulai memproses...\n")

            for filename in all_images:
                image_path = os.path.join(IMAGE_FOLDER, filename)
                folder_name = os.path.splitext(filename)[0]
                output_dir = os.path.join("hasil_ekstraksi", folder_name)

                print(f">>> Memproses: {filename}")
                try:
                    count = run_png_to_excel(image_path, output_dir)
                    print(f"    Selesai. {count} tabel ditemukan.")
                except Exception as e:
                    print(f"    [FATAL ERROR] {e}")

            print("\n" + "=" * 40)
            print("  PROSES SEMUA GAMBAR SELESAI")
            print("=" * 40)