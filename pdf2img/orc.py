import os
import re
import time
import json
import unicodedata
from io import StringIO
from typing import Any
import cv2
import numpy as np
import pandas as pd
from paddleocr import PPStructure
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

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

OCR_LANG = os.getenv("OCR_LANG", "en")

try:
    TABLE_ENGINE = PPStructure(
        layout=True,
        table=True,
        ocr=True,
        show_log=False,
        lang=OCR_LANG,
    )
    print("[INFO] AI Engine siap!")
except Exception as e:
    if OCR_LANG != "en":
        print(f"[WARN] Gagal pakai lang='{OCR_LANG}' ({e}), fallback ke 'en'")
        try:
            TABLE_ENGINE = PPStructure(
                layout=True,
                table=True,
                ocr=True,
                show_log=False,
                lang="en",
            )
            print("[INFO] AI Engine siap (fallback en)!")
        except Exception as fallback_error:
            print(f"[FATAL] Gagal inisialisasi: {fallback_error}")
            exit(1)
    else:
        print(f"[FATAL] Gagal inisialisasi: {e}")
        exit(1)


# ─────────────────────────────────────────────
# 2. FUNGSI PENDUKUNG
# ─────────────────────────────────────────────
def resize_if_large(img: np.ndarray, max_dim: int = 2200) -> np.ndarray:
    """
    Resize gambar besar agar proses OCR lebih stabil.
    """
    height, width = img.shape[:2]
    if max(height, width) <= max_dim:
        return img

    scale = max_dim / max(height, width)
    resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    print(f"    [INFO] Gambar diresize ke {resized.shape[1]}x{resized.shape[0]}")
    return resized


def deskew_light(img: np.ndarray) -> np.ndarray:
    """
    Koreksi kemiringan ringan berbasis minimum area rectangle.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))

    if len(coords) < 100:
        return img

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.4:
        return img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _apply_clahe_bgr(img: np.ndarray, clip_limit: float, grid_size: tuple[int, int]) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def generate_preprocess_variants(image_path: str) -> list[tuple[str, np.ndarray]]:
    """
    Buat beberapa mode preprocessing untuk memilih output OCR terbaik.
    """
    src = cv2.imread(image_path)
    if src is None:
        print(f"[ERROR] Tidak bisa membaca gambar: {image_path}")
        return []

    src = resize_if_large(src)
    src = deskew_light(src)

    variants: list[tuple[str, np.ndarray]] = []

    # Mode 1: balanced (default)
    balanced = cv2.medianBlur(src, 3)
    balanced = _apply_clahe_bgr(balanced, clip_limit=2.0, grid_size=(8, 8))
    variants.append(("balanced", balanced))

    # Mode 2: low_contrast
    low_contrast = cv2.bilateralFilter(src, d=7, sigmaColor=70, sigmaSpace=70)
    low_contrast = _apply_clahe_bgr(low_contrast, clip_limit=3.0, grid_size=(6, 6))
    variants.append(("low_contrast", low_contrast))

    # Mode 3: noisy_table_lines
    noisy = cv2.fastNlMeansDenoisingColored(src, None, 8, 8, 7, 17)
    gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    sharp = cv2.GaussianBlur(gray, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray, 1.5, sharp, -0.5, 0)
    noisy = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    variants.append(("noisy_table_lines", noisy))
    return variants


def detect_table_grid_stats(img: np.ndarray) -> dict[str, Any]:
    """
    Estimasi struktur grid tabel dari garis horizontal/vertikal.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        5,
    )

    h, w = bin_img.shape[:2]
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(18, h // 25)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, w // 25), 1))

    vertical = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    horizontal = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    v_count, _, v_stats, _ = cv2.connectedComponentsWithStats(vertical, connectivity=8)
    h_count, _, h_stats, _ = cv2.connectedComponentsWithStats(horizontal, connectivity=8)

    # komponen dengan panjang signifikan dihitung sebagai garis.
    vertical_lines = 0
    for i in range(1, v_count):
        x, y, ww, hh, area = v_stats[i]
        if hh > h * 0.2 and ww <= max(10, w * 0.03) and area > 30:
            vertical_lines += 1

    horizontal_lines = 0
    for i in range(1, h_count):
        x, y, ww, hh, area = h_stats[i]
        if ww > w * 0.2 and hh <= max(10, h * 0.03) and area > 30:
            horizontal_lines += 1

    estimated_cols = max(vertical_lines - 1, 1) if vertical_lines >= 2 else 0
    estimated_rows = max(horizontal_lines - 1, 1) if horizontal_lines >= 2 else 0
    confidence = min((vertical_lines + horizontal_lines) / 24.0, 1.0)

    return {
        "vertical_lines": int(vertical_lines),
        "horizontal_lines": int(horizontal_lines),
        "estimated_cols": int(estimated_cols),
        "estimated_rows": int(estimated_rows),
        "grid_confidence": round(float(confidence), 4),
    }


def normalize_text_cell(value: Any) -> Any:
    """
    Normalisasi teks OCR untuk mengurangi karakter liar/non-latin.
    """
    if not isinstance(value, str):
        return value

    text = unicodedata.normalize("NFKC", value)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[\u4e00-\u9fff]+", "", text)  # buang karakter CJK
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"[|~_\[\]\{\}\^]", "", text)

    # Confusable characters (kontekstual, konservatif).
    if re.search(r"\d", text):
        text = re.sub(r"(?<=\d)[oO](?=\d)", "0", text)
        text = re.sub(r"(?<=\d)[lI](?=\d)", "1", text)
        text = re.sub(r"(?<=\d)B(?=\d)", "8", text)
    if re.search(r"[A-Za-z]", text):
        text = re.sub(r"(?<=[A-Za-z])0(?=[A-Za-z])", "O", text)

    # Koreksi kata OCR umum pada laporan keuangan (konservatif)
    token_replacements = {
        "PITIOLAAN": "PENGELOLAAN",
        "MOOAL": "MODAL",
        "KIWAJBAN": "KEWAJIBAN",
        "Cietor": "CETOR",
        "Treury": "Treasury",
    }
    for wrong, right in token_replacements.items():
        text = text.replace(wrong, right)

    return text


def parse_accounting_number(value: Any) -> Any:
    """
    Parse angka akuntansi dari teks OCR:
    - '1 338 539' -> 1338539
    - '(4 240 284)' -> -4240284
    """
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    is_negative = text.startswith("(") and text.endswith(")")
    if is_negative:
        text = text[1:-1]

    text = re.sub(r"\s+", " ", text)
    text = text.replace("O", "0").replace("o", "0")
    text = re.sub(r"(?<=\d)[lI](?=\d)", "1", text)

    # hanya parse bila string dominan numerik/pemisah
    if not re.fullmatch(r"[\d\s.,\-]+", text):
        return value

    compact = re.sub(r"\s+", "", text)
    if compact.count(",") > 0 and compact.count(".") == 0:
        compact = compact.replace(",", ".")
    elif compact.count(".") > 1 and compact.count(",") == 0:
        compact = compact.replace(".", "")
    elif compact.count(".") > 0 and compact.count(",") > 0:
        # format campuran, asumsi titik ribuan & koma desimal
        compact = compact.replace(".", "").replace(",", ".")

    try:
        number = float(compact)
        if number.is_integer():
            number = int(number)
        if is_negative:
            number = -number
        return number
    except ValueError:
        return value


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

    # Bersihkan karakter noise OCR + normalisasi bahasa
    df = df.map(normalize_text_cell)

    # Parse angka akuntansi dari teks
    df = df.map(parse_accounting_number)

    # Konversi kolom ke numerik jika memungkinkan (threshold 60%)
    for col in df.columns:
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() > len(df) * 0.6:
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


def maybe_promote_first_row_as_header(df: pd.DataFrame) -> pd.DataFrame:
    """
    Jika header default numerik, coba gunakan baris pertama sebagai header.
    """
    if df is None or df.empty:
        return df

    numeric_like_header = sum(str(c).strip().isdigit() for c in df.columns)
    if numeric_like_header < max(2, len(df.columns) // 2):
        return df

    first_row = df.iloc[0].tolist()
    candidate = [str(v).strip() for v in first_row]
    valid = [v for v in candidate if v and v.lower() != "nan"]
    if len(valid) < max(2, len(df.columns) // 2):
        return df

    # Jika baris pertama dominan numerik, besar kemungkinan itu data bukan header.
    numeric_cells = sum(bool(re.search(r"\d", v)) for v in candidate if v)
    if numeric_cells > len(candidate) * 0.6:
        return df

    seen: dict[str, int] = {}
    new_cols = []
    for col in candidate:
        base = col if col else "column"
        seen[base] = seen.get(base, 0) + 1
        new_cols.append(base if seen[base] == 1 else f"{base}_{seen[base]}")

    body = df.iloc[1:].copy()
    body.columns = new_cols
    return body.reset_index(drop=True)


def merge_two_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gabungkan dua baris awal jika keduanya terlihat seperti header bertingkat.
    """
    if df is None or df.empty or len(df) < 2:
        return df

    row0 = [str(v).strip() for v in df.iloc[0].tolist()]
    row1 = [str(v).strip() for v in df.iloc[1].tolist()]
    row0_alpha = sum(bool(re.search(r"[A-Za-z]", v)) for v in row0 if v)
    row1_alpha = sum(bool(re.search(r"[A-Za-z]", v)) for v in row1 if v)
    row0_num = sum(bool(re.search(r"\d", v)) for v in row0 if v)
    row1_num = sum(bool(re.search(r"\d", v)) for v in row1 if v)

    # Dua baris awal dianggap header bila dominan label non-angka.
    if (row0_alpha + row1_alpha) < max(2, len(df.columns) // 2):
        return df
    if (row0_num + row1_num) > len(df.columns) * 1.2:
        return df

    merged = []
    for left, right in zip(row0, row1):
        lv = left if left and left.lower() != "nan" else ""
        rv = right if right and right.lower() != "nan" else ""
        merged_name = f"{lv} {rv}".strip()
        merged.append(merged_name if merged_name else "column")

    seen: dict[str, int] = {}
    final_cols = []
    for col in merged:
        seen[col] = seen.get(col, 0) + 1
        final_cols.append(col if seen[col] == 1 else f"{col}_{seen[col]}")

    body = df.iloc[2:].copy()
    body.columns = final_cols
    return body.reset_index(drop=True)


def remove_duplicate_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buang baris yang identik dengan header.
    """
    if df is None or df.empty:
        return df

    header_vals = [str(c).strip().lower() for c in df.columns]
    to_drop = []
    for idx, row in df.iterrows():
        row_vals = [str(v).strip().lower() for v in row.tolist()]
        if row_vals == header_vals:
            to_drop.append(idx)

    if to_drop:
        df = df.drop(index=to_drop)
    return df.reset_index(drop=True)


def dataframe_quality_score(df: pd.DataFrame, grid_stats: dict[str, Any] | None = None) -> float:
    """
    Skor sederhana untuk menilai kualitas tabel hasil parsing.
    """
    if df is None or df.empty:
        return 0.0

    rows, cols = df.shape
    total_cells = max(rows * cols, 1)
    non_empty = df.replace("", np.nan).notna().sum().sum()
    fill_ratio = non_empty / total_cells
    row_non_empty = df.replace("", np.nan).notna().sum(axis=1)
    row_consistency = 0.0
    if len(row_non_empty) > 0:
        row_consistency = 1.0 - (row_non_empty.std(ddof=0) / max(cols, 1))
        row_consistency = float(np.clip(row_consistency, 0.0, 1.0))

    numeric_like = df.applymap(
        lambda v: bool(re.search(r"\d", str(v))) if not pd.isna(v) else False
    )
    numeric_ratio = float(numeric_like.sum().sum() / total_cells)

    # Penalti jika terlalu banyak kolom "column"/duplikat pseudo header
    generic_cols = sum(str(col).lower().startswith("column") for col in df.columns)
    generic_penalty = generic_cols / max(cols, 1)
    long_text_penalty = 0.0
    if cols == 1:
        avg_len = float(df.iloc[:, 0].astype(str).str.len().mean())
        long_text_penalty = min(avg_len / 40.0, 1.0)

    grid_alignment_bonus = 0.0
    grid_mismatch_penalty = 0.0
    if grid_stats and grid_stats.get("grid_confidence", 0.0) >= 0.2:
        expected_cols = int(grid_stats.get("estimated_cols", 0))
        if expected_cols > 0:
            if cols == expected_cols:
                grid_alignment_bonus = 0.2
            else:
                diff_ratio = abs(cols - expected_cols) / max(expected_cols, 1)
                grid_mismatch_penalty = min(diff_ratio, 1.0) * 0.25

    # Bonus untuk tabel yang cukup lebar/tinggi
    structure_bonus = min(rows / 12, 1.0) * 0.2 + min(cols / 8, 1.0) * 0.2
    score = (
        fill_ratio * 0.55
        + structure_bonus
        + row_consistency * 0.15
        + min(numeric_ratio, 0.4) * 0.15
        + grid_alignment_bonus
        - generic_penalty * 0.2
        - long_text_penalty * 0.25
        - grid_mismatch_penalty
    )
    return round(max(score, 0.0), 4)


def parse_table_html(html_content: str, grid_stats: dict[str, Any] | None = None) -> pd.DataFrame | None:
    """
    Parse HTML tabel dengan beberapa strategi agar hasil lebih stabil.
    """
    candidates = []
    parse_variants = [
        {"header": 0},
        {"header": None},
        {"header": [0, 1]},
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
        cand = flatten_columns(cand)
        cand = merge_two_header_rows(cand)
        cand = maybe_promote_first_row_as_header(cand)
        cand = clean_dataframe(cand)
        cand = remove_duplicate_header_rows(cand)
        score = dataframe_quality_score(cand, grid_stats=grid_stats)
        if score > best_score:
            best_score = score
            best_df = cand

    if best_df is None:
        return None
    return best_df.reset_index(drop=True)

def write_styled_excel(raw_df: pd.DataFrame, clean_df: pd.DataFrame, excel_path: str):
    """
    Simpan 2 sheet (raw + cleaned) lalu beri styling tabel.
    """
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="raw_extraction", index=False)
        clean_df.to_excel(writer, sheet_name="cleaned_output", index=False)

    workbook = openpyxl.load_workbook(excel_path)
    thin = Side(border_style="thin", color="000000")
    header_fill = PatternFill(start_color="E2F0D9", end_color="E2F0D9", fill_type="solid")
    header_font = Font(bold=True)

    for ws in workbook.worksheets:
        max_row = ws.max_row
        max_col = ws.max_column

        for row in ws.iter_rows(min_row=1, max_row=max_row, min_col=1, max_col=max_col):
            for cell in row:
                cell.border = Border(top=thin, bottom=thin, left=thin, right=thin)
                if cell.row == 1:
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                else:
                    cell.alignment = Alignment(vertical="top", wrap_text=True)

        for col_cells in ws.columns:
            max_len = 0
            col_letter = col_cells[0].column_letter
            for cell in col_cells:
                cell_val = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(cell_val))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 48)

    workbook.save(excel_path)


def write_raw_only_excel(raw_df: pd.DataFrame, excel_path: str):
    """
    Simpan hanya sheet raw ketika quality gate gagal.
    """
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="raw_extraction", index=False)


def apply_quality_gate(clean_df: pd.DataFrame, grid_stats: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    """
    Validasi kualitas minimum sebelum ekspor final.
    """
    reasons = []
    if clean_df is None or clean_df.empty:
        reasons.append("dataframe kosong")
        return False, reasons

    rows, cols = clean_df.shape
    if rows < 2:
        reasons.append("baris kurang dari 2")

    empty_ratio = 1.0 - (clean_df.replace("", np.nan).notna().sum().sum() / max(rows * cols, 1))
    if empty_ratio > 0.78:
        reasons.append(f"rasio kosong terlalu tinggi ({empty_ratio:.2f})")

    if cols == 1:
        avg_text_len = float(clean_df.iloc[:, 0].astype(str).str.len().mean())
        if avg_text_len > 22:
            reasons.append("indikasi collapse satu kolom")

    if grid_stats and grid_stats.get("grid_confidence", 0.0) >= 0.2:
        expected_cols = int(grid_stats.get("estimated_cols", 0))
        if expected_cols > 1 and cols == 1:
            reasons.append(f"grid mendeteksi {expected_cols} kolom, hasil hanya 1 kolom")

    return len(reasons) == 0, reasons


# ─────────────────────────────────────────────
# 3. FUNGSI UTAMA
# ─────────────────────────────────────────────
def run_png_to_excel(image_path: str, output_dir: str = "hasil_ekstraksi") -> dict[str, Any]:
    """
    Memproses satu gambar: deteksi tabel → parsing → simpan ke Excel & CSV.

    Args:
        image_path: Path gambar yang akan diproses.
        output_dir: Direktori output untuk file hasil.

    Returns:
        dict[str, Any]: Ringkasan proses per file.
    """
    start = time.time()
    variants = generate_preprocess_variants(image_path)
    if not variants:
        return {
            "tables_detected": 0,
            "tables_saved": 0,
            "tables_skipped_by_gate": 0,
            "best_mode": "none",
            "avg_quality_score": 0.0,
            "error_count": 1,
            "gate_log_count": 0,
            "elapsed_sec": 0.0,
        }

    # Buat direktori output
    vis_dir = os.path.join(output_dir, "visualisasi")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    best_candidate: dict[str, Any] | None = None
    variant_errors = 0
    for mode_name, processed_img in variants:
        grid_stats = detect_table_grid_stats(processed_img)
        try:
            result = TABLE_ENGINE(processed_img)
        except Exception as e:
            variant_errors += 1
            print(f"    [WARN] Mode {mode_name} gagal: {e}")
            continue

        current_tables: list[dict[str, Any]] = []
        detected = 0
        for region in result:
            if region.get("type") != "table":
                continue
            detected += 1

            html_content = None
            if "res" in region and isinstance(region["res"], dict):
                html_content = region["res"].get("html")
            elif "html" in region:
                html_content = region["html"]
            if not html_content:
                continue

            raw_list = pd.read_html(StringIO(html_content), header=None)
            if not raw_list:
                continue
            raw_df = flatten_columns(raw_list[0])

            clean_df = parse_table_html(html_content, grid_stats=grid_stats)
            if clean_df is None or clean_df.empty:
                continue

            quality = dataframe_quality_score(clean_df, grid_stats=grid_stats)
            current_tables.append(
                {"raw_df": raw_df, "clean_df": clean_df, "quality": quality, "grid_stats": grid_stats}
            )

        if not current_tables:
            continue

        avg_score = float(np.mean([tbl["quality"] for tbl in current_tables]))
        candidate = {
            "mode_name": mode_name,
            "tables": current_tables,
            "detected": detected,
            "avg_score": avg_score,
        }
        if best_candidate is None:
            best_candidate = candidate
        else:
            # Prioritas: avg score, lalu jumlah tabel
            if (
                candidate["avg_score"] > best_candidate["avg_score"]
                or (
                    candidate["avg_score"] == best_candidate["avg_score"]
                    and len(candidate["tables"]) > len(best_candidate["tables"])
                )
            ):
                best_candidate = candidate

    if best_candidate is None:
        print("    [INFO] Tidak ada tabel valid dari semua mode preprocessing")
        elapsed = round(time.time() - start, 3)
        return {
            "tables_detected": 0,
            "tables_saved": 0,
            "tables_skipped_by_gate": 0,
            "best_mode": "none",
            "avg_quality_score": 0.0,
            "error_count": variant_errors,
            "gate_log_count": 0,
            "elapsed_sec": elapsed,
        }

    saved_count = 0
    skipped_by_gate = 0
    gate_logs = []
    for idx, table in enumerate(best_candidate["tables"], start=1):
        clean_df = table["clean_df"]
        raw_df = table["raw_df"]
        grid_stats = table.get("grid_stats", {})

        if clean_df.shape[0] < 2 or clean_df.shape[1] < 1:
            print(f"    [WARN] Tabel {idx} terlalu kecil, dilewati")
            continue

        try:
            gate_ok, reasons = apply_quality_gate(clean_df, grid_stats=grid_stats)
            excel_path = os.path.join(output_dir, f"tabel_{idx}.xlsx")
            if gate_ok:
                write_styled_excel(raw_df, clean_df, excel_path)
            else:
                skipped_by_gate += 1
                write_raw_only_excel(raw_df, excel_path)
                reason_text = "; ".join(reasons)
                gate_logs.append({"table": idx, "reasons": reasons})
                print(f"    [WARN] Quality gate skip tabel {idx}: {reason_text}")

            csv_path = os.path.join(output_dir, f"tabel_{idx}.csv")
            if gate_ok:
                clean_df.to_csv(csv_path, index=False)
                saved_count += 1
                print(
                    f"    [SUCCESS] tabel_{idx}.xlsx tersimpan ({clean_df.shape[0]} baris × {clean_df.shape[1]} kolom)"
                )
            else:
                raw_df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"    [ERROR] Gagal menyimpan tabel {idx}: {str(e)[:120]}")

    elapsed = round(time.time() - start, 3)
    print(
        f"    [INFO] Mode terbaik: {best_candidate['mode_name']} | "
        f"quality={best_candidate['avg_score']:.3f}"
    )
    return {
        "tables_detected": int(best_candidate["detected"]),
        "tables_saved": int(saved_count),
        "tables_skipped_by_gate": int(skipped_by_gate),
        "best_mode": best_candidate["mode_name"],
        "avg_quality_score": float(round(best_candidate["avg_score"], 4)),
        "error_count": int(variant_errors),
        "gate_log_count": int(len(gate_logs)),
        "elapsed_sec": elapsed,
    }


# ─────────────────────────────────────────────
# 4. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    IMAGE_FOLDER = "image"
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    quality_rows: list[dict[str, Any]] = []
    sample_limit = int(os.getenv("SAMPLE_LIMIT", "0"))
    regression_list = [x.strip() for x in os.getenv("REGRESSION_IMAGES", "").split(",") if x.strip()]

    if not os.path.exists(IMAGE_FOLDER):
        print(f"[ERROR] Folder '{IMAGE_FOLDER}' tidak ditemukan!")
    else:
        all_images = [
            f for f in os.listdir(IMAGE_FOLDER)
            if f.lower().endswith(SUPPORTED_FORMATS)
        ]
        all_images.sort()

        if regression_list:
            all_images = [img for img in all_images if img in regression_list]
            print(f"[INFO] Mode regression set aktif: {len(all_images)} gambar dipilih.")
        elif sample_limit > 0:
            all_images = all_images[:sample_limit]
            print(f"[INFO] SAMPLE_LIMIT aktif: memproses {len(all_images)} gambar.")

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
                    summary = run_png_to_excel(image_path, output_dir)
                    print(
                        f"    Selesai. ditemukan={summary['tables_detected']}, "
                        f"tersimpan={summary['tables_saved']}, "
                        f"gate_skip={summary['tables_skipped_by_gate']}, "
                        f"mode={summary['best_mode']}, "
                        f"durasi={summary['elapsed_sec']}s"
                    )
                    quality_rows.append({
                        "filename": filename,
                        **summary,
                    })
                except Exception as e:
                    print(f"    [FATAL ERROR] {e}")
                    quality_rows.append({
                        "filename": filename,
                        "tables_detected": 0,
                        "tables_saved": 0,
                        "tables_skipped_by_gate": 0,
                        "best_mode": "fatal_error",
                        "avg_quality_score": 0.0,
                        "error_count": 1,
                        "gate_log_count": 0,
                        "elapsed_sec": 0.0,
                    })

            if quality_rows:
                report_dir = "hasil_ekstraksi"
                os.makedirs(report_dir, exist_ok=True)

                report_df = pd.DataFrame(quality_rows)
                report_csv = os.path.join(report_dir, "quality_report.csv")
                report_json = os.path.join(report_dir, "quality_report.json")
                report_df.to_csv(report_csv, index=False)

                with open(report_json, "w", encoding="utf-8") as f:
                    json.dump(quality_rows, f, ensure_ascii=False, indent=2)

                avg_saved = report_df["tables_saved"].mean() if not report_df.empty else 0
                print(f"[INFO] Quality report tersimpan: {report_csv}")
                print(f"[INFO] Rata-rata tabel tersimpan per gambar: {avg_saved:.2f}")

            print("\n" + "=" * 40)
            print("  PROSES SEMUA GAMBAR SELESAI")
            print("=" * 40)