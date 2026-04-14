from pdf2image import convert_from_path

images = convert_from_path(
    "p.pdf",
    dpi=300,
    poppler_path=r"C:\test program\poppler-25.12.0\Library\bin"  # ⬅️ WAJIB diisi
)

for i, img in enumerate(images):
    img.save(f"page_{i}.png", "PNG")