
from pathlib import Path
from io import BytesIO

import pyarrow.parquet as pq
from PIL import Image

DATA_DIR = Path(
    r"D:\MedVision-Chest\datasets\ChestXray14-320\data"
)

OUTPUT_DIR = Path(
    r"C:\Users\ASUS\MedVision-AI\test_images"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

parquet_files = sorted(DATA_DIR.glob("test-*.parquet"))

if not parquet_files:
    raise FileNotFoundError("No test parquet files found.")

saved = 0

for parquet_path in parquet_files:
    parquet_file = pq.ParquetFile(parquet_path)

    for batch in parquet_file.iter_batches(
        batch_size=16,
        columns=["image"],
    ):
        images = batch.column(0).to_pylist()

        for image_data in images:
            if isinstance(image_data, dict):
                image_bytes = image_data.get("bytes")
                image_path = image_data.get("path")

                if image_bytes:
                    image = Image.open(BytesIO(image_bytes))
                elif image_path:
                    path = Path(image_path)
                    if not path.is_absolute():
                        path = parquet_path.parent / path
                    image = Image.open(path)
                else:
                    continue

            elif isinstance(image_data, bytes):
                image = Image.open(BytesIO(image_data))
            else:
                continue

            with image:
                output = image.convert("RGB")
                saved += 1
                output.save(
                    OUTPUT_DIR / f"chest_test_{saved:02d}.png"
                )

            print(f"Saved image {saved}/10")

            if saved >= 10:
                print(f"\nDone! Images saved to:\n{OUTPUT_DIR}")
                raise SystemExit(0)

print(f"Finished. Saved {saved} images.")
