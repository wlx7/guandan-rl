"""Print downscaled JPEG copies of screenshots as base64 between markers (CI logs)."""
import base64
import io
import sys

from PIL import Image

for path in sys.argv[1:]:
    try:
        im = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"{path}: {e}")
        continue
    im.thumbnail((900, 900))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    print(f"=====BEGIN {path} {im.size[0]}x{im.size[1]}")
    data = base64.b64encode(buf.getvalue()).decode()
    for i in range(0, len(data), 4000):
        print(data[i:i + 4000])
    print(f"=====END {path}")
