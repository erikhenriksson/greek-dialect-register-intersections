import os
import urllib.request

BASE_URL = (
    "https://raw.githubusercontent.com/StergiosCha/Greek_dialect_corpus/main/DATA/"
)

FILES = {
    "v1": [
        "clean_file_SMG_part1.txt",
        "clean_file_SMG_part2.txt",
        "clean_file_SMG_part3.txt",
    ],
    "v2": [
        "cretan.txt",
        "cypriot.txt",
        "nothern.txt",  # Note: typo in original
        "pontic.txt",
    ],
}

for version, files in FILES.items():
    os.makedirs(f"DATA/{version}", exist_ok=True)

    for filename in files:
        url = f"{BASE_URL}{version}/{filename}"
        filepath = f"DATA/{version}/{filename}"

        try:
            urllib.request.urlretrieve(url, filepath)
            size = os.path.getsize(filepath)
            print(f"✓ {version}/{filename} ({size:,} bytes)")
        except Exception as e:
            print(f"✗ {version}/{filename}: {e}")

print("Done.")
