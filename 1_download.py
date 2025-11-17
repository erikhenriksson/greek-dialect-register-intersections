import os
import urllib.request

# Create directory structure
os.makedirs("DATA/v1", exist_ok=True)
os.makedirs("DATA/v2", exist_ok=True)

# GitHub raw content base URL
base_url_v1 = (
    "https://raw.githubusercontent.com/StergiosCha/Greek_dialect_corpus/main/DATA/v1/"
)
base_url_v2 = (
    "https://raw.githubusercontent.com/StergiosCha/Greek_dialect_corpus/main/DATA/v2/"
)

# Files to download from v2 (dialects)
v2_files = [
    "cretan.txt",
    "cypriot.txt",
    "nothern.txt",
    "pontic.txt",
]  # Note: typo in original

# Files to download from v1 (Standard Greek)
v1_files = [
    "clean_file_SMG_part1.txt",
    "clean_file_SMG_part2.txt",
    "clean_file_SMG_part3.txt",
]

print("Downloading Greek dialect corpus files from GitHub...")
print()

print("=== Downloading Standard Greek (v1) ===")
for filename in v1_files:
    url = base_url_v1 + filename
    filepath = f"DATA/v1/{filename}"

    print(f"Downloading {filename}...", end=" ")
    try:
        urllib.request.urlretrieve(url, filepath)
        # Check file size
        size = os.path.getsize(filepath)
        print(f"✓ ({size:,} bytes)")
    except Exception as e:
        print(f"✗ Error: {e}")

print()
print("=== Downloading Dialects (v2) ===")
for filename in v2_files:
    url = base_url_v2 + filename
    filepath = f"DATA/v2/{filename}"

    print(f"Downloading {filename}...", end=" ")
    try:
        urllib.request.urlretrieve(url, filepath)
        # Check file size
        size = os.path.getsize(filepath)
        print(f"✓ ({size:,} bytes)")
    except Exception as e:
        print(f"✗ Error: {e}")

print()
print("Download complete!")
print("Files saved to DATA/v1/ and DATA/v2/")
print("You can now run: python train_greek_dialects.py")
