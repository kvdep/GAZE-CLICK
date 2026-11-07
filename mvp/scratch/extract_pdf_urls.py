import re

with open("mvp/scratch/hybridgaze_paper.pdf", "rb") as f:
    data = f.read()

# Extract ASCII strings
urls = re.findall(b'https?://[a-zA-Z0-9./_~%-]+', data)
unique_urls = sorted(list(set(urls)))
print(f"Total URLs found in PDF: {len(unique_urls)}")
for u in unique_urls:
    u_str = u.decode('latin1', errors='ignore')
    if any(k in u_str.lower() for k in ['kaggle', 'github', 'zenodo', 'dataset', 'drive', 'huggingface', 'data', 'hybridgaze']):
        print("Relevant URL:", u_str)
