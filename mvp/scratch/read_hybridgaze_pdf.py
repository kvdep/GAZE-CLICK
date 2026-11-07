import pypdf
import re

reader = pypdf.PdfReader("mvp/scratch/hybridgaze_paper.pdf")
print("Total pages:", len(reader.pages))

full_text = ""
for i, page in enumerate(reader.pages):
    t = page.extract_text()
    full_text += f"\n--- PAGE {i+1} ---\n" + t

with open("mvp/scratch/hybridgaze_paper_text.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

# Search for links in text
urls = re.findall(r'https?://[^\s<>"\')]+', full_text)
print("Found URLs in text:")
for u in set(urls):
    print(" -", u)

# Search for keywords
keywords = ['kaggle', 'github', 'dataset', 'code', 'available', 'repository', 'data availability']
for line in full_text.split('\n'):
    if any(k in line.lower() for k in keywords):
        print("MATCH:", line[:120])
