import subprocess
import re

res = subprocess.run(["curl.exe", "-s", "-L", "https://zenodo.org/records/13852050"], capture_output=True, text=True, encoding='utf-8', errors='ignore')
html = res.stdout

title_match = re.search(r'<title>(.*?)</title>', html)
print("Title:", title_match.group(1) if title_match else "None")

# Find creators
creators = re.findall(r'<meta name="citation_author" content="([^"]+)">', html)
print("Creators:", creators)

# Find files and sizes
files = re.findall(r'href="(/records/13852050/files/[^"]+)"', html)
print("Files count:", len(files))
for f in set(files):
    print(" -", f)

# Find publication date
pub_date = re.search(r'<meta name="citation_publication_date" content="([^"]+)">', html)
print("Pub date:", pub_date.group(1) if pub_date else "None")

# Find DOI
doi = re.search(r'<meta name="citation_doi" content="([^"]+)">', html)
print("DOI:", doi.group(1) if doi else "None")
