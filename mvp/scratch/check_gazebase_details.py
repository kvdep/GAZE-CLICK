import subprocess
import re

res = subprocess.run(["curl.exe", "-s", "-L", "https://doi.org/10.6084/m9.figshare.12912257"], capture_output=True, text=True, errors='ignore')
html = res.stdout

desc = re.search(r'<meta name="description" content="([^"]+)"', html)
print("Description:", desc.group(1) if desc else "None")

# Find file names
filenames = re.findall(r'title="([^"]+\.zip)"', html)
print("ZIP files in GazeBase:", set(filenames))

csv_files = re.findall(r'title="([^"]+\.csv)"', html)
print("CSV files in GazeBase:", set(csv_files))

# Check Nature paper abstract
res_paper = subprocess.run(["curl.exe", "-s", "-L", "https://doi.org/10.1038/s41597-021-00959-y"], capture_output=True, text=True, errors='ignore')
paper_html = res_paper.stdout
paper_title = re.search(r'<title>(.*?)</title>', paper_html)
print("Paper title:", paper_title.group(1) if paper_title else "None")
