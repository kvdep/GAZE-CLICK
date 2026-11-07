with open('mvp/scratch/weyeds_page.html', encoding='utf-8') as f:
    html = f.read()

import re
matches = re.findall(r'<section id="download".*?</section>', html, re.DOTALL)
if matches:
    print("Download section:")
    print(matches[0])
else:
    # Print lines with download
    for line in html.split('\n'):
        if 'download' in line.lower() or 'request' in line.lower() or 'form' in line.lower() or 'email' in line.lower():
            print(line)
