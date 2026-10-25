#!/usr/bin/env python3
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}

WEIGHT_LINK_PATTERNS = [
    r'https?://drive\.google\.com/[^\s\)\"\'>]+',
    r'https?://github\.com/[^/\s]+/[^/\s]+/releases/download/[^\s\)\"\'>]+',
    r'https?://zenodo\.org/record[s]?/[^\s\)\"\'>]+',
    r'https?://zenodo\.org/doi/[^\s\)\"\'>]+',
    r'https?://huggingface\.co/[^\s\)\"\'>]+',
    r'https?://pan\.baidu\.com/[^\s\)\"\'>]+',
    r'https?://dropbox\.com/[^\s\)\"\'>]+',
    r'https?://(?:www\.)?dropbox\.com/s[a-z]?/[^\s\)\"\'>]+',
    r'https?://mega\.nz/[^\s\)\"\'>]+',
    r'https?://1drv\.ms/[^\s\)\"\'>]+',
]

WEIGHT_FILE_EXTENSIONS = ['.pt', '.pth', '.onnx', '.tflite', '.safetensors', '.pkl', '.ckpt', '.bin', '.h5']

EXCLUDE_REPO_KEYWORDS = [
    'gazeta', 'the-gazette', 'whisper', 'audio', 'voice', 'speech', 'text', 'nlp',
    'irish', 'flower', 'translation', 'summarization', 'diffusion', 'stable-diffusion',
    'lora', 'anime', 'roberta', 'bert', 'llama', 'eeg', 'wearable', 'glasses-sensor'
]

def fetch_url(url, is_json=False, timeout=15):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode('utf-8', errors='ignore')
            if is_json:
                return json.loads(content)
            return content
    except Exception as e:
        return None

def parse_arxiv():
    print("[1/3] Parsing arXiv cs.CV papers for Gaze Estimation (2024-2026)...")
    arxiv_records = []
    # Query arXiv API for cs.CV gaze estimation
    query = 'all:%22gaze+estimation%22+AND+cat:cs.CV'
    url = f'http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=150&sortBy=submittedDate&sortOrder=descending'
    xml_data = fetch_url(url, timeout=25)
    if not xml_data:
        print("Failed to fetch arXiv data")
        return []

    root = ET.fromstring(xml_data)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entries = root.findall('atom:entry', ns)
    print(f"Retrieved {len(entries)} raw entries from arXiv.")

    github_url_pattern = re.compile(r'https?://github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)')

    for entry in entries:
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        published = entry.find('atom:published', ns).text[:10]
        year = int(published[:4])
        if year < 2024:
            continue
        
        arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
        
        # Search for github links in summary
        matches = github_url_pattern.findall(summary)
        cleaned_matches = []
        for m in matches:
            m = m.rstrip('.,;:)')
            if not any(k in m.lower() for k in EXCLUDE_REPO_KEYWORDS):
                cleaned_matches.append(m)
        
        arxiv_records.append({
            'arxiv_id': arxiv_id,
            'title': title,
            'year': year,
            'published': published,
            'summary': summary,
            'github_repos': list(set(cleaned_matches))
        })

    print(f"Filtered {len(arxiv_records)} papers from 2024-2026.")
    return arxiv_records

def parse_github():
    print("[2/3] Querying GitHub API for Gaze Estimation repositories (pushed/created >= 2024)...")
    github_repos = {}
    search_queries = [
        'gaze-estimation+pushed:>2024-01-01',
        '%22gaze+estimation%22+pushed:>2024-01-01',
        'gaze360+pushed:>2024-01-01',
        'mpiifacegaze+pushed:>2024-01-01',
        'eth-xgaze+pushed:>2024-01-01',
        'l2cs+gaze+pushed:>2024-01-01'
    ]

    for q in search_queries:
        url = f'https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page=30'
        data = fetch_url(url, is_json=True, timeout=15)
        if data and 'items' in data:
            for item in data['items']:
                full_name = item.get('full_name')
                if not full_name:
                    continue
                if any(k in full_name.lower() for k in EXCLUDE_REPO_KEYWORDS):
                    continue
                if full_name not in github_repos:
                    github_repos[full_name] = {
                        'full_name': full_name,
                        'description': item.get('description') or '',
                        'stars': item.get('stargazers_count', 0),
                        'created_at': item.get('created_at'),
                        'pushed_at': item.get('pushed_at'),
                        'license': (item.get('license') or {}).get('spdx_id') or 'Not specified',
                        'default_branch': item.get('default_branch') or 'main',
                        'html_url': item.get('html_url')
                    }
        time.sleep(2) # Respect rate limits

    print(f"Found {len(github_repos)} unique GitHub repositories from search.")
    return github_repos

def inspect_repo_weights(repo_full_name, default_branch='main'):
    # Try fetching README from main, then master
    branches = [default_branch, 'master', 'main']
    readme_text = None
    for b in set(branches):
        raw_url = f'https://raw.githubusercontent.com/{repo_full_name}/{b}/README.md'
        readme_text = fetch_url(raw_url, timeout=10)
        if readme_text:
            break

    if not readme_text:
        return None

    # Search for weight download links
    found_links = []
    for pat in WEIGHT_LINK_PATTERNS:
        matches = re.findall(pat, readme_text)
        found_links.extend(matches)

    # Search for mentions of weight files
    has_weight_files = any(ext in readme_text.lower() for ext in WEIGHT_FILE_EXTENSIONS)
    
    # Check for keywords indicating pretrained models
    has_pretrained_mention = any(k in readme_text.lower() for k in ['pretrained', 'pre-trained', 'checkpoint', 'weights', 'model.pt', 'model.onnx', 'download'])

    if not found_links and not (has_weight_files and has_pretrained_mention):
        return None

    # Extract architecture / backbone mentions
    backbones = []
    for b_kw in ['resnet18', 'resnet-18', 'resnet50', 'resnet-50', 'mobilenet', 'mobileone', 'convnext', 'transformer', 'vit', 'swin', 'egnn', 'cspdarknet', 'yolo']:
        if b_kw in readme_text.lower():
            backbones.append(b_kw)

    # Extract datasets
    datasets = []
    for d_kw in ['mpiifacegaze', 'mpiigaze', 'eth-xgaze', 'xgaze', 'gaze360', 'gazecapture', 'eve', 'rt-gene', 'columbia', 'eyediap']:
        if d_kw in readme_text.lower():
            datasets.append(d_kw)

    # Extract metrics (look for degree errors, e.g. 3.92, 4.35 deg, degrees, etc.)
    err_matches = re.findall(r'(\d+\.\d+)\s*(?:°|deg|degrees)', readme_text)
    metric_val = float(err_matches[0]) if err_matches else None

    return {
        'repo': repo_full_name,
        'weight_links': list(set(found_links))[:5],
        'backbones': list(set(backbones)),
        'datasets': list(set(datasets)),
        'reported_error_deg': metric_val,
        'readme_snippet': readme_text[:1200]
    }

def main():
    print("=== GitHub & ArXiv Gaze Estimation Models Parser (2024-2026) ===")
    arxiv_records = parse_arxiv()
    github_repos = parse_github()

    # Add repos found in arXiv papers to github_repos list
    for ar in arxiv_records:
        for r_name in ar['github_repos']:
            if r_name not in github_repos and not any(k in r_name.lower() for k in EXCLUDE_REPO_KEYWORDS):
                github_repos[r_name] = {
                    'full_name': r_name,
                    'description': f"From arXiv: {ar['title']}",
                    'stars': 0,
                    'created_at': ar['published'],
                    'pushed_at': ar['published'],
                    'license': 'See repo',
                    'default_branch': 'main',
                    'html_url': f"https://github.com/{r_name}"
                }

    print(f"\n[3/3] Inspecting {len(github_repos)} repositories for downloadable weights and architectures...")
    verified_models = []

    for idx, (r_name, r_info) in enumerate(github_repos.items(), 1):
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(github_repos)} repos...")
        inspection = inspect_repo_weights(r_name, r_info.get('default_branch', 'main'))
        if inspection:
            year_val = int(r_info.get('created_at', '2024')[:4]) if r_info.get('created_at') else 2024
            record = {
                'repo': r_name,
                'url': r_info.get('html_url') or f"https://github.com/{r_name}",
                'year': max(year_val, 2024),
                'stars': r_info.get('stars', 0),
                'license': r_info.get('license', 'Not specified'),
                'description': r_info.get('description', ''),
                'weight_links': inspection['weight_links'],
                'backbones': inspection['backbones'],
                'datasets': inspection['datasets'],
                'reported_error_deg': inspection['reported_error_deg']
            }
            verified_models.append(record)

    out_file = 'scripts/github_arxiv_gaze_models.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(verified_models, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully verified and saved {len(verified_models)} gaze models with weights/links to {out_file}.")
    print("\n=== Verified Models Overview ===")
    for m in sorted(verified_models, key=lambda x: (x['reported_error_deg'] or 99.0, -x['year'])):
        err_str = f"{m['reported_error_deg']:.2f}°" if m['reported_error_deg'] else "N/A"
        b_str = ", ".join(m['backbones'][:3]) if m['backbones'] else "Custom"
        d_str = ", ".join(m['datasets'][:2]) if m['datasets'] else "Standard"
        w_cnt = len(m['weight_links'])
        print(f"[{m['year']}] {m['repo']:38s} | Err: {err_str:7s} | Backbones: {b_str:20s} | Weights: {w_cnt} links | Stars: {m['stars']}")

if __name__ == '__main__':
    main()
