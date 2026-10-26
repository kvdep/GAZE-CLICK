import json

with open('scripts/huggingface_gaze_models.json', 'r', encoding='utf-8') as f:
    hf_data = json.load(f)

with open('scripts/github_arxiv_gaze_models.json', 'r', encoding='utf-8') as f:
    gh_data = json.load(f)

print(f"Loaded {len(hf_data)} HF repos and {len(gh_data)} GH repos.")

# Detailed classification
hf_models = []
for h in hf_data:
    weight_files = [f for f in h.get('files', []) if any(f.endswith(ext) for ext in ('.pt', '.pth', '.onnx', '.tflite', '.safetensors', '.pkl', '.bin'))]
    hf_models.append({
        'id': h['id'],
        'year': h.get('year', 2024),
        'weight_files_count': len(weight_files),
        'sample_files': weight_files[:3],
        'downloads': h.get('downloads', 0),
        'license': h.get('license', 'Not specified')
    })

gh_models_with_weights = []
for g in gh_data:
    wlinks = g.get('weight_links', [])
    if wlinks:
        gh_models_with_weights.append({
            'repo': g['repo'],
            'year': g.get('year', 2024),
            'stars': g.get('stars', 0),
            'weight_links': wlinks,
            'backbones': g.get('backbones', []),
            'error': g.get('reported_error_deg')
        })

print("\n--- Summary ---")
print(f"1. Hugging Face Repositories with models: {len(hf_models)}")
total_hf_checkpoints = sum(m['weight_files_count'] for m in hf_models)
print(f"   Total discrete weight files in HF repos: {total_hf_checkpoints}")

print(f"2. GitHub / ArXiv Repositories with weights links: {len(gh_models_with_weights)}")

# Categories of download
direct_http_releases = []
direct_hf = []
gdrive = []
zenodo = []
baidu = []

for m in hf_models:
    direct_hf.append(m['id'])

for g in gh_models_with_weights:
    wlinks = g['weight_links']
    if any('releases/download' in w for w in wlinks):
        direct_http_releases.append(g['repo'])
    elif any('drive.google' in w for w in wlinks):
        gdrive.append(g['repo'])
    elif any('zenodo' in w for w in wlinks):
        zenodo.append(g['repo'])
    elif any('baidu' in w for w in wlinks):
        baidu.append(g['repo'])

print("\n--- Download Method Breakdown ---")
print(f"A. Direct Hugging Face Hub (hf_hub_download / API / wget): {len(direct_hf)} repos")
print(f"B. Direct GitHub Releases (curl / wget / direct HTTP): {len(set(direct_http_releases))} repos")
print(f"C. Google Drive (gdown / direct link): {len(set(gdrive))} repos")
print(f"D. Zenodo Open Science (direct REST API / curl): {len(set(zenodo))} repos")
print(f"E. Baidu Wangpan: {len(set(baidu))} repos")

# Unique distinct architectural approaches ready for testing:
print("\n--- Direct List of GH models with releases: ---")
for r in set(direct_http_releases):
    print(" *", r)

print("\n--- Direct List of GH models with Google Drive: ---")
for r in set(gdrive):
    print(" *", r)
