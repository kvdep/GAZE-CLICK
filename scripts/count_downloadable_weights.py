import json

with open('scripts/huggingface_gaze_models.json', 'r', encoding='utf-8') as f:
    hf = json.load(f)

with open('scripts/github_arxiv_gaze_models.json', 'r', encoding='utf-8') as f:
    gh = json.load(f)

direct_automated_links = []
# 1. HF
for h in hf:
    repo_id = h['id']
    for f in h.get('files', []):
        if any(f.endswith(ext) for ext in ('.pt', '.pth', '.onnx', '.tflite', '.safetensors', '.pkl', '.bin')):
            direct_automated_links.append((repo_id, f, f"https://huggingface.co/{repo_id}/resolve/main/{f}"))

# 2. GH releases
for g in gh:
    repo = g['repo']
    for w in g.get('weight_links', []):
        if 'releases/download' in w:
            fname = w.split('/')[-1]
            direct_automated_links.append((repo, fname, w))

print(f"Total direct automated download checkpoints (HF + GH Releases): {len(direct_automated_links)}")

gdrive_links = []
for g in gh:
    repo = g['repo']
    for w in g.get('weight_links', []):
        if 'drive.google' in w:
            gdrive_links.append((repo, w))

zenodo_links = []
for g in gh:
    repo = g['repo']
    for w in g.get('weight_links', []):
        if 'zenodo' in w:
            zenodo_links.append((repo, w))

print(f"Total Google Drive repos: {len(gdrive_links)}")
print(f"Total Zenodo links: {len(zenodo_links)}")

# Group by unique model architectures for our benchmark:
print("\nUnique architectures in docs/11: 27")
