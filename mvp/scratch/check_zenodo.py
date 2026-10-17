import urllib.request
import json

url = "https://zenodo.org/api/records?q=gaze%20estimation%20dataset&size=20&sort=bestmatch"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        hits = data.get("hits", {}).get("hits", [])
        print(f"Total results: {data.get('hits', {}).get('total', 0)}")
        for h in hits:
            rec_id = h.get("id")
            meta = h.get("metadata", {})
            title = meta.get("title")
            access = meta.get("access_right")
            doi = meta.get("doi")
            files = [f.get("key") for f in h.get("files", [])]
            file_sizes = [f.get("size", 0) for f in h.get("files", [])]
            total_size_mb = sum(file_sizes) / (1024 * 1024)
            print(f"ID: {rec_id} | Access: {access} | Size: {total_size_mb:.2f} MB")
            print(f"  Title: {title}")
            print(f"  DOI: {doi}")
            print(f"  Files count: {len(files)}")
            print("-" * 50)
except Exception as e:
    print(f"Error: {e}")
