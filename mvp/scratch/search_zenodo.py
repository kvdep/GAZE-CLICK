import urllib.request, json

queries = [
    "gaze estimation dataset",
    "eye tracking dataset",
    "gaze dataset",
    "appearance based gaze",
    "head pose gaze dataset"
]

found_records = {}

for q in queries:
    encoded_q = urllib.parse.quote(q)
    url = f"https://zenodo.org/api/records?q={encoded_q}&size=25&sort=bestmatch"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as res:
            data = json.loads(res.read().decode("utf-8"))
            hits = data.get("hits", {}).get("hits", [])
            for h in hits:
                rec_id = h.get("id")
                if rec_id not in found_records:
                    meta = h.get("metadata", {})
                    found_records[rec_id] = {
                        "id": rec_id,
                        "title": meta.get("title", ""),
                        "doi": h.get("doi", ""),
                        "access_right": meta.get("access_right", ""),
                        "publication_date": meta.get("publication_date", ""),
                        "description": meta.get("description", "")[:200],
                        "files_count": len(h.get("files", [])),
                        "total_size_mb": sum(f.get("size", 0) for f in h.get("files", [])) / (1024 * 1024)
                    }
    except Exception as e:
        print(f"Error querying {q}: {e}")

print(f"Total unique Zenodo records found: {len(found_records)}")
for rec_id, rec in found_records.items():
    if rec["access_right"] == "open" and rec["files_count"] > 0:
        print(f"ID: {rec_id} | Files: {rec['files_count']} ({rec['total_size_mb']:.1f} MB) | Date: {rec['publication_date']} | Title: {rec['title']}")
