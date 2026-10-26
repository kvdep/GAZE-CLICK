import os
import sys
import subprocess
import urllib.request
import json
import csv

sys.stdout.reconfigure(encoding='utf-8')

SHEET_ID = "1J4LHhZXiZYnBFjymL6ayOt2jAafnt4sISfhsUsNiKLo"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit?usp=sharing"
CSV_PATH = os.path.join("docs", "leaderboard.csv")
TSV_PATH = os.path.join("docs", "leaderboard.tsv")

def copy_tsv_to_clipboard() -> bool:
    if not os.path.exists(TSV_PATH):
        print(f"Error: {TSV_PATH} not found.")
        return False
    try:
        subprocess.run(["powershell", "-Command", f"Get-Content {TSV_PATH} -Raw | Set-Clipboard"], check=True)
        print(f"Leaderboard TSV successfully copied to Windows clipboard.")
        print(f"Target Sheet URL: {SHEET_URL}")
        print("To update sheet manually: Open sheet, select cell A1, and press Ctrl+V.")
        return True
    except Exception as e:
        print(f"Clipboard copy error: {e}")
        return False

def check_leaderboard_files() -> bool:
    if not os.path.exists(CSV_PATH) or not os.path.exists(TSV_PATH):
        print("Leaderboard files missing. Running scripts/export_leaderboard.py...")
        subprocess.run([sys.executable, "scripts/export_leaderboard.py"], check=True)
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
        print(f"CSV Validation: {len(rows)} model rows, {len(header)} columns.")
    with open(TSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        tsv_header = next(reader)
        tsv_rows = list(reader)
        print(f"TSV Validation: {len(tsv_rows)} model rows, {len(tsv_header)} columns.")
    return True

def sync_via_webhook(webhook_url: str) -> bool:
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        all_data = list(reader)
    payload = json.dumps({"sheet_id": SHEET_ID, "values": all_data}).encode("utf-8")
    req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            res_body = resp.read().decode("utf-8")
            print(f"Webhook response: {res_body}")
            return True
    except Exception as e:
        print(f"Webhook error: {e}")
        return False

if __name__ == "__main__":
    check_leaderboard_files()
    if len(sys.argv) > 2 and sys.argv[1] == "--webhook":
        sync_via_webhook(sys.argv[2])
    else:
        copy_tsv_to_clipboard()
