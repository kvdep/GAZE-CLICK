import os
import sys
import subprocess
import json
import csv
import time
import urllib.request

sys.stdout.reconfigure(encoding='utf-8')

SHEET_ID = "1J4LHhZXiZYnBFjymL6ayOt2jAafnt4sISfhsUsNiKLo"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit?usp=sharing"
CSV_PATH = os.path.join("docs", "leaderboard.csv")
TSV_PATH = os.path.join("docs", "leaderboard.tsv")
EXPORT_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

def copy_tsv_to_clipboard() -> bool:
    if not os.path.exists(TSV_PATH):
        print(f"Error: {TSV_PATH} not found.")
        return False
    try:
        subprocess.run(["powershell", "-Command", f"Get-Content {TSV_PATH} -Raw | Set-Clipboard"], check=True)
        print(f"Leaderboard TSV copied to Windows clipboard.")
        return True
    except Exception as e:
        print(f"Clipboard copy error: {e}")
        return False

def verify_cloud_sync() -> bool:
    req = urllib.request.Request(
        EXPORT_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode("utf-8")
            lines = [l for l in data.splitlines() if l.strip()]
            print(f"Cloud Sheet Verification: {len(lines)} rows synchronized on Google Drive.")
            return len(lines) > 1
    except Exception as e:
        print(f"Cloud verification note: {e}")
        return False

def sync_via_selenium(headless: bool = True) -> bool:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    print(f"Starting browser automation sync to: {SHEET_URL}")
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-gpu")
    opts.add_experimental_option("prefs", {
        "profile.content_settings.exceptions.clipboard": {
            "[*.]google.com,*": {"setting": 1}
        }
    })

    driver = webdriver.Chrome(options=opts)
    try:
        driver.execute_cdp_cmd("Browser.grantPermissions", {
            "permissions": ["clipboardReadWrite", "clipboardSanitizedWrite"],
            "origin": "https://docs.google.com"
        })
        driver.get(SHEET_URL)
        time.sleep(6)

        with open(TSV_PATH, "r", encoding="utf-8") as f:
            tsv_data = f.read()

        driver.execute_script("navigator.clipboard.writeText(arguments[0]);", tsv_data)
        time.sleep(1)

        driver.execute_cdp_cmd("Input.dispatchMouseEvent", {
            "type": "mousePressed", "x": 60, "y": 170, "button": "left", "clickCount": 1
        })
        driver.execute_cdp_cmd("Input.dispatchMouseEvent", {
            "type": "mouseReleased", "x": 60, "y": 170, "button": "left", "clickCount": 1
        })
        time.sleep(1)

        driver.execute_cdp_cmd("Input.dispatchKeyEvent", {
            "type": "keyDown", "modifiers": 2, "windowsVirtualKeyCode": 86, "key": "v", "code": "KeyV"
        })
        driver.execute_cdp_cmd("Input.dispatchKeyEvent", {
            "type": "keyUp", "modifiers": 2, "windowsVirtualKeyCode": 86, "key": "v", "code": "KeyV"
        })
        time.sleep(6)

        print("Data paste dispatched via CDP. Waiting for cloud persistence...")
        return True
    except Exception as e:
        print(f"Selenium automation error: {e}")
        return False
    finally:
        driver.quit()

def check_leaderboard_files() -> bool:
    if not os.path.exists(CSV_PATH) or not os.path.exists(TSV_PATH):
        print("Generating leaderboard files...")
        subprocess.run([sys.executable, "scripts/export_leaderboard.py"], check=True)
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
        print(f"Local CSV: {len(rows)-1} models, {len(rows[0])} columns.")
    return True

if __name__ == "__main__":
    check_leaderboard_files()
    success = sync_via_selenium(headless=True)
    copy_tsv_to_clipboard()
    if success:
        time.sleep(3)
        verify_cloud_sync()
