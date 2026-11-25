"""Download BIRD Mini-Dev (SQLite part) into ./data (or $BIRD_DATA_DIR).

- Questions, official baseline prompt and schemas: Hugging Face `birdsql/bird_mini_dev`.
- SQLite databases and column description files: minidev.zip from the BIRD team
  (about 800 MB; only the SQLite databases are extracted).

BIRD data is released under CC BY-SA 4.0.
Usage:  python -m text2sql_agent.download_data
"""
import shutil
import urllib.request
import zipfile

from .common import DATA_DIR, DB_ROOT, OFFICIAL_SCHEMAS_FILE, PROMPT_TEMPLATE_FILE, QUESTIONS_FILE

HF = "https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/main/"
FILES = {
    QUESTIONS_FILE: HF + "data/mini_dev_sqlite.jsonl",
    PROMPT_TEMPLATE_FILE: HF + "baseline_inference/prompt_template.txt",
    OFFICIAL_SCHEMAS_FILE: HF + "baseline_inference/schemas/sqlite.json",
}
MINIDEV_ZIP = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
DB_MARKER = "MINIDEV/dev_databases/"   # zip entries look like minidev/MINIDEV/dev_databases/<db>/...


def fetch(url: str, dest):
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for dest, url in FILES.items():
        if not dest.exists():
            fetch(url, dest)
    if DB_ROOT.exists() and any(DB_ROOT.iterdir()):
        print(f"{DB_ROOT} already exists, skipping databases")
        return
    zip_path = DATA_DIR / "minidev.zip"
    if not zip_path.exists():
        fetch(MINIDEV_ZIP, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            name = m.filename.replace("\\", "/")
            pos = name.find(DB_MARKER)
            if pos != -1 and not m.is_dir() and "__MACOSX" not in name:
                target = DB_ROOT / name[pos + len(DB_MARKER):]
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    print(f"databases extracted to {DB_ROOT}; you can delete {zip_path}")


if __name__ == "__main__":
    main()
