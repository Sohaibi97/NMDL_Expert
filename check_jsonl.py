#!/usr/bin/env python3
import json
from pathlib import Path

FILE = Path("/home/khamlichi/Projekt_NMDL_2/Data/data.jsonl")

def main() -> None:
    bad = 0
    total = 0

    with FILE.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            total += 1
            s = line.strip()
            if not s:
                # leere Zeilen sind optional "ok" – falls du sie nicht willst: bad += 1
                continue
            try:
                obj = json.loads(s)
                # optionaler Minimal-Check: erwartete Struktur
                if not isinstance(obj, dict) or "messages" not in obj:
                    bad += 1
                    print(f"[BAD STRUCT] line {lineno}: missing 'messages'")
            except json.JSONDecodeError as e:
                bad += 1
                preview = (s[:200] + "...") if len(s) > 200 else s
                print(f"[BAD JSON] line {lineno}: {e.msg} (col {e.colno})")
                print(f"          preview: {preview}")

    print(f"\nDone. total lines={total}, bad lines={bad}")

if __name__ == "__main__":
    main()