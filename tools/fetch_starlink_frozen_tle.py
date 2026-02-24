# tools/fetch_starlink_frozen_tle.py
from pathlib import Path
from datetime import datetime, timezone
import json
import requests


def main():
    url = "https://celestrak.org/NORAD/elements/gp.php?FORMAT=tle&GROUP=starlink"
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%MZ")
    tle_path = out_dir / f"starlink_frozen_{ts}.tle"
    meta_path = out_dir / f"starlink_frozen_{ts}.meta.json"

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    text = r.text.strip()
    if not text:
        raise RuntimeError("Downloaded TLE is empty")

    tle_path.write_text(text + "\n", encoding="utf-8")

    meta = {
        "source": "CelesTrak",
        "url": url,
        "group": "starlink",
        "format": "tle",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Frozen snapshot for reproducible ATG+LEO experiments"
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    print(f"Saved TLE : {tle_path}")
    print(f"Saved Meta: {meta_path}")
    print(f"Total lines: {len(lines)}  (~{len(lines)//3} sats if name+L1+L2 format)")


if __name__ == "__main__":
    main()