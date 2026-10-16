"""
V2 — automatic folder watcher
==============================
Watches data/drop/ for new data files (csv, txt, xlsx, xls).
When one appears and finishes copying, it runs the V1 pipeline on it
and writes the report to data/output/<name>/.

Usage:
    python src/v2_watcher.py [--watch DIR] [--out DIR] [--once]

    --once   process files already in the folder, then exit
             (default: keep watching until Ctrl+C)

Needs: pip install watchdog
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    sys.exit("The 'watchdog' package is not installed. "
             "Run: pip install watchdog")

import core

SETTLE_ROUNDS = 2      # consecutive stable size checks before processing
POLL_INTERVAL = 1      # seconds between size checks
SETTLE_TIMEOUT = 120   # give up waiting after this many seconds


def _is_temp_file(path: Path):
    name = path.name
    return (name.startswith("~$") or name.startswith(".~")
            or name.endswith(".tmp") or name.endswith(".crdownload")
            or name.endswith(".part"))


def _wait_until_settled(path: Path):
    """Wait until the file stops growing (copy finished)."""
    start = time.time()
    last_size, stable_rounds = -1, 0
    while time.time() - start < SETTLE_TIMEOUT:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == last_size and size > 0:
            stable_rounds += 1
            if stable_rounds >= SETTLE_ROUNDS:
                return True
        else:
            stable_rounds = 0
        last_size = size
        time.sleep(POLL_INTERVAL)
    return False


def process_file(path: Path, out_root: Path):
    ext = path.suffix.lower()
    if ext not in core.SUPPORTED_EXTENSIONS:
        print(f"  skip: '{ext}' is not a supported data file")
        return
    if _is_temp_file(path):
        print("  skip: temporary file")
        return
    print("  waiting for the copy to finish…")
    if not _wait_until_settled(path):
        print("  skip: file never settled")
        return
    out_dir = out_root / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = core.analyze(path, name=path.name)
    except Exception as e:
        print(f"  ERROR: could not analyze ({e})")
        return
    df = result["df"]
    df.to_csv(out_dir / "cleaned.csv", index=False)
    (out_dir / "cleaning_log.txt").write_text(
        "\n".join(result["clean_lines"]), encoding="utf-8")
    if not result["desc"].empty:
        result["desc"].round(4).to_csv(out_dir / "summary_statistics.csv")
    result["missing"].to_csv(out_dir / "missing_values.csv")
    if not result["top_cats"].empty:
        result["top_cats"].to_csv(out_dir / "top_categories.csv", index=False)
    (out_dir / "report.md").write_text(result["report"], encoding="utf-8")
    core.save_figures(result["figs"], out_dir / "charts")
    print(f"  done → {out_dir}/ "
          f"({df.shape[0]} rows × {df.shape[1]} cols)")


class DropHandler(FileSystemEventHandler):
    def __init__(self, out_root):
        self.out_root = out_root
        self._seen = set()  # (path, mtime) so re-dropped files re-run

    def _handle(self, path_str):
        path = Path(path_str)
        try:
            key = (str(path), path.stat().st_mtime_ns)
        except FileNotFoundError:
            return
        if key in self._seen or not path.is_file():
            return
        self._seen.add(key)
        print(f"\n📥 New file: {path.name}")
        process_file(path, self.out_root)

    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_moved(self, event):
        # catches files moved/dragged into the folder
        if not event.is_directory:
            self._handle(event.dest_path)


def main():
    ap = argparse.ArgumentParser(
        description="V2: watch a folder and auto-analyze new data files.")
    ap.add_argument("--watch", default="data/drop",
                    help="folder to watch (default: data/drop)")
    ap.add_argument("--out", default="data/output",
                    help="where reports go (default: data/output)")
    ap.add_argument("--once", action="store_true",
                    help="process files already in the folder, then exit")
    args = ap.parse_args()

    watch_dir = Path(args.watch)
    watch_dir.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.once:
        files = sorted(p for p in watch_dir.iterdir() if p.is_file())
        print(f"Processing {len(files)} file(s) in {watch_dir}/ …")
        for f in files:
            print(f"\n📥 {f.name}")
            process_file(f, out_root)
        print("\nDone.")
        return

    print(f"👀 Watching {watch_dir.resolve()}/ — drop a data file in!")
    print(f"   Reports go to {out_root.resolve()}/")
    print("   Press Ctrl+C to stop.\n")

    handler = DropHandler(out_root)
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping watcher…")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
