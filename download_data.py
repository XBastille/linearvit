import os
import sys
import argparse
import requests
from tqdm import tqdm


DATASETS = {
    "unlabelled": {
        "url": "https://cernbox.cern.ch/remote.php/dav/public-files/e3pqxcIznqdYyRv/Dataset_Specific_Unlabelled.h5",
        "filename": "Dataset_Specific_Unlabelled.h5",
    },
    "labelled": {
        "url": "https://portal.nersc.gov/cfs/m4392/G25/Dataset_Specific_labelled_full_only_for_2i.h5",
        "filename": "Dataset_Specific_labelled_full_only_for_2i.h5",
    },
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download_file(url, dest_path, chunk_size=8192, quiet=False):
    """Download a file with progress bar. Skips if file already exists."""
    if os.path.exists(dest_path):
        print(f"[SKIP] {dest_path} already exists.")
        return

    print(f"[DOWNLOAD] {url}")
    print(f"       --> {dest_path}")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    tmp_path = dest_path + ".tmp"
    with open(tmp_path, "wb") as f:
        if quiet:
            print(f"Downloading {os.path.basename(dest_path)} ({total_size / 1024**3:.2f} GB) silently. This may take a while...")
            downloaded = 0
            last_printed = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded - last_printed >= 1024**3:
                        print(f"  ... downloaded {downloaded / 1024**3:.1f} GB / {total_size / 1024**3:.2f} GB")
                        last_printed = downloaded
        else:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(dest_path), mininterval=2.0) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    os.rename(tmp_path, dest_path)
    print(f"[DONE] {dest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download CMS E2E datasets")
    parser.add_argument("--labelled-only", action="store_true", help="Download only the labelled dataset")
    parser.add_argument("--unlabelled-only", action="store_true", help="Download only the unlabelled dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print download info without downloading")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Target directory for datasets")
    parser.add_argument("--quiet", action="store_true", help="Minimal output to prevent terminal from crashing")
    args = parser.parse_args()

    targets = []
    if args.labelled_only:
        targets = ["labelled"]
    elif args.unlabelled_only:
        targets = ["unlabelled"]
    else:
        targets = ["labelled", "unlabelled"]

    for key in targets:
        info = DATASETS[key]
        dest = os.path.join(args.data_dir, info["filename"])

        if args.dry_run:
            print(f"[DRY RUN] Would download: {info['url']}")
            print(f"          Target: {dest}")
            continue

        try:
            download_file(info["url"], dest, quiet=args.quiet)
        except Exception as e:
            print(f"[ERROR] Failed to download {key}: {e}", file=sys.stderr)
            print(f"        You can manually download from: {info['url']}")
            print(f"        Place the file at: {dest}")


if __name__ == "__main__":
    main()
