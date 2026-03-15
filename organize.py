#!/usr/bin/env python3

import os
import re
import shutil
from pathlib import Path


def get_person_id_real(filename):
    match = re.match(r"id(\d+)_\d+\.mp4", filename)
    if match:
        return int(match.group(1))
    return None


def get_person_id_fake(filename):
    match = re.match(r"id\d+_id(\d+)_\d+\.mp4", filename)
    if match:
        return int(match.group(1))
    return None


def organize(archive_root: str, output_root: str, dry_run: bool = False):
    archive_root = Path(archive_root)
    output_root = Path(output_root)

    real_dir = archive_root / "Celeb-real"
    fake_dir = archive_root / "Celeb-synthesis"

    if not real_dir.exists() and not fake_dir.exists():
        print(f"ERROR: Neither Celeb-real nor Celeb-synthesis found in {archive_root}")
        return

    stats = {"real": 0, "fake": 0, "skipped": 0}

    if real_dir.exists():
        print(f"\nProcessing Celeb-real from: {real_dir}")
        for f in sorted(real_dir.iterdir()):
            if not f.is_file() or not f.suffix == ".mp4":
                continue
            person_id = get_person_id_real(f.name)
            if person_id is None:
                print(f"  [SKIP] Unrecognized filename: {f.name}")
                stats["skipped"] += 1
                continue
            dest_dir = output_root / "Celeb-real" / f"person_{person_id}"
            dest_file = dest_dir / f.name
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest_file)
            print(f"  [real] {f.name} → Celeb-real/person_{person_id}/")
            stats["real"] += 1
    else:
        print("  Celeb-real folder not found, skipping.")

    if fake_dir.exists():
        print(f"\nProcessing Celeb-synthesis from: {fake_dir}")
        for f in sorted(fake_dir.iterdir()):
            if not f.is_file() or not f.suffix == ".mp4":
                continue
            person_id = get_person_id_fake(f.name)
            if person_id is None:
                print(f"  [SKIP] Unrecognized filename: {f.name}")
                stats["skipped"] += 1
                continue
            dest_dir = output_root / "Celeb-fake" / f"person_{person_id}"
            dest_file = dest_dir / f.name
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest_file)
            print(f"  [fake] {f.name} → Celeb-fake/person_{person_id}/")
            stats["fake"] += 1
    else:
        print("  Celeb-synthesis folder not found, skipping.")

    print("\n" + "=" * 50)
    if dry_run:
        print("DRY RUN complete (no files were copied).")
    else:
        print("Done!")
    print(f"  Real videos organized : {stats['real']}")
    print(f"  Fake videos organized : {stats['fake']}")
    print(f"  Skipped (bad names)   : {stats['skipped']}")
    print(f"\nOutput folder: {output_root.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Organize archive-2 videos by person ID."
    )
    parser.add_argument(
        "--input",
        default="archive-2",
        help="Path to the archive-2 folder (default: ./archive-2)",
    )
    parser.add_argument(
        "--output",
        default="archive-2-organized",
        help="Path to the output folder (default: ./archive-2-organized)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would happen without copying any files",
    )
    args = parser.parse_args()

    organize(args.input, args.output, dry_run=args.dry_run)
