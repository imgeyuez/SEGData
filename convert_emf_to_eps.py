#!/usr/bin/env python3
"""
Convert all .emf files under a directory tree to .eps files.

Preferred toolchain:
  1) Inkscape (fast, high-fidelity)
  2) LibreOffice --headless to PDF + pdftops -eps
  3) ImageMagick (magick/convert) as a last resort

Usage:
  python convert_emf_to_eps.py [--root DIR] [--force] [--workers N]

Notes:
  - Writes .eps next to each .emf (same basename).
  - Skips existing .eps unless --force is provided or source is newer.
  - Handles filenames with spaces and non-ASCII characters.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Tuple


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def find_emf_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.emf"):
        if p.is_file():
            yield p


def needs_conversion(src: Path, dst: Path, force: bool) -> bool:
    if force:
        return True
    if not dst.exists():
        return True
    try:
        return src.stat().st_mtime > dst.stat().st_mtime
    except OSError:
        return True


def run(cmd: Iterable[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def convert_with_inkscape(src: Path, dst: Path) -> bool:
    inkscape = which("inkscape")
    if not inkscape:
        return False

    # Try modern Inkscape (>=1.0)
    cmd_modern = [
        inkscape,
        str(src),
        "--export-filename",
        str(dst),
    ]
    code, out, err = run(cmd_modern)
    if code == 0 and dst.exists():
        return True

    # Try legacy Inkscape (<=0.92)
    cmd_legacy = [
        inkscape,
        "--without-gui",
        f"--file={src}",
        f"--export-eps={dst}",
    ]
    code, out, err = run(cmd_legacy)
    return code == 0 and dst.exists()


def convert_with_libreoffice_pdftops(src: Path, dst: Path) -> bool:
    soffice = which("soffice") or which("libreoffice")
    pdftops = which("pdftops")
    if not (soffice and pdftops):
        return False

    with tempfile.TemporaryDirectory(prefix="emf2eps_") as tmp:
        tmpdir = Path(tmp)
        # Convert EMF -> PDF via LibreOffice
        cmd_pdf = [
            soffice,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(tmpdir),
            str(src),
        ]
        code, out, err = run(cmd_pdf)
        if code != 0:
            return False

        pdf = tmpdir / (src.stem + ".pdf")
        if not pdf.exists():
            # LibreOffice may name output differently; attempt to locate a PDF
            candidates = list(tmpdir.glob("*.pdf"))
            if len(candidates) == 1:
                pdf = candidates[0]
            else:
                return False

        # Convert PDF -> EPS via pdftops
        cmd_eps = [pdftops, "-eps", str(pdf), str(dst)]
        code, out, err = run(cmd_eps)
        return code == 0 and dst.exists()


def convert_with_imagemagick(src: Path, dst: Path) -> bool:
    magick = which("magick")
    convert = which("convert")

    if magick:
        code, out, err = run([magick, str(src), str(dst)])
        return code == 0 and dst.exists()
    if convert:
        code, out, err = run([convert, str(src), str(dst)])
        return code == 0 and dst.exists()
    return False


def convert_one(src: Path, force: bool = False) -> Tuple[Path, Optional[str]]:
    dst = src.with_suffix(".eps")
    if not needs_conversion(src, dst, force):
        return dst, None

    # Ensure parent directory exists (should, since src exists)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if convert_with_inkscape(src, dst):
        return dst, None
    if convert_with_libreoffice_pdftops(src, dst):
        return dst, None
    if convert_with_imagemagick(src, dst):
        return dst, None

    return dst, (
        "No suitable converter found (need Inkscape, LibreOffice+pdftops, or ImageMagick)."
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to search (default: current working directory)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing or older .eps files",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel workers (default: CPU count)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    root: Path = args.root.resolve()

    if not root.exists() or not root.is_dir():
        print(f"Error: root directory not found: {root}", file=sys.stderr)
        return 2

    emf_files = list(find_emf_files(root))
    if not emf_files:
        print("No .emf files found.")
        return 0

    print(f"Found {len(emf_files)} .emf files under {root}")

    errors = []
    converted = 0
    skipped = 0

    def task(src: Path) -> Tuple[Path, Optional[str]]:
        return convert_one(src, force=args.force)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_to_src = {ex.submit(task, src): src for src in emf_files}
        for fut in concurrent.futures.as_completed(fut_to_src):
            src = fut_to_src[fut]
            try:
                dst, err = fut.result()
            except Exception as e:  # noqa: BLE001
                errors.append((src, str(e)))
                print(f"[FAIL] {src} -> exception: {e}")
                continue

            if not needs_conversion(src, dst, args.force):
                skipped += 1
                print(f"[SKIP] {src} -> {dst}")
            elif err is None and dst.exists():
                converted += 1
                print(f"[OK]   {src} -> {dst}")
            else:
                errors.append((src, err or "unknown error"))
                print(f"[FAIL] {src} -> {err}")

    print(
        f"Done. Converted: {converted}, Skipped: {skipped}, Failed: {len(errors)}",
    )
    if errors:
        print("Some files failed to convert:", file=sys.stderr)
        for src, msg in errors:
            print(f" - {src}: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

