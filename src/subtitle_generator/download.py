"""Download LOC MARC bulk data files."""

import gzip
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

import click

BASE_URL = "https://www.loc.gov/cds/downloads/MDSConnect"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
USER_AGENT = "subtitle-generator/0.1.0 (research project; LOC MARC open-access)"

# 2016 retrospective: 43 parts, UTF-8 encoding
TOTAL_PARTS = 43
CHUNK_SIZE = 1024 * 1024  # 1 MB


def _part_url(part: int) -> str:
    return f"{BASE_URL}/BooksAll.2016.part{part:02d}.utf8.gz"


def _part_gz_path(part: int) -> Path:
    return DATA_DIR / f"BooksAll.2016.part{part:02d}.utf8.gz"


def _part_mrc_path(part: int) -> Path:
    return DATA_DIR / f"BooksAll.2016.part{part:02d}.utf8.mrc"


def download_part(part: int, decompress: bool = True, force: bool = False) -> Path:
    """Download a single MARC part file. Returns path to the .mrc file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    mrc_path = _part_mrc_path(part)
    gz_path = _part_gz_path(part)

    if mrc_path.exists() and not force:
        click.echo(f"Part {part:02d}: already extracted at {mrc_path}")
        return mrc_path

    if gz_path.exists() and not force:
        click.echo(f"Part {part:02d}: already downloaded, skipping download")
    else:
        url = _part_url(part)
        click.echo(f"Part {part:02d}: downloading from {url}")
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req) as response, open(gz_path, "wb") as f_out:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                f_out.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)
                else:
                    mb = downloaded / (1024 * 1024)
                    print(f"\r  {mb:.0f} MB", end="", flush=True)
            print()  # newline after progress

    if decompress:
        click.echo(f"Part {part:02d}: decompressing...")
        with gzip.open(gz_path, "rb") as f_in, open(mrc_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()  # remove .gz to save space
        click.echo(f"Part {part:02d}: ready at {mrc_path}")
        return mrc_path

    return gz_path


def parse_parts_arg(parts_str: str) -> list[int]:
    """Parse a parts specifier like '1', '1-5', '1,3,5', or 'all'."""
    if parts_str.lower() == "all":
        return list(range(1, TOTAL_PARTS + 1))

    result = []
    for segment in parts_str.split(","):
        segment = segment.strip()
        if "-" in segment:
            start, end = segment.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(segment))

    return sorted(set(p for p in result if 1 <= p <= TOTAL_PARTS))
