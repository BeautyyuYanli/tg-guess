from __future__ import annotations

from csv import DictReader, writer as csv_writer
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote
import argparse

from llama_index.readers.wikipedia import WikipediaReader
from rich import print
import tiktoken


CSV_PATH: str = "wiki_links.csv"
OUTPUT_DIR: str = "output"
BATCH_SIZE: int = 1
LANG_PREFIX: str = "zh"


def read_pages_from_csv(csv_path: str) -> list[str]:
    pages: list[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader: DictReader[str] = DictReader(f)
        for row in reader:
            href: str | None = row.get("href")
            if not href:
                continue
            last_component: str = href.split("/")[-1]
            decoded: str = unquote(last_component)
            if decoded:
                pages.append(decoded)
    return pages


def chunked(iterable: Iterable[str], size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def ensure_output_dir(path: str) -> Path:
    p: Path = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_tokens_cl100k(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def write_page(out_dir: Path, page_title: str, content: str) -> None:
    if not content:
        return
    outfile: Path = out_dir / f"{page_title}.txt"
    outfile.write_text(content, encoding="utf-8")




def download_batch_and_write(reader: WikipediaReader, batch: list[str], out_dir: Path) -> list[str]:
    documents = reader.load_data(pages=batch, lang_prefix=LANG_PREFIX)
    written_pages: list[str] = []
    for index, doc in enumerate(documents):
        texts: list[str] = [
            doc.text_resource.text
            for _unused in [0]
            if doc.text_resource is not None
            and doc.text_resource.text is not None
        ]
        content: str = "\n\n".join(texts) if texts else ""
        page_title: str = batch[index] if index < len(batch) else "page"
        write_page(out_dir, page_title, content)
        written_pages.append(page_title)
    return written_pages


def write_tokens_csv_from_dir(out_dir: Path) -> None:
    csv_path: Path = out_dir / "page_tokens.csv"
    txt_files: list[Path] = sorted(out_dir.glob("*.txt"))
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv_writer(f)
        w.writerow(["page", "cl100k_len"])
        for txt in txt_files:
            page: str = txt.stem
            content: str = txt.read_text(encoding="utf-8")
            token_len: int = count_tokens_cl100k(content)
            w.writerow([page, token_len])


def parse_start_index() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Download Wikipedia pages in batches")
    parser.add_argument("--start-index", type=int, default=0, help="Resume from this zero-based page index")
    args = parser.parse_args()
    start_index: int = max(0, int(args.start_index))
    return start_index


def main() -> None:
    pages: list[str] = read_pages_from_csv(CSV_PATH)
    start_index: int = parse_start_index()
    if start_index >= len(pages):
        print(f"[yellow]Start index {start_index} >= total pages {len(pages)}. Nothing to do.[/yellow]")
        return

    pages_to_process: list[str] = pages[start_index:]
    total: int = len(pages_to_process)
    if total == 0:
        print("[yellow]No pages to download.[/yellow]")
        return

    out_dir: Path = ensure_output_dir(OUTPUT_DIR)
    reader: WikipediaReader = WikipediaReader()

    processed: int = 0
    for batch_index, batch in enumerate(chunked(pages_to_process, BATCH_SIZE), start=1):
        print(f"[cyan]Batch {batch_index}[/cyan] ({len(batch)} items) starting at global index {start_index + processed}")
        written_pages = download_batch_and_write(reader, batch, out_dir)
        for page_title in written_pages:
            processed += 1
            print(f"  [green][{processed}/{total}][/green] Downloaded: {page_title}")

    write_tokens_csv_from_dir(out_dir)
    print(f"[bold green]Done.[/bold green] Wrote files to {out_dir}")


if __name__ == "__main__":
    main()
