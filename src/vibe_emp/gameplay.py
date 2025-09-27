from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vibe_emp.guess import OpenAIChatModel, guess as run_guess_round


def get_output_dir() -> Path:
    from pathlib import Path as _Path

    project_root: _Path = _Path(__file__).resolve().parents[2]
    return project_root / "output"


def read_candidates(output_dir: Path) -> list[tuple[str, int, Path]]:
    import csv
    from pathlib import Path as _Path

    page_tokens_csv: _Path = output_dir / "page_tokens.csv"
    if not page_tokens_csv.exists():
        raise FileNotFoundError(f"Missing file: {page_tokens_csv}")

    candidates: list[tuple[str, int, _Path]] = []
    with page_tokens_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            page: str = row["page"].strip()
            try:
                length: int = int(row["cl100k_len"])  # token length
            except Exception:
                continue
            file_path: _Path = output_dir / f"{page}.txt"
            if file_path.exists():
                candidates.append((page, length, file_path))

    if not candidates:
        raise FileNotFoundError(f"No candidates with existing .txt found in {output_dir}")

    return [(page, length, _Path(file_path)) for page, length, file_path in candidates]


def compute_weight(token_length: int) -> float:
    base: float = token_length / 5000.0
    if base < 0.1:
        return base
    if base > 1.0:
        return 1.0
    return base


def choose_file(candidates: list[tuple[str, int, Path]]) -> tuple[str, int, Path]:
    import random

    weights: list[float] = [compute_weight(length) for _, length, _ in candidates]
    chosen_index: int = random.choices(range(len(candidates)), weights=weights, k=1)[0]
    return candidates[chosen_index]


def choose_document() -> tuple[str, str]:
    output_dir: Path = get_output_dir()
    candidates: list[tuple[str, int, Path]] = read_candidates(output_dir)
    _, _, chosen_file = choose_file(candidates)
    with chosen_file.open("r", encoding="utf-8") as f:
        document: str = f.read()
    doc_name: str = chosen_file.name[:-4] if chosen_file.name.endswith(".txt") else chosen_file.name
    return document, doc_name


async def advance_game_round(
    *,
    llm: OpenAIChatModel,
    document: str,
    rounds: list[tuple[str, str]],
    user_guess: str,
    last_hint_text: str | None,
) -> tuple[str, bool, list[tuple[str, str]], str | None]:
    if user_guess.strip() == "":
        return "Please reply with text to continue the game.", False, rounds, last_hint_text

    last_note: str = (last_hint_text or "")
    rounds.append((last_note, user_guess))

    kind, content = await run_guess_round(llm=llm, document=document, rounds=rounds)

    if kind == "CORRECT":
        return content, True, rounds, None

    return content, False, rounds, content


async def play(document: str, llm: OpenAIChatModel, fail_label: str) -> None:
    rounds: list[tuple[str, str]] = []

    MAX_ROUNDS: int = 10
    for round_index in range(MAX_ROUNDS):
        kind, content = await run_guess_round(llm, document, rounds)
        if kind == "CORRECT":
            print(content)
            return
        print(content)
        try:
            user_guess: str = input("你的猜测: ")
        except EOFError:
            return
        rounds.append((content, user_guess))

    print(fail_label)
