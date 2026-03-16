from __future__ import annotations

import json
import re
from pathlib import Path


BRACKET_TOKEN_MAP = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
    "-COLON-": ":",
}
WHITESPACE_RE = re.compile(r"\s+")


def clean_sentence(sentence: str) -> str:
    if sentence is None:
        return ""

    cleaned = str(sentence)

    if "\t" in cleaned:
        cleaned = cleaned.split("\t", 1)[0]

    for token, replacement in BRACKET_TOKEN_MAP.items():
        cleaned = cleaned.replace(token, replacement)

    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def clean_articles(articles: dict) -> tuple[dict, dict]:
    if not isinstance(articles, dict):
        return {}, {
            "article_count": 0,
            "sentence_count": 0,
            "empty_sentence_count": 0,
            "changed_sentence_count": 0,
        }

    total_sentences = 0
    empty_sentences = 0
    changed_sentences = 0
    cleaned_articles = {}

    for doc_id, sentences in articles.items():
        if not isinstance(sentences, list):
            cleaned_articles[doc_id] = []
            continue

        cleaned_list = []
        for sentence in sentences:
            original = "" if sentence is None else str(sentence)
            cleaned = clean_sentence(original)

            total_sentences += 1
            if cleaned == "":
                empty_sentences += 1
            if cleaned != original:
                changed_sentences += 1

            cleaned_list.append(cleaned)

        cleaned_articles[doc_id] = cleaned_list

    return cleaned_articles, {
        "article_count": len(cleaned_articles),
        "sentence_count": total_sentences,
        "empty_sentence_count": empty_sentences,
        "changed_sentence_count": changed_sentences,
    }


def output_path_for(input_path: Path, suffix: str, out_dir: Path | None) -> Path:
    if out_dir is not None:
        return out_dir / f"{input_path.stem}{suffix}{input_path.suffix}"
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def process_file(input_path: Path, output_path: Path, limit: int | None = None) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    total_articles = 0
    total_sentences = 0
    empty_sentences = 0
    changed_sentences = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if limit is not None and row_count >= limit:
                break

            raw = line.strip()
            if not raw:
                continue

            row = json.loads(raw)
            cleaned_articles, stats = clean_articles(row.get("articles", {}))
            row["articles"] = cleaned_articles

            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

            row_count += 1
            total_articles += stats["article_count"]
            total_sentences += stats["sentence_count"]
            empty_sentences += stats["empty_sentence_count"]
            changed_sentences += stats["changed_sentence_count"]

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": row_count,
        "articles": total_articles,
        "sentences": total_sentences,
        "empty_sentences": empty_sentences,
        "changed_sentences": changed_sentences,
    }


def main() -> None:
    inputs = [ "data/fever/train_formatted.jsonl", "data/fever/test_formatted.jsonl" ]
    suf = "_cleaned"
    out_dir = None
    results = []

    for input_name in inputs:
        input_path = Path(input_name)
        if not input_path.exists():
            print(f"Skipping missing file: {input_path}")
            continue

        output_path = output_path_for(input_path, suf, out_dir)
        stats = process_file(input_path, output_path)
        results.append(stats)

    if not results:
        print("No files processed.")
        return

    print("Cleanup complete:")
    for stats in results:
        print(stats)


if __name__ == "__main__":
    main()
