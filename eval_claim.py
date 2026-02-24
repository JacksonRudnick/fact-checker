import argparse
import re
from pathlib import Path

from gat_verifier import verify_claim_with_gat
from sentence_ranker import select_sentence


def split_sentences(article_text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", article_text.strip())
    sentences = [part.strip() for part in parts if part and part.strip()]
    return sentences if sentences else [article_text.strip()]


def read_multiline(prompt: str) -> str:
    print(prompt)
    print("Finish with a single line containing: END")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate claim with sentence ranker + GAT verifier.")
    parser.add_argument("--sentence-ranker-path", default="outputs/bert-fact-verifier/")
    parser.add_argument("--gat-verifier-path", default="outputs/gat-fact-verifier/")
    parser.add_argument("--claim", default=None)
    parser.add_argument("--article-text", default=None)
    parser.add_argument("--article-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sentence_ranker_path = Path(args.sentence_ranker_path)
    gat_verifier_path = Path(args.gat_verifier_path)

    if not sentence_ranker_path.exists():
        raise FileNotFoundError(f"Sentence ranker model path not found: {sentence_ranker_path}")
    if not gat_verifier_path.exists():
        raise FileNotFoundError(f"GAT verifier model path not found: {gat_verifier_path}")

    claim = args.claim.strip() if args.claim else input("Claim: ").strip()

    if args.article_text:
        article_text = args.article_text.strip()
    elif args.article_file:
        article_text = Path(args.article_file).read_text(encoding="utf-8").strip()
    else:
        article_text = read_multiline("Paste article text:")

    if not claim:
        raise ValueError("Claim cannot be empty")
    if not article_text:
        raise ValueError("Article text cannot be empty")

    sentences = split_sentences(article_text)
    if not sentences:
        raise ValueError("No sentences found in article text")

    sentence_idx = select_sentence(str(sentence_ranker_path), claim, sentences)
    selected_sentence = sentences[sentence_idx]

    verification = verify_claim_with_gat(
        model_path=str(gat_verifier_path),
        claim=claim,
        sentence=selected_sentence,
    )

    print("\n=== Two-Part Pipeline Result ===")
    print(f"Claim: {claim}")
    print(f"Total sentences in article: {len(sentences)}")
    print(f"Selected sentence index: {sentence_idx}")
    print(f"Selected sentence: {selected_sentence}")

    print("\n=== GAT Verification ===")
    print(f"Prediction: {verification['prediction']}")
    print(f"Confidence: {verification['confidence']:.4f}")
    if "scores" in verification:
        print(f"Scores: {verification['scores']}")


if __name__ == "__main__":
    main()
