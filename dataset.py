from pathlib import Path
import argparse
import json

from datasets import Dataset, DatasetDict, load_dataset


FEVER_DIR = Path("data/fever")
WIKI_DIR = FEVER_DIR / "wiki-pages" / "wiki-pages"


def _normalize_nested_values(value):
	if isinstance(value, list):
		return [_normalize_nested_values(item) for item in value]
	if value is None:
		return None
	return str(value)


def _load_claims_jsonl(path: Path) -> Dataset:
	rows = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue

			item = json.loads(line)
			rows.append(
				{
					"id": item.get("id"),
					"claim": item.get("claim"),
					"label": item.get("label"),
					"verifiable": item.get("verifiable"),
					"evidence": _normalize_nested_values(item.get("evidence", [])),
				}
			)

	return Dataset.from_list(rows)


def _normalize_doc_id(doc_id: str) -> str:
	if not doc_id:
		return ""

	normalized = str(doc_id)
	normalized = normalized.replace("-lrb-", "(")
	normalized = normalized.replace("-rrb-", ")")
	normalized = normalized.replace("-lsb-", "[")
	normalized = normalized.replace("-rsb-", "]")
	normalized = normalized.replace("-lcb-", "{")
	normalized = normalized.replace("-rcb-", "}")
	normalized = normalized.replace("-colon-", ":")
	return normalized


def _extract_evidence_doc_ids(claim_row: dict) -> list[str]:
	doc_ids = []
	seen = set()

	for evidence_set in claim_row.get("evidence", []):
		if not isinstance(evidence_set, list):
			continue

		for evidence_item in evidence_set:
			if not isinstance(evidence_item, list) or len(evidence_item) < 3:
				continue

			doc_id = _normalize_doc_id(evidence_item[2])
			if not doc_id or doc_id in seen:
				continue

			seen.add(doc_id)
			doc_ids.append(doc_id)

	return doc_ids


def _extract_evidence_pairs(claim_row: dict) -> list[dict]:
	evidence_pairs = []
	seen = set()

	for evidence_set in claim_row.get("evidence", []):
		if not isinstance(evidence_set, list):
			continue

		for evidence_item in evidence_set:
			if not isinstance(evidence_item, list) or len(evidence_item) < 4:
				continue

			doc_id = _normalize_doc_id(evidence_item[2])
			sentence_raw = evidence_item[3]

			if not doc_id:
				continue

			try:
				sentence_id = int(sentence_raw)
			except (TypeError, ValueError):
				continue

			pair_key = (doc_id, sentence_id)
			if pair_key in seen:
				continue

			seen.add(pair_key)
			evidence_pairs.append(
				{
					"doc_id": doc_id,
					"sentence_id": sentence_id,
				}
			)

	return evidence_pairs


def _parse_wiki_lines(lines_value: str) -> list[str]:
	if not lines_value:
		return []

	parsed = {}
	max_index = -1

	for line in str(lines_value).split("\n"):
		if not line:
			continue

		parts = line.split("\t", 1)
		if len(parts) != 2:
			continue

		index_text, sentence = parts
		try:
			index = int(index_text)
		except ValueError:
			continue

		parsed[index] = sentence
		if index > max_index:
			max_index = index

	if max_index < 0:
		return []

	return [parsed.get(i, "") for i in range(max_index + 1)]


def _build_wiki_lookup(wiki_split: Dataset, needed_doc_ids: set[str]) -> dict[str, list[str]]:
	lookup = {}
	if not needed_doc_ids:
		return lookup

	pending = set(needed_doc_ids)
	for row in wiki_split:
		row_id = _normalize_doc_id(row.get("id", ""))
		if row_id in pending:
			sentences = _parse_wiki_lines(row.get("lines", ""))
			if not sentences:
				text_value = str(row.get("text", "")).strip()
				sentences = [text_value] if text_value else []

			lookup[row_id] = sentences
			pending.remove(row_id)
			if not pending:
				break

	return lookup


def _collect_needed_doc_ids(claims_split: Dataset) -> set[str]:
	needed = set()
	for claim_row in claims_split:
		for doc_id in _extract_evidence_doc_ids(claim_row):
			needed.add(doc_id)
	return needed


def _format_claim_record(claim_row: dict, wiki_lookup: dict[str, list[str]]) -> dict:
	evidence_pairs = _extract_evidence_pairs(claim_row)
	ordered_doc_ids = []
	seen = set()
	for pair in evidence_pairs:
		doc_id = pair["doc_id"]
		if doc_id not in seen:
			seen.add(doc_id)
			ordered_doc_ids.append(doc_id)

	articles = {doc_id: wiki_lookup.get(doc_id, []) for doc_id in ordered_doc_ids}

	return {
		"id": claim_row.get("id"),
		"claim": claim_row.get("claim"),
		"verifiable": claim_row.get("verifiable"),
		"label": claim_row.get("label"),
		"evidence": evidence_pairs,
		"articles": articles,
	}


def export_formatted_split(claims_split: Dataset, wiki_lookup: dict[str, list[str]], output_path: Path) -> dict:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	rows_written = 0
	claims_without_articles = 0

	with output_path.open("w", encoding="utf-8") as handle:
		for claim_row in claims_split:
			record = _format_claim_record(claim_row, wiki_lookup)
			has_any_articles = any(record["articles"].get(doc_id) for doc_id in record["articles"])
			if not has_any_articles and record["evidence"]:
				claims_without_articles += 1

			handle.write(json.dumps(record) + "\n")
			rows_written += 1

	return {
		"output_path": str(output_path),
		"rows_written": rows_written,
		"claims_without_articles": claims_without_articles,
	}


def retrieve_evidence_articles(claim_row: dict, wiki_split: Dataset) -> dict:
	target_doc_ids = _extract_evidence_doc_ids(claim_row)
	if not target_doc_ids:
		return {
			"claim_id": claim_row.get("id"),
			"claim": claim_row.get("claim"),
			"label": claim_row.get("label"),
			"evidence_doc_ids": [],
			"articles": [],
			"missing_doc_ids": [],
		}

	pending = set(target_doc_ids)
	found_articles = {}

	for row in wiki_split:
		row_id = _normalize_doc_id(row.get("id", ""))
		if row_id in pending:
			found_articles[row_id] = {
				"doc_id": row_id,
				"text": row.get("text", ""),
				"lines": row.get("lines", ""),
			}
			pending.remove(row_id)
			if not pending:
				break

	return {
		"claim_id": claim_row.get("id"),
		"claim": claim_row.get("claim"),
		"label": claim_row.get("label"),
		"evidence_doc_ids": target_doc_ids,
		"articles": [found_articles[doc_id] for doc_id in target_doc_ids if doc_id in found_articles],
		"missing_doc_ids": sorted(pending),
	}


def build_fever_datasets() -> DatasetDict:
	claims_train = _load_claims_jsonl(FEVER_DIR / "train.jsonl")
	claims_test = _load_claims_jsonl(FEVER_DIR / "test.jsonl")

	wiki_files = sorted(str(path) for path in WIKI_DIR.glob("wiki-*.jsonl"))
	wiki = load_dataset("json", data_files={"wiki": wiki_files})

	return DatasetDict(
		{
			"claims_train": claims_train,
			"claims_test": claims_test,
			"wiki": wiki["wiki"],
		}
	)




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--export-formatted", action="store_true")
	args = parser.parse_args()

	fever = build_fever_datasets()

	print(fever)
	print("\nSplit sizes:")
	for split_name, split_data in fever.items():
		print(f"- {split_name}: {len(split_data)} rows")

	print("\nColumn names:")
	for split_name, split_data in fever.items():
		print(f"- {split_name}: {split_data.column_names}")

	print("\nSample claim row (claims_train[0]):")
	print(fever["claims_train"][0])

	wiki_sample = next((row for row in fever["wiki"] if row.get("id")), fever["wiki"][0])
	print("\nSample wiki row (first non-empty id):")
	print(wiki_sample)

	sample_claim_row = fever["claims_train"][0]
	oracle_articles = retrieve_evidence_articles(sample_claim_row, fever["wiki"])
	print("\nEvidence-based retriever sample (all wiki pages referenced by claim evidence):")
	print(
		{
			"claim_id": oracle_articles["claim_id"],
			"label": oracle_articles["label"],
			"evidence_doc_ids": oracle_articles["evidence_doc_ids"],
			"found_articles": len(oracle_articles["articles"]),
			"missing_doc_ids": oracle_articles["missing_doc_ids"],
		}
	)

	for article in oracle_articles["articles"]:
		print(
			{
				"doc_id": article["doc_id"],
				"text_preview": article["text"][:200],
			}
		)

	if args.export_formatted:
		all_needed_doc_ids = _collect_needed_doc_ids(fever["claims_train"])
		all_needed_doc_ids.update(_collect_needed_doc_ids(fever["claims_test"]))

		wiki_lookup = _build_wiki_lookup(fever["wiki"], all_needed_doc_ids)

		train_stats = export_formatted_split(
			fever["claims_train"],
			wiki_lookup,
			FEVER_DIR / "train_formatted.jsonl",
		)
		test_stats = export_formatted_split(
			fever["claims_test"],
			wiki_lookup,
			FEVER_DIR / "test_formatted.jsonl",
		)

		print("\nFormatted export complete:")
		print(train_stats)
		print(test_stats)