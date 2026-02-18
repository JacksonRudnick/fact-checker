from tqdm import tqdm
import csv
import json
import os
import re
import time
import unicodedata
from urllib.parse import quote
from urllib.request import urlopen
import wikipediaapi

class FetchDataset:
    def __init__(self, filepath: str, force_reprocess: bool = False):
        self.filepath = filepath
        self.data = []
        self.processed_path = self._get_processed_path(filepath)

        if os.path.exists(self.processed_path) and not force_reprocess:
            with open(self.processed_path, 'r') as f:
                for line in tqdm(f, desc="Reading processed file"):
                    self.data.append(json.loads(line))
        else:
            self.wiki_wiki = wikipediaapi.Wikipedia('fact-checker (jrudnick@cub.uca.edu)', 'en')

            with open(filepath, 'r') as f:
                for line in tqdm(f, desc="Reading input file"):
                    self.data.append(json.loads(line))
            self.process()
            self.fetch_articles()
            self.save(self.processed_path)

    def _get_processed_path(self, filepath: str) -> str:
        file_name = os.path.basename(filepath)
        return os.path.join('data/fever', f'processed_{file_name}')

    def _get_split_name(self) -> str:
        file_name = os.path.basename(self.filepath)
        if file_name.endswith('.jsonl'):
            return file_name[:-6]
        return file_name

    def _normalize_doc_id(self, doc_id: str) -> str:
        token_map = {
            '-LRB-': '(',
            '-RRB-': ')',
            '-LSB-': '[',
            '-RSB-': ']',
            '-LCB-': '{',
            '-RCB-': '}',
            '-COLON-': ':',
        }
        normalized = doc_id
        for token, replacement in token_map.items():
            normalized = normalized.replace(token, replacement)
        normalized = unicodedata.normalize('NFC', normalized)
        return normalized

    def _doc_id_candidates(self, doc_id: str):
        normalized = self._normalize_doc_id(doc_id)
        candidates = [normalized]
        if '_' in normalized:
            candidates.append(normalized.replace('_', ' '))
        ascii_normalized = unicodedata.normalize('NFKD', normalized).encode('ascii', 'ignore').decode('ascii')
        if ascii_normalized and ascii_normalized not in candidates:
            candidates.append(ascii_normalized)
            if '_' in ascii_normalized:
                candidates.append(ascii_normalized.replace('_', ' '))
        deduped = []
        seen = set()
        for candidate in candidates:
            if candidate not in seen:
                deduped.append(candidate)
                seen.add(candidate)
        return deduped

    def _split_sentences(self, text: str):
        parts = re.split(r'\n+|(?<=[.!?])\s+', text)
        return [part.strip() for part in parts if part and part.strip()]

    def _fetch_with_wikipediaapi(self, title: str):
        page = self.wiki_wiki.page(title)
        if page.exists() and page.text:
            return self._split_sentences(page.text)
        return None

    def _fetch_with_mediawiki_api(self, title: str):
        encoded_title = quote(title, safe='')
        api_url = (
            'https://en.wikipedia.org/w/api.php'
            '?action=query'
            '&format=json'
            '&prop=extracts'
            '&explaintext=1'
            '&redirects=1'
            '&titles=' + encoded_title
        )

        with urlopen(api_url, timeout=15) as response:
            payload = json.loads(response.read().decode('utf-8'))

        pages = payload.get('query', {}).get('pages', {})
        if not pages:
            return None

        page_data = next(iter(pages.values()))
        extract = page_data.get('extract', '')
        if not extract:
            return None

        return self._split_sentences(extract)

    def _fetch_article_sentences(self, doc_id: str, retries: int = 3):
        candidates = self._doc_id_candidates(doc_id)

        for candidate in candidates:
            for attempt in range(retries):
                try:
                    sentences = self._fetch_with_wikipediaapi(candidate)
                    if sentences:
                        return sentences

                    sentences = self._fetch_with_mediawiki_api(candidate)
                    if sentences:
                        return sentences
                except Exception:
                    if attempt < retries - 1:
                        time.sleep(0.5 * (attempt + 1))

        return ['No article found']

    def process(self):
        for item in tqdm(self.data, desc="Processing data"):
            label = item['label']
            if label == 'SUPPORTS':
                label_id = 0
            elif label == 'REFUTES':
                label_id = 1
            else:
                label_id = 2

            claim = item['claim'].lower()

            evidence_sets = []
            evidence_flat = []
            seen_pairs = set()
            for ev_set in item['evidence']:
                processed_set = []
                for ev in ev_set:
                    doc_id = ev[2]
                    sentence_id = ev[3]
                    if doc_id is None:
                        continue

                    doc_id = self._normalize_doc_id(doc_id)
                    evidence_record = {
                        'doc_id': doc_id,
                        'sentence_id': sentence_id
                    }
                    processed_set.append(evidence_record)

                    pair_key = (doc_id, sentence_id)
                    if pair_key not in seen_pairs:
                        evidence_flat.append(evidence_record)
                        seen_pairs.add(pair_key)

                if processed_set:
                    evidence_sets.append(processed_set)

            item['id'] = item['id']
            item['claim'] = claim
            item['label'] = label
            item['label_id'] = label_id
            item['evidence'] = evidence_sets
            item['evidence_flat'] = evidence_flat
            item['doc_ids'] = sorted({ev['doc_id'] for ev in evidence_flat})
            item['verifiable'] = item.get('verifiable', 'UNKNOWN')

    def fetch_articles(self):
        self.wiki_wiki = wikipediaapi.Wikipedia('fact-checker (jrudnick@cub.uca.edu)', 'en')
        fetch_failures = 0
        fetch_successes = 0

        for claim in tqdm(self.data, desc="Fetching articles"):
            articles = {}
            for doc_id in claim['doc_ids']:
                if doc_id in articles:
                    continue

                sentences = self._fetch_article_sentences(doc_id)
                if len(sentences) == 1 and sentences[0] == 'No article found':
                    fetch_failures += 1
                else:
                    fetch_successes += 1
                articles[doc_id] = sentences

            claim['articles'] = articles

        print(f"Article fetch summary: {fetch_successes} succeeded, {fetch_failures} failed")

    def export_ml_files(self, output_dir='data/fever'):
        split_name = self._get_split_name()
        claims_path = os.path.join(output_dir, f'ml_claims_{split_name}.jsonl')
        sentence_rows_path = os.path.join(output_dir, f'ml_sentence_rows_{split_name}.csv')

        os.makedirs(output_dir, exist_ok=True)

        with open(claims_path, 'w') as claims_file:
            for item in tqdm(self.data, desc="Saving ML claims JSONL"):
                claims_record = {
                    'id': item['id'],
                    'claim': item['claim'],
                    'label': item['label'],
                    'label_id': item['label_id'],
                    'verifiable': item['verifiable'],
                    'doc_ids': item['doc_ids'],
                    'evidence': item['evidence'],
                    'articles': item['articles']
                }
                claims_file.write(json.dumps(claims_record) + '\n')

        with open(sentence_rows_path, 'w', newline='') as csv_file:
            fieldnames = [
                'id',
                'claim',
                'label',
                'label_id',
                'verifiable',
                'doc_id',
                'sentence_index',
                'sentence_text',
                'is_evidence'
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for item in tqdm(self.data, desc="Saving ML sentence rows CSV"):
                evidence_pairs = {
                    (ev['doc_id'], ev['sentence_id'])
                    for ev in item['evidence_flat']
                }

                for doc_id, sentences in item['articles'].items():
                    for sentence_index, sentence_text in enumerate(sentences):
                        writer.writerow({
                            'id': item['id'],
                            'claim': item['claim'],
                            'label': item['label'],
                            'label_id': item['label_id'],
                            'verifiable': item['verifiable'],
                            'doc_id': doc_id,
                            'sentence_index': sentence_index,
                            'sentence_text': sentence_text,
                            'is_evidence': int((doc_id, sentence_index) in evidence_pairs)
                        })

        return claims_path, sentence_rows_path

    def get_data(self):
        return self.data
    
    def print_sample(self, n=5):
        for item in self.data[:n]:
            print(json.dumps(item, indent=2))

    def save(self, filepath):
        with open(filepath, 'w') as f:
            for item in tqdm(self.data, desc="Saving data"):
                f.write(json.dumps(item) + '\n')



if __name__ == "__main__":
    test_data = FetchDataset('data/fever/test.jsonl', force_reprocess=True)
    claims_path, sentence_rows_path = test_data.export_ml_files()
    print(f"Saved ML claims file: {claims_path}")
    print(f"Saved ML sentence rows file: {sentence_rows_path}")
