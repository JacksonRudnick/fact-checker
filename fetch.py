#imports
from calendar import c
from tqdm import tqdm
import json
import wikipediaapi
import os
import re

class FetchDataset:
    def __init__(self, filepath: str):
        self.data = []

        if os.path.exists('data/fever/processed_' + filepath.strip('data/fever/')):
            with open('data/fever/processed_' + filepath.strip('data/fever/'), 'r') as f:
                for line in tqdm(f, desc="Reading processed file"):
                    self.data.append(json.loads(line))
        else:
            self.wiki_wiki = wikipediaapi.Wikipedia('fact-checker (jrudnick@cub.uca.edu)', 'en')

            with open(filepath, 'r') as f:
                for line in tqdm(f, desc="Reading input file"):
                    self.data.append(json.loads(line))
            self.process()
            self.fetch_articles()
            self.save('data/fever/processed_' + filepath.strip('data/fever/'))

    def process(self):
        #process dataset
        #grab label, claim, and evidence
        #convert label to 0,1,2
        #lowercase claim
        #format doc_id to match wikipedia format aka
        #2_Hearts_-LRB-Kylie_Minogue_song-RRB- -> 2_Hearts_(Kylie_Minogue_song)
        #convert evidence into tuple of (doc_id, sentence_id)
        for item in tqdm(self.data, desc="Processing data"):
            label = item['label']
            if label == 'SUPPORTS':
                item['label'] = 0
            elif label == 'REFUTES':
                item['label'] = 1
            else:
                item['label'] = 2

            item['claim'] = item['claim'].lower()

            # Process evidence: drop first two elements and format doc_id
            processed_evidence = []
            for ev_set in item['evidence']:
                processed_set = []
                for ev in ev_set:
                    doc_id = ev[2]
                    sentence_id = ev[3]
                    if doc_id is None:
                        processed_set.append([None, None])
                        continue
                    doc_id = doc_id.replace('-LRB-', '(').replace('-RRB-', ')')
                    processed_set.append([doc_id, sentence_id])
                processed_evidence.append(processed_set)
            item['evidence'] = processed_evidence

        #drop other keys
        self.data = [{ 'claim': item['claim'],
                       'label': item['label'],
                       'evidence': item['evidence'][0]} for item in self.data]

    def fetch_articles(self):
        #create a new entry in data for each article that contains array of article's sentences
        # for each item in data, for each evidence, fetch article text and split into sentences, store in new key 'articles' as dict of doc_id to sentences
        
        for claim in tqdm(self.data, desc="Fetching articles"):
            articles = {}
            for ev in claim['evidence']:
                doc_id = ev[0]
                if doc_id is None:
                    articles[doc_id] = ['No article found']
                    continue
                elif doc_id in articles:
                    continue
                try:
                    page = self.wiki_wiki.page(doc_id)
                    if not page.exists():
                        articles[doc_id] = ['No article found']
                        continue
                    sentences = re.split(r'\n+|(?<=[.!?])\s+', page.text)
                    articles[doc_id] = sentences
                except Exception as e:
                    articles[doc_id] = ['No article found']

            claim['articles'] = articles

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

    #load datasets
    train_data = FetchDataset('data/fever/train.jsonl')
    #test_data = FetchDataset('data/fever/test.jsonl')
    #test_data.print_sample()

    #isolate important sentence based on sentence id

    #run bert on each sentence in article
    #fine tune as needed

    #graph all data using bert output
    #all nodes should be connected

    #run graph attention network on graph
    #fine tune as needed

    #run each node through a linear layer to get single value

    #softmax all node weights

    #multiply each node vector by node weight

    #feed into linear layer to get 3 class output

    #softmax to get final output probabilities
    #0=supports, 1=refutes, 2=not enough info
