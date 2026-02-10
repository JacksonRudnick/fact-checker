#imports
import json
import wikipediaapi
import os

class FetchDataset:
    def __init__(self, filepath: str):
        self.data = []

        if os.path.exists('data/fever/processed_' + filepath.strip('data/fever/')):
            with open('data/fever/processed_' + filepath.strip('data/fever/'), 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
        else:
            self.wiki_wiki = wikipediaapi.Wikipedia('fact-checker (jrudnick@cub.uca.edu)', 'en')

            with open(filepath, 'r') as f:
                for line in f:
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
        for item in self.data:
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
                       'evidence': item['evidence']} for item in self.data]

    def fetch_articles(self):
        #create a new entry in data for each article that contains array of article's sentences
        for item in self.data:
            articles = {}
            for ev_set in item['evidence']:
                for ev in ev_set:
                    doc_id = ev[0]
                    if doc_id is None or doc_id in articles:
                        continue
                    try:
                        page = self.wiki_wiki.page(doc_id)
                        if not page.exists():
                            articles[doc_id] = []
                            continue
                        sentences = page.text.split('. ')
                        articles[doc_id] = sentences
                        #should implement saving to disk to avoid re-fetching
                    except Exception as e:
                        articles[doc_id] = []

            item['articles'] = articles

    def get_data(self):
        return self.data

    def save(self, filepath):
        with open(filepath, 'w') as f:
            for item in self.data:
                f.write(json.dumps(item) + '\n')



if __name__ == "__main__":

    #load datasets
    #train_data = Dataset('data/fever/train.jsonl')
    test_data = FetchDataset('data/fever/paper_test.jsonl')
    print(json.dumps(test_data.data[:5], indent=2))

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
