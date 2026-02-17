from fetch import FetchDataset
import json
import csv

if __name__ == "__main__":
    #load datasets
    test_data = FetchDataset('data/fever/paper_test.jsonl')

    #save to csv
    with open('data/fever/test.csv', 'w', newline='') as csvfile:
        fieldnames = ['claim', 'label', 'evidence', 'articles']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for item in test_data.get_data():
            writer.writerow({
                'claim': item['claim'],
                'label': item['label'],
                'evidence': json.dumps(item['evidence']),
                'articles': json.dumps(item['articles'])
            })