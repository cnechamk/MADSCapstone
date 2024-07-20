import time
import json

import requests
import pandas as pd
from boilerpy3 import extractors
from tqdm.contrib.concurrent import process_map  # or thread_map

from Scripts.Scrape.urls import MinnURLs
from Scripts.Scrape import beige_book_urls
from Scripts.Scrape.utils import *

EXTRACTOR = extractors.ArticleExtractor()


HEADERS = [
    "Summary of Economic Activity",
    "Labor Markets",
    "Prices",
    "Consumer Spending",
    "Manufacturing",
    "Real Estate and Construction",
    "Financial Services",
    "Nonfinancial Services",
    "Community Conditions"
]

def _parse_page(url):
    headers = {'User-Agent': get_random_user_agent()}
    res = requests.get(url, headers=headers)
    content = EXTRACTOR.get_content(res.text)
    return content

def _main_helper(row):
    time.sleep(random.randint(0, 2))
    endpoint = row[-1]
    url = MinnURLs.urljoin(endpoint)
    data = _parse_page(url)
    text = json.dumps(data)
    row.append(text)
    return row

def main(save_p: str, urls_p: str = None):
    if urls_p is None:
        df = beige_book_urls.main()
    else:
        df = pd.read_csv(urls_p)

    df_rows = [
        [
            getattr(row, 'district'),
            getattr(row, 'date'),
            getattr(row, 'url')
        ] for row in df.itertuples()]

    rows = process_map(_main_helper, df_rows)
    time.sleep(2)

    df = pd.DataFrame(rows, columns=['district', 'date', 'url', 'text'])
    df.to_csv(save_p, index=False)

if __name__ == "__main__":
    p = "../../Data/Raw/beige_book_urls.csv"
    save_p = "../../Data/beige_books.csv"
    main(save_p, p)
