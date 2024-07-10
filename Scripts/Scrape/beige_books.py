import time
import json
import random
import pickle

import requests
import pandas as pd
from boilerpy3 import extractors
from tqdm.contrib.concurrent import process_map  # or thread_map

from Scripts.urls import MinnURLs
from Scripts.Scrape import beige_book_urls

EXTRACTOR = extractors.ArticleExtractor()

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.10240",
    "Mozilla/5.0 (Windows NT 6.3; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0",
    "Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36"
]

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

def get_random_user_agent():
    return random.choice(USER_AGENTS)

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

def main(save_p: str, urls_p: str = None, url_col='url'):
    if urls_p is None:
        df = beige_book_urls.main()
    else:
        df = pd.read_csv(urls_p)

    # df['date'] = pd.to_datetime(df['date'], format="%B %Y")

    df_rows = [
        [
            getattr(row, 'district'),
            getattr(row, 'date'),
            getattr(row, 'url')
        ] for row in df.itertuples()]

    rows = process_map(_main_helper, df_rows)
    time.sleep(2)
    with open("/Users/joshfisher/PycharmProjects/MADSCapstone/Data/data.pkl", 'wb') as f:
        pickle.dumps(rows)

    df = pd.DataFrame(rows, columns=['district', 'date', 'url', 'text'])
    df.to_csv(save_p, index=False)

if __name__ == "__main__":
    p = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/beige_book_urls.csv"
    save_p = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/beige_books.csv"

    main(save_p, p)
