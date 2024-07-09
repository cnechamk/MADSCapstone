import time
import json
import random

import numpy as np
import requests
import pandas as pd
from bs4 import BeautifulSoup
from Scripts.urls import MinnURLs
from Scripts.Scrape import beige_book_urls
from tqdm.contrib.concurrent import process_map  # or thread_map


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
    soup = BeautifulSoup(res.content, 'lxml')
    ps = soup.find_all('p')
    d = {}
    take2 = False
    take3 = False
    for p in ps:
        txt = p.text.split('\n')
        if txt[0] in HEADERS:
            d[txt[0]] = txt[1]
        elif (p.strong and p.text) or take2:
            if not p.strong:
                header = "summary"
                take2 = False
            else:
                header = p.strong.text
            text = p.text
            text = text.replace(header, "")
            header = header.strip()
            text = text.strip()
            if text == "":
                take2 = True
                continue
            d[header] = text
        elif (p.b and p.text) or take3:
            if not p.b:
                header = "Summary"
                take3 = False
            else:
                header = p.b.text
            text = p.text
            text = text.replace(header, "")
            header = header.strip()
            text = text.strip()
            if text == "":
                take3 = True
                continue
            d[header] = text

    return d


def _main_helper(row):
    if not pd.isna(row[-1]):
        return row
    else:
        time.sleep(random.randint(0, 2))
        endpoint = row[2]
        url = MinnURLs.urljoin(endpoint)
        data = _parse_page(url)
        text = json.dumps(data)
        row[-1] = text
        return row

def main(save_p: str, urls_p: str = None, url_col='url'):
    if urls_p is None:
        df = beige_book_urls.main()
    else:
        df = pd.read_csv(urls_p)

    df.text = np.nan
    # df['date'] = pd.to_datetime(df['date'], format="%B %Y")

    df_rows = [
        [
            getattr(row, 'district'),
            getattr(row, 'date'),
            getattr(row, 'url'),
            getattr(row, 'text')
        ] for row in df.itertuples()]

    rows = process_map(_main_helper, df_rows)

    df = pd.DataFrame(rows, columns=['district', 'date', 'url', 'text'])
    df.to_csv(save_p, index=False)

if __name__ == "__main__":
    # from argparse import ArgumentParser
    #
    # parser = ArgumentParser(description="Scrapes beige books text and adds to dataframe")
    # parser.add_argument("save_p", type=str, help="Where to save the csv with scraped data")
    # parser.add_argument("urls_p", type=str, help="URLs ")
    p = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/beige_book_urls.csv"
    save_p = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/bb_new.csv"

    main(save_p, p)
