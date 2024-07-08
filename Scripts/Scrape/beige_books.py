import json

import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from Scripts.urls import MinnURLs
from Scripts.Scrape import beige_book_urls


def _parse_page(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.content, 'lxml')
    ps = soup.find_all('p')
    d = {}
    for p in ps:
        if p.strong and p.text:
            header = p.strong.text
            text = p.text
            text = text.replace(header, "")
            header = header.strip()
            text = text.strip()
            if text == "": continue

            d[header] = text

    return d


def main(save_p: str, urls_p: str = None, url_col='url'):
    if urls_p is None:
        df = beige_book_urls.main()
    else:
        df = pd.read_csv(urls_p)

    df['date'] = pd.to_datetime(df['date'], format="%B %Y")

    n = len(df)
    text_list = []
    for row in tqdm(df.itertuples(), total=n):
        endpoint = getattr(row, url_col)
        url = MinnURLs.urljoin(endpoint)
        data = _parse_page(url)
        text_list.append(json.dumps(data))

    df['text'] = text_list
    df.to_csv(save_p, index=False)

if __name__ == "__main__":
    # from argparse import ArgumentParser
    #
    # parser = ArgumentParser(description="Scrapes beige books text and adds to dataframe")
    # parser.add_argument("save_p", type=str, help="Where to save the csv with scraped data")
    # parser.add_argument("urls_p", type=str, help="URLs ")
    p = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/beige_book_urls.csv"
    save_p = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/beige_books.csv"

    main(save_p, p)
