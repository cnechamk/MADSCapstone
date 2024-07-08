import json
import pandas as pd
from Scripts.urls import MinnURLs
from bs4 import BeautifulSoup
import requests
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
    def helper(row):
        endpoint = row[url_col]
        url = MinnURLs.urljoin(endpoint)
        data = _parse_page(url)
        return json.dumps(data)

    if urls_p is None:
        df = beige_book_urls.main()
    else:
        df = pd.read_csv(urls_p)

    df['text'] = df.apply(helper, axis=1)
    df.to_csv(save_p, index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Scrapes beige books text and adds to dataframe")
    parser.add_argument("save_p", type=str, help="Where to save the csv with scraped data")
    parser.add_argument("urls_p", type=str, help=)
    p = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/beige_book_urls.csv"
    save_p = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/beige_books.csv"

    main(save_p, p)
