import time

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm.contrib.concurrent import process_map  # or thread_map

from Scripts.Scrape.utils import *
from Scripts.Scrape.urls import FedURLs

month_d = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "Jul": "July",
    "Oct": "October",
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}


def _postprocess_dates(df):
    df.month = df.month.str.split("/", expand=True)[0].replace(month_d).replace(month_d)
    df.day = df.day.str.split("-", expand=True)[0].str.split(' ', expand=True)[0]
    df = pd.to_datetime(df)
    return df


def recent_fomc():
    url = FedURLs.fomc_recent_dates
    res = requests.get(url)

    soup = BeautifulSoup(res.content, 'lxml')
    rows = {
        'year': [],
        'month': [],
        'day': []
    }

    for panel in soup.find_all('div', attrs={'class': "panel panel-default"}):
        year_div = panel.find_next('div', attrs={'class': "panel-heading"})
        year = year_div.text.split(" ")[0]
        for row in year_div.find_next_siblings('div'):
            txt = row.text
            if txt[0] == "*":
                continue

            _, month, day = row.text.split("\n")[:3]
            month = month.split("/")[0]
            day = day.split("-")[0]

            rows['year'].append(year)
            rows['month'].append(month)
            rows['day'].append(day)

    df = pd.DataFrame(rows)
    return df


def _parse_year(year):
    url = FedURLs.get_historical_materials_by_year(year)
    rows = {'year': [], 'header': []}
    headers = {'User-Agent': get_random_user_agent()}
    time.sleep(random.choice([0.3, 0.5, 1]))
    res = requests.get(url, headers=headers)

    soup = BeautifulSoup(res.content, 'lxml')

    for header in soup.find_all('h5'):
        rows['year'].append(year)
        rows['header'].append(header.text)

    return pd.DataFrame(rows)


def _get_historic_dates():
    dfs = process_map(_parse_year, range(1930, 2019))
    df = pd.concat(dfs)
    df = df.loc[df.header.str.contains('Meeting')]
    tmp = df.header.str.split(" _ ", expand=True)[0].str.split(" ", expand=True)[[0, 1]]
    df = pd.concat((df['year'], tmp), axis=1)
    df.columns = ["year", "month", "day"]
    df = _postprocess_dates(df)
    return df


def _get_recent_dates():
    df = recent_fomc()
    df = _postprocess_dates(df)
    return df


def main(save_p):
    historic_dates = _get_historic_dates()
    recent_dates = _get_recent_dates()

    df = pd.concat((historic_dates, recent_dates))

    df.to_csv(save_p, index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="scrapes dates of FOMC meetings")
    parser.add_argument('save_p', type=str, help='Where to save FOMC dates')

    args = parser.parse_args()
    main(args.save_p)
