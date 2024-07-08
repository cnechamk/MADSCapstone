"""
Scrapes Beige Book URLs
"""

import os.path

import mechanicalsoup
import pandas as pd
from tqdm import tqdm
from Scripts.urls import MinnURLs

def postprocess_bb_df(df: pd.DataFrame):
    """
    Post-process beige book dataframe
    Args:
        df:

    Returns:

    """
    def helper(row):
        (district, _), (date, url) = row.values
        return pd.Series((district, date, url), index=("district", "date", "url"))
    df = df.apply(helper, axis=1, result_type='expand')
    return df


def main(save_p: str = None, start_year: int = None, end_year: int = None):
    """

    Args:
        save_p:
        start_from: starts from this year

    Returns:
        df
    """
    start_year = 0 if start_year is None else start_year
    end_year = 9999 if end_year is None else end_year

    if save_p is not None and os.path.exists(save_p):
        raise FileExistsError(save_p)

    base_url = MinnURLs.beige_book_archive
    browser = mechanicalsoup.StatefulBrowser(soup_config={'features': 'lxml'})
    browser.open(base_url)
    soup = browser.page
    years = soup.find(
        'form', attrs={'id': "bb_search"}
    ).find(
        'select', attrs={'name': "bb_year"}
    ).find_all(
        'option'
    )
    years = [year.contents[0] for year in years if end_year >= int(year.contents[0]) >= start_year]

    dfs = []
    for year in tqdm(years):
        browser.open(base_url)
        browser.select_form('form[id="bb_search"]')
        browser['bb_district'] = "Any"
        browser['bb_year'] = year
        res = browser.submit_selected()

        df = pd.read_html(
            res.content,
            flavor='lxml',
            extract_links='all'
        )[0]
        df = postprocess_bb_df(df)
        dfs.append(df)

    df = pd.concat(dfs)

    if save_p is not None:
        df.to_csv(save_p, index=False)

    return df


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="scrapes all beige book urls as a csv")
    parser.add_argument("save_p", type=str, help="where to save csv")
    parser.add_argument(
        "--start_year", type=int, help="Optional inclusive, year to start from, defaults to first possible year"
    )
    parser.add_argument(
        "--end_year", type=int, help="Optional inclusive, year to end at, default to last possible year"
    )

    args = parser.parse_args()

    main(args.save_p, start_year=args.start_year, end_year=args.end_year)
