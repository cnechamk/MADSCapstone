import random
import urllib.parse

base_url = "https://www.federalreserve.gov"

def get_url(endpoint: str, base_url=base_url) -> str:
    """
    Combines base_url with endpoint and returns full url
    Args:
        endpoint: url starting from end of base_url
        base_url: base url, default: "https://www.federalreserve.gov"

    Returns:
        full url: str
    """
    return urllib.parse.urljoin(base_url, endpoint)


historical_links_url = get_url("monetarypolicy/fomc_historical_year.htm")  # page has links to various years' materials
beige_books_archive_url = get_url("/monetarypolicy/beige-book-archive.htm")  # page has links to various years' beige b

def get_historical_materials_by_year(year: int = None):
    """

    Args:
        year:

    Returns:

    """
    start_year = 1936
    end_year = 2018

    if year is None:
        year = random.randint(start_year, end_year)

    assert 2018 >= year >= 1936, f"year must be between 1936-2018, got {year}"

    endpoint = f"/monetarypolicy/fomchistorical{year}.htm"
    return get_url(endpoint)

def yield_historical_materials_by_year(start=1936, end=2018):
    for year in range(start, end+1, 1):
        yield year, get_historical_materials_by_year(year)


if __name__ == "__main__":
    hist_urls = yield_historical_materials_by_year()
    for url in hist_urls:
        print(url)
