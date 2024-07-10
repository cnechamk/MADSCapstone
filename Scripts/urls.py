import random
from urllib.parse import urljoin


class _BaseURLs:
    base_url = None
    @classmethod
    def urljoin(cls, path):
        if cls.base_url is None:
            raise NotImplementedError("_BaseURLs subclass must implement base_url class attribute")

        return urljoin(cls.base_url, path)

class MinnURLs(_BaseURLs):
    """ URLs for Minneapolis fed """
    base_url = "https://www.minneapolisfed.org"
    beige_book_archive = urljoin(  # page has links to archived beige books
        base_url,
        "/region-and-community/regional-economic-indicators/beige-book-archive"
    )


class FedURLs(_BaseURLs):
    base_url = "https://www.federalreserve.gov"
    historical_links_url = urljoin(  # page has links to various years' materials
        base_url,
        "monetarypolicy/fomc_historical_year.htm")
    beige_books_archive_url = urljoin(  # page has links to various years' beige books
        base_url,
        "/monetarypolicy/beige-book-archive.htm")

    @classmethod
    def get_historical_materials_by_year(cls, year: int = None):
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
        return urljoin(cls.base_url, endpoint)

    @classmethod
    def yield_historical_materials_by_year(cls, start=1936, end=2018):
        for year in range(start, end+1, 1):
            yield year, cls.get_historical_materials_by_year(year)


if __name__ == "__main__":
    print(MinnURLs.urljoin("/beige-book-reports/2023/2023-04-cl"))
    # hist_urls = fed_urls.yield_historical_materials_by_year()
    # for url in hist_urls:
    #     print(url)
