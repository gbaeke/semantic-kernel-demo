import requests
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from typing import Annotated

class Scrape:
    @kernel_function(
        description="Scrapes a website and returns its content.",
        name="scrape",
    )
    def scrape(self, url: Annotated[str, "the url to scrape, add https:// when missing"]) -> str:
        """
        Scrapes the specified URL and returns its content.

        Args:
            url (str): The URL to scrape. Make sure to include 'https://' when missing.

        Returns:
            str: The content of the scraped website.
        """
        response = requests.get("https://r.jina.ai/" + url)
        
        # return first 200 characters of response
        return response.text[:200]