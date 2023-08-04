import os, re, tempfile, time, timeit
from PriorityQueue import PriorityQueue
from typing import List
import requests, json
from simhash import Simhash
import numpy as np
from fake_useragent import UserAgent
# Method for sending and receiving websites and sending http requests (urllib) and parsing them (BeautifulSoup)
import urllib3
from urllib.error import HTTPError
from urllib3.exceptions import NewConnectionError
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
# For checking whether a page is English or German
from py3langid.langid import LanguageIdentifier, MODEL_FILE
# Own imports
from File_loader import load_frontier, load_visited_pages, load_index, save_frontier_pages, save_visited_pages, \
    save_index, load_similarity_hash, save_similarity_hash
from Embedder import Embedder
from utils import preprocessing

"""
This file describes the Web crawler. It is focused towards english documents which are related to Tübingen. 
"""

# User-Agent list for pretending to be a real user when pages detect the crawler
ua = UserAgent()
user_agent_list = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0'
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36 Edg/87.0.664.75',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.18363',
    'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_4_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/7046A194A'
    'Mozilla/5.0 (Linux; U; Android 4.0.3; ko-kr; LG-L160L Build/IML74K) AppleWebkit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30',
    'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
    'Opera/9.80 (X11; Linux i686; Ubuntu/14.10) Presto/2.12.388 Version/12.16.2',
    ua.random,
    ua.googlechrome,
    ua.edge
]

class FocusedWebCrawler:
    def __init__(self, max_pages: int = np.inf, frontier: List[str] = None):
        """
        Initializes the Crawler object with a frontier and a maximum number of pages to be crawled.
        If the frontier is not given (None or no argument given), then the last search will be continued.
        :param max_pages: Number indicating the maximum number of webpages to be crawled
        :param frontier: np.ndarray of urls (Strings) or None if the past search should be continued!
        """
        # If no frontier is given --> Load the frontier, visited pages and index from a previous search
        self.embedder = Embedder('bert-base-uncased')
        if frontier is None:
            self.frontier = load_frontier()
            self.visited = load_visited_pages()
            index_path = os.path.join("data_files", 'forward_index.joblib')
            inverted_index_path = os.path.join("data_files", "inverted_index.joblib")
            embedding_index_path = os.path.join("data_files", "embedding_index.joblib")
            simhash_path = os.path.join("data_files", "simhash.joblib")
            self.inverted_index_db = load_index(inverted_index_path)
            self.index_db = load_index(index_path)
            self.index_embeddings_db = load_index(embedding_index_path)
            self.hashvalues = load_similarity_hash(simhash_path)
        else:
            self.frontier = PriorityQueue()
            for doc in frontier:
                self.frontier.put((1, doc))
            self.visited = set()
            self.index_db = {}
            self.inverted_index_db = {}
            self.index_embeddings_db = {}
            self.index_embeddings_pre_db = {}
            # store hashvalues of already indexed pages for duplicate detection
            self.hashvalues = {}
        # Maximum pages to be indexed
        self.max_pages = max_pages
        # Language identifier for checking the language of a document
        self.identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
        # self.identifier.set_languages(['de', 'en', 'fr'])

    def crawl(self, frontier: PriorityQueue, index_db):
        """
        Crawls the web with the given frontier and saves the results to "data_files" folder
        :param frontier: The frontier of known URLs to crawl. You will initially populate this with
        your seed set of URLs and later maintain all discovered (but not yet crawled) URLs here.
        :param index_db: The location of the local index storing the discovered documents.
        """
        if index_db == {}:
            num_pages_crawled = 0
        else:
            num_pages_crawled = max(index_db.keys()) + 1

        # initialize priority queue and add seed urls
        sss = time.time()
        while not frontier.empty() and num_pages_crawled <= self.max_pages:
            _, url = frontier.get()

            # If page has already been visited --> Continue loop
            if url in self.visited or is_forbidden_file(url):
                continue

            # skip urls that are disallowed in the robots.txt file
            robots_content = get_robots_content(url)
            if not is_allowed(url, robots_content):
                self.visited.add(url)
                continue

            time.sleep(0.15)

            print(f"Crawling page: {num_pages_crawled} with url: {url}", flush=True)

            # get page content and links on the page
            start = timeit.default_timer()
            page_links, page_header, page_content, page_footer = get_web_content_and_urls(url)
            print(f" getting content and urls took: {timeit.default_timer() - start:.2f}")
            # print(f" Page content: {page_content}")
            # print(f" Page links: {page_links}")

            # skip empty pages
            if page_links is None and page_content is None:
                self.visited.add(url)
                continue

            # Check if "Tübingen" or "Tuebingen" is contained somewhere in the URL or document
            contains_tuebingen = has_tuebingen(url) or has_tuebingen_content(url,
                                                                             " ".join([page_header, page_content,
                                                                                       page_footer]))
            start = timeit.default_timer()
            page_language = self.detect_language(page_content)
            print(f" Detecting language took: {timeit.default_timer() - start:.2f}s")

            page_priority = get_priority(contains_tuebingen, page_language)
            print(f" Detected priority was {page_priority}")
            page_links = set(page_links)

            # Add the URL to the Visited links,
            self.visited.add(url)

            if page_priority is None or page_priority >= 2:
                print(" THIS PAGE IS NOT RELEVANT!!! Continuing search")
                print("________________________________________________________")
                continue
            # Add newly discovered URLs to the frontier, assign priority 1 to topic relevant docs
            for link in page_links:
                if not (link in self.visited):
                    if is_valid_url(link):
                        frontier.put((page_priority, link))
                    else:
                        print(f" An invalid URL has been found and could not be added to the frontier: {link}")
                else:
                    print(f" The URL has already been visited. Skipping:{link}")
            # Add the URL and page content to the index

            # duplicate detection
            if is_duplicate(page_content, self.hashvalues):
                continue

            # Add the URL and page content to the index
            if page_priority == 1:  # save only english pages with tübingen content
                self.index_embeddings(page_content, num_pages_crawled, pre=False)
                preprocessed_page_content = preprocessing(page_content)
                # self.index_embeddings(preprocessed_page_content, num_pages_crawled)
                self.inverted_index(preprocessed_page_content, num_pages_crawled)
                self.index(url, num_pages_crawled, page_content)

            self.hashvalues[url] = compute_similarity_hash(page_content)

            # Save everything to files after every 25 documents and at the end of crawling
            if num_pages_crawled % 25 == 0 or num_pages_crawled == self.max_pages:
                try:
                    # Use temporary files for saving
                    temp_index_path = "temp_forward_index.joblib"
                    temp_inverted_index_path = "temp_inverted_index.joblib"
                    temp_embedding_index_path = self.embedder.model_name + " temp_embedding_index_3.joblib"
                    temp_embedding_index_path_pre = self.embedder.model_name + " temp_embedding_index_pre_3.joblib"
                    temp_simhash_path = "temp_simhash.joblib"
                    temp_visited_path = "temp_visited_pages.json"
                    temp_frontier_path = "temp_frontier_pages.joblib"

                    # Save to temporary files
                    save_index(temp_index_path, self.index_db)
                    save_index(temp_inverted_index_path, self.inverted_index_db)
                    save_index(temp_embedding_index_path, self.index_embeddings_db)
                    save_index(temp_embedding_index_path_pre, self.index_embeddings_pre_db)
                    save_similarity_hash(temp_simhash_path, self.hashvalues)
                    save_visited_pages(temp_visited_path, self.visited)
                    save_frontier_pages(temp_frontier_path, frontier)

                    # If all saves are successful, move the temporary files to the actual save locations
                    file_folder = "data_files"
                    os.replace(temp_index_path, os.path.join(file_folder, "forward_index.joblib"))
                    os.replace(temp_embedding_index_path, os.path.join(file_folder, "embedding_index.joblib"))
                    os.replace(temp_inverted_index_path, os.path.join(file_folder, "inverted_index.joblib"))
                    os.replace(temp_visited_path, os.path.join(file_folder, "visited_pages.json"))
                    os.replace(temp_frontier_path, os.path.join(file_folder, "frontier_pages.joblib"))
                    os.replace(temp_simhash_path, os.path.join(file_folder, "simhash.joblib"))

                    print("Data saved successfully.")
                except Exception as e:
                    # Handle any exceptions that occur during saving
                    print(f"An error occurred while saving data: {str(e)}")
                    print("Data not saved.")

            # After page has been crawled, increment the number of visited pages by 1
            num_pages_crawled += 1
            print("____________________________")

        print(f"Index is: {self.index_db}")
        print(f"took time: {time.time() - sss}")

    def index(self, url: str, key: int, content: str) -> None:
        """
        Add a document to the index. You need (at least) two parameters:
        :param url: The URL with which the document was retrieved
        :param key: Key of the document to be indexed
        :param content: Content of the document to be indexed
        """
        length = len(content.split())
        self.index_db[key] = (url, length)

    def index_embeddings(self, doc: str, key: int, pre: bool = True) -> None:
        """
        Add a document embedding to the embedding index
        :param doc: The document to be indexed already preprocessed
        :param pre: True if text was preprocessed
        :param key: Key of the document to be indexed
        """
        if pre:
            self.index_embeddings_pre_db[key] = self.embedder.embed(doc)
        else:
            self.index_embeddings_db[key] = self.embedder.embed(doc)

    def inverted_index(self, doc: str, key: int) -> None:
        """
        Add a document to the inverted index. You need (at least) two parameters:
        :param doc: The document to be indexed already preprocessed
        :param key Key of the docuemnt to be indexed
        """
        terms = doc.split()
        for position, term in enumerate(terms):
            if term not in self.inverted_index_db:
                self.inverted_index_db[term] = [[key, [position]]]
            else:
                found = False
                for entry in self.inverted_index_db[term]:
                    if entry[0] == key:
                        entry[1].append(position)
                        found = True
                        break
                if not found:
                    self.inverted_index_db[term].append([key, [position]])

    def detect_language(self, text: str) -> str or None:
        """
        Method that detects the language that was used in a document to prevent German and documents of other
        languages to get into our index
        :param text: The text that is to be classified into a language
        :return: Shortcut for the text language, e.g. 'de', 'en', ...
        """
        try:
            detected_languages = {}
            for sentence in text.split('.'):  # Split text into sentences
                lang, confidence = self.identifier.classify(sentence)
                if confidence >= 0.5:  # Set a confidence threshold
                    detected_languages[lang] = detected_languages.get(lang, 0) + 1

            lang_with_most_sentences = max(detected_languages, key=detected_languages.get)
            print(f" The detected language was: {lang_with_most_sentences} from the occurences {detected_languages}")
            return lang_with_most_sentences

        except Exception as e:
            print(f"Some error occured during language detection of the string: {str(e)}")
            return None


# checks if given url is valid (considered valid if host and port components are present)
def is_valid_url(url) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_base_url(url: str) -> str:
    """
    Method that strips the given URL and returns only the base part of the URL.
    Example: https://www.tuebingen.de/blumenschmuck -> https://www.tuebingen.de
    :param url:
    :return:
    """
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url


def send_get_request(url: str, max_retries: int = 1, retry_delay: float = 2) -> bytes:
    """
    Method that sends an http get request and returns the http page in bytes
    :param url: URL to send the GET request to
    :param max_retries: Optional, Number of maximum retries if the get request fails
    :param retry_delay: Optional, The delay between requests if a get request fails
    :return: HTML bytes of the internet page
    """
    raw_html_content = b""
    for idx, user_agent in enumerate(user_agent_list):
        if raw_html_content == b"":
            print(f"Trying user agent identity {idx}")
            retry = urllib3.Retry(total=3, redirect=3)
            timeout = urllib3.Timeout(total=5.0, connect=3.0, read=3.0)
            headers = {
                'User-Agent': user_agent,
                'Accept-Language': '*'
            }
            http = urllib3.PoolManager(retries=retry, timeout=timeout, headers=headers)
            for retry_count in range(max_retries):
                try:
                    with http.request('GET', url, headers=headers, preload_content=False) as response:
                        raw_html_content = b""
                        for chunk in response.stream(4096):
                            raw_html_content += chunk
                        if response.status == 200:
                            break
                        else:
                            raw_html_content = b""
                            raise Exception(
                                f"Exception in GET request. The response status was not 200 OK "
                                f"but was {response.status}.")
                except Exception as e:
                    error_str = f"Attempt {retry_count + 1} failed. "
                    if retry_count > 1:
                        error_str += "Retrying after {retry_delay} seconds.\n "
                    error_str += f"Exception: {e}"
                    print(error_str)
                    time.sleep(retry_delay)
    return raw_html_content


def get_web_content_and_urls(url: str) \
        -> (List[str], str, str, str) or (None, None, None, None):
    """
    Method that gets the html content of  the given URL and gives the contained header, content, footer and URLs back
    :param url: URL of the website that should be retrieved
    :return (links:List[str], header_content:str, body_content:str, footer_content:str)
    """
    # handling failed requests
    raw_html_content = send_get_request(url)

    if raw_html_content != "" or raw_html_content != b"" or raw_html_content is not None:
        # Decode the retrieved html web page
        html_content = raw_html_content.decode('utf-8', 'ignore')
        # Create a BeautifulSoup object to parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove style and script tags as they contain no information
        for data in soup(['style', 'script']):
            # Remove tags
            data.decompose()

        # Extract header
        header = soup.find('header')
        header_content = header.extract().get_text(separator=" ") if header else ""

        # Extract footer
        footer = soup.find('footer')
        footer_content = footer.extract().get_text(separator=" ") if footer else ""

        # Get body content. After having extracted the header and footer, only the title and the body of the document
        # shall remain
        body_content = soup.get_text(separator=" ")
        body_content = re.sub(r'\s+', ' ', body_content)

        # Extract all the <a> html-tags for links IF they don't start with # because those are usually internal links
        # within a webpage (anchor links) and also don't include JavaScript links because they often execute a
        # JavaScript script or are not relevant here
        links = [a['href'] for a in soup.find_all('a', href=True)
                 if not a['href'].startswith(('#', 'javascript:'))]
        # Some links are given in an absolute (http...) form and some are given in a relative form (/example...).
        # The latter need to be transformed. The rest stays the same
        links = get_absolute_links(url, links)
        return links, header_content, body_content, footer_content
    return None, None, None, None


def get_absolute_links(url: str, links: List[str]) -> List[str]:
    """
    Method that returns absolute links for a list of absolute and/or relative links
    :param url: The website url that is origin of all received links
    :param links: List of links that were retrieved from the url
    :return: A list of Strings (URLs) which contains only absolute links which are directly callable
    """
    base_url = get_base_url(url)
    absolute_links = set()
    for link in links:
        # If link is relative then join it with the base page url
        absolute_link = link if link.startswith(('http://', 'https://')) else urljoin(base_url, link)
        # Only add the page if it is not the page that was used to retrieve all the links to prevent unnecessary
        # requests
        if absolute_link != url and absolute_link != base_url:
            absolute_links.add(absolute_link)
    return list(absolute_links)


def get_robots_content(url: str) -> str:
    """
    Method that returns content of the robots.txt file for a given URL
    :param url: The website URL
    :return: A string containing the content of the robots.txt file
    """
    root_url = get_base_url(url)
    robots_url = root_url + "/robots.txt"

    try:
        robot_content_bytes = send_get_request(robots_url)
        try:
            content = robot_content_bytes.decode('utf-8', 'ignore')
        except UnicodeDecodeError:
            content = robot_content_bytes.decode('latin-1', 'ignore')
        return content
    except HTTPError as e:
        print(f"HTTP error occurred while retrieving robots.txt: {str(e)}")
    except NewConnectionError as e:
        print(f"URL error occurred while retrieving robots.txt: {str(e)}")
    except Exception as e:
        print(f"Another error occured while retrieving robots.txt: {str(e)}")

    return ""


def get_user_agent() -> None or str:
    """
    method that returns the current user agent
    """
    try:
        response = requests.get('https://httpbin.org/user-agent', timeout=5.0)
        response_json = response.json()
        user_agent = response_json.get('user-agent')
        return user_agent
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"Error retrieving user agent: {str(e)}")
        return None


def is_allowed(url: str, robots_content: str) -> bool:
    """
    Method that checks if crawling a given url is allowed in the current robots.txt file
    :param url: current url
    :param robots_content: content of the current robots.txt file
    :return: False if crawling the url is disallowed in robots, True otherwise
    """
    # get the path of the url without base url
    # e.g.: ('http://www.example.com/hithere/something/else' -> /hithere/something/else)
    path = urlparse(url).path
    # save rules relevant for the current user agent
    # user_agent_rules = []
    current_user_agent = None
    for line in robots_content.splitlines():
        if line.lower().startswith("user-agent"):
            current_user_agent = line.split(":")[1].strip()
        elif line.lower().startswith("disallow") and current_user_agent == "*":
            disallowed_path = line.split(":")[1].strip()
            if path.startswith(disallowed_path):
                print(f"disallowed url detected: {path}")
                return False
            # append relevant rules
            # user_agent_rules.append(disallowed_path)
    return True

    # check if the provided path is allowed
    # for rule in user_agent_rules:
    #     if path.startswith(rule):
    #         print(f"disallowed url detected: {path}")
    #         return False
    # return True


def compute_similarity_hash(page_content: str, k: int = 5) -> str:
    """
    Method tht returns a 64bit binary similarity hash value for a page content
    :param page_content: the content of the current page
    :param k: threshold for bit difference
    """
    # Compute the similarity hash for a string
    hash_value = Simhash(page_content).value
    similarity_hash = hash_value >> k
    binary_hash = format(similarity_hash, '064b')

    return binary_hash


def is_duplicate(content: str, previous_hashes, k: int = 5):
    """
    Method that checks a document against an existing collection of previously seen documents for near duplicates
    :param content: page content of the current page
    :param previous_hashes: contains the hash values of all pages that have been indexed before
    :param k: threshold of bit difference that is necessary to consider two documents duplicates
    :return: True if the current document is a duplicate of any previously indexed document, False otherwise
    """
    current_hash = compute_similarity_hash(content)

    for hash_ in previous_hashes:
        bit_difference = np.sum(np.abs(
            np.array([int(bit) for bit in current_hash]) - np.array([int(bit) for bit in previous_hashes[hash_]])))
        if bit_difference <= k:
            return True

    return False


def is_forbidden_file(url:str)->bool:
    """
    Checks if the url is a file. If yes, the URL should not be loaded as it is not a HTML document.
    :param url: String definining the url
    :return: True if it is forbidden, False if it is okay (HTML document that can be parsed with bs4)
    """
    forbidden_file_endings = [".jpg", ".png", ".jpeg", ".pdf", ".ppt", ".pptx"]
    for ending in forbidden_file_endings:
        if url.lower().endswith(ending):
            return True
    return False

def has_tuebingen(string_to_check: str) -> bool:
    """
    Check if a webpage is relevant based on the presence of the word "Tübingen" or "Tuebingen" within the content.
    The uppercase should be ignored here
    :param string_to_check: The string that is to be checked
    :return: True if the webpage is relevant (contains "Tübingen" or "Tuebingen"), False otherwise
    """
    tuebingen_umlaut_regexp = re.compile(r"Tübingen", re.IGNORECASE)
    tuebingen_regexp = re.compile(r"Tuebingen", re.IGNORECASE)
    tuebingen_reg = re.compile(r"Tubingen", re.IGNORECASE)

    if tuebingen_umlaut_regexp.search(string_to_check) or tuebingen_regexp.search(
            string_to_check) or tuebingen_reg.search(string_to_check):
        return True

    return False


def has_tuebingen_content(url: str, string_to_check: str) -> bool:
    """
    Check if a webpage is relevant based on the presence of the word "Tübingen" or "Tuebingen" or "Tubingen" within
    the content. The check is case-insensitive. An additional case has been introduced for wikipedia pages. They must
    contain a form of Tübingen 5 or more times to be relevant as irrelevant Wikipedia pages often contain Tübingen
    only once in their references.
    :param url: url of the page to check
    :param string_to_check: The string that is to be checked
    :return: True if the webpage is relevant (contains "Tübingen" or "Tuebingen"), False otherwise
    """
    pattern = r'(t(?:ü|ue|u)?binge[nr])'
    matches = re.findall(pattern, string_to_check, re.IGNORECASE)

    pattern_location = re.compile(r'7207[0246] T(?:ü|ue|u)?bingen', re.IGNORECASE)

    if re.search(r'wikipedia', get_base_url(url), re.IGNORECASE):  # Case for Wikipedia
        threshold = 5
    else:  # Case for all other pages
        threshold = 3

    print(f" threshold for checking occurences of Tübingen is: {threshold}")

    if len(matches) >= threshold or pattern_location.search(string_to_check):
        return True
    else:
        return False


def get_priority(contains_tuebingen: bool, language: str) -> int or None:
    """
    Returns the priority of a document given the information if it contains Tübingen and its langauge
    :param contains_tuebingen: bool, Parameter that indicates whether some form of
    the word "Tübingen" is contained in the document
    :param language: str, String that represents the abbreviation of the most used language in the document
    :return: Integer indicating the priority where 1 is the highest and 4 is the lowest priority or None if
    the document is not of relevance of any sort.
    """
    if contains_tuebingen and language == 'en':
        return 1
    elif contains_tuebingen and language == 'de':
        return 2
    elif language == 'en':
        return 3
    elif language == 'de':
        return 4
    else:
        return None


# -----------------------------
# just testing
if __name__ == '__main__':
    urls = ['https://en.wikipedia.org/wiki/T%C3%BCbingen',
            'https://www.dzne.de/en/about-us/sites/tuebingen',
            'https://www.britannica.com/place/Tubingen-Germany',
            'https://tuebingenresearchcampus.com/en/tuebingen/general-information/local-infos/',
            'https://wikitravel.org/en/T%C3%BCbingen',
            'https://www.tasteatlas.com/local-food-in-tubingen',
            'https://www.citypopulation.de/en/germany/badenwurttemberg/t%C3%BCbingen/08416041__t%C3%BCbingen/',
            'https://www.braugasthoefe.de/en/guesthouses/gasthausbrauerei-neckarmueller/']

    crawler = FocusedWebCrawler(frontier=urls, max_pages=10000)
    crawler.crawl(frontier=crawler.frontier, index_db=crawler.index_db)
