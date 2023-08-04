import os
from typing import Tuple, List

from bs4 import BeautifulSoup
from django.shortcuts import render
from django.http import HttpResponse
from gtts.tts import gTTS
from FocusedWebCrawler import send_get_request
from Ranker import Ranker
import nltk
nltk.download('omw-1.4')

from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

# Static variables
results_per_page = 11
abstract_length = 300

# Paths to data files for creating a ranker object
data_files_path = 'data_files'
results_path = os.path.join(data_files_path, 'results')
index_path = os.path.join(data_files_path, 'forward_index.joblib')
inverted_index_path = os.path.join(data_files_path, 'inverted_index.joblib')
embedding_index_path = os.path.join(data_files_path, 'embedding_index.joblib')
ranker = Ranker(index_path=index_path,
                inverted_index_path=inverted_index_path,
                embedding_index_path=embedding_index_path,
                results_path=results_path)


def get_title_and_text(website_url: str) -> Tuple[str, str]:
    raw_website_html_content = send_get_request(website_url)
    if raw_website_html_content != "" or raw_website_html_content != b"" or raw_website_html_content is not None:
        # Decode the retrieved html web page
        website_html_content = raw_website_html_content.decode('utf-8', 'ignore')
        # Create a BeautifulSoup object to parse the HTML content
        soup = BeautifulSoup(website_html_content, 'html.parser')
        website_title = soup.title.string.strip() if soup.title else "No Title Found"
        elements_to_extract = ["header", "head", "footer", "foot", "a", "nav", "href"]
        for element in elements_to_extract:
            for el in soup.find_all(element):
                el.extract()
        website_text = soup.get_text(separator=". ", strip=True)
        return website_title, website_text
    return "Website content could not be loaded", "Website content could not be loaded"


def generate_audio_files(query: str, start_index: int, websites: List[Tuple[str, str, str]]) -> List[str]:
    audio_files = []
    audio_dir = "media"
    for f in os.listdir(audio_dir):
        os.remove(os.path.join(audio_dir, f))
    for i, result in enumerate(websites):
        link, title, website_abstract = result
        tts = gTTS(text=f"Reading result {i + 1}. {title}. {website_abstract}. Please open the website for further "
                        f"information.", lang="en")
        audio_file_name = f"{query}_audio_file_{start_index + i}.mp3"
        audio_path = os.path.join("media", audio_file_name)
        tts.save(audio_path)
        audio_files.append(audio_file_name)
    return audio_files


def get_abstract(website_text):
    # Object of automatic summarization.
    auto_abstractor = AutoAbstractor()
    # Set tokenizer.
    auto_abstractor.tokenizable_doc = SimpleTokenizer()
    # Set delimiter for making a list of sentence.
    auto_abstractor.delimiter_list = [".", "\n"]
    # Object of abstracting and filtering document.
    abstractable_doc = TopNRankAbstractor()
    # Summarize document.
    result_dict = auto_abstractor.summarize(website_text, abstractable_doc)

    result_string = ''.join(map(str, result_dict['summarize_result']))

    return result_string


def generate_titles_and_abstracts(websites : List[str]) -> List[Tuple[str, str, str]]:
    results = []
    for ranked_website in websites:
        website_title, website_text = get_title_and_text(ranked_website)
        abstract_text = get_abstract(website_text)[:abstract_length] + '...'
        results.append((ranked_website, website_title, abstract_text))
    return results


# Create your views here.
def open_mainview(request):
    context = {}
    # Delete potential previous session
    request.session.pop('search_results', None)
    request.session.pop('query', None)
    return render(request, 'mainview.html', context)


def search(request):
    query = str(request.GET.get('queryField'))
    print(f"Query is: {query}")
    ranking_method = request.GET.get('ranker_select')
    print(f"Selected ranking_method is: {ranking_method}")

    # Check if the current query matches the stored query in the session
    stored_query = request.session.get('query', '')
    stored_ranking_method = request.session.get('ranking_method', '')
    print(f"Stored Query is: {stored_query}")
    if query != stored_query or ranking_method != stored_ranking_method:
        # The query has changed, so remove the stored search results from the session
        request.session.pop('search_results', None)
        request.session.pop('ranking_method', None)

    # Check if results are already stored in the session
    if 'search_results' not in request.session or 'ranker' not in request.session:
        print("Generating results")
        ranker.rank_method = ranking_method
        ranker_result = ranker.rank(query)
        # Store search results and query in the session
        request.session['search_results'] = ranker_result
        request.session['query'] = query
        request.session['ranking_method'] = ranking_method
    else:
        print("using old result")
        ranker_result = request.session['search_results']

    # Perform search operation based on the query
    start_index = int(request.GET.get('start_index', 0))
    limited_ranking_results_links = ranker_result[start_index: start_index + results_per_page]
    limited_results = generate_titles_and_abstracts(limited_ranking_results_links)
    audio_files = generate_audio_files(query, start_index, limited_results)
    # Boolean to whether next or previous elements can be shown
    show_more = start_index + results_per_page < len(ranker_result)
    show_previous = start_index >= results_per_page
    # Number of remaining elements
    remaining_elements = len(ranker_result) - (start_index + results_per_page)
    if remaining_elements > results_per_page:
        remaining_elements = results_per_page

    context = {
        'query': query,
        'search_results': limited_results,
        'next_start_index': start_index + results_per_page,
        'previous_start_index': start_index - results_per_page,
        'show_more': show_more,
        'show_previous': show_previous,
        'remaining_elements': remaining_elements,
        'audio_files': audio_files,
        'ranking_method': ranking_method
    }
    print(context['previous_start_index'])
    print(context['next_start_index'])
    # print(f"Full results are: ")
    # print(results)
    # print(context)
    return render(request, 'searchview.html', context)
