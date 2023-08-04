import errno
import json
import os
import sys
from PriorityQueue import PriorityQueue
from typing import List, Dict
from joblib import dump, load


"""
The following file is for saving and loading files which are used by the FocusedWebCrawler and other .py files which
might need to work with the indexes.
"""

# Methods for loading files
def save_index(file_name: str, index: Dict[int, tuple]):
    """
        Save the index dictionary to a file using joblib.
        :param file_name: File path to save the forward_index.
        :param index: Dictionary containing forward index data.
        """
    dump(index, file_name)


def save_visited_pages(file_name: str, visited_pages: set):
    """
    Save the visited_pages set to a JSON file.
    Exceptions need to be handled when calling the method and are not taken care of within.
    :param file_name: File path to save the visited_pages.
    :param visited_pages: Set containing visited pages.
    """
    with open(file_name, 'w') as file:
        json.dump(list(visited_pages), file)


def save_frontier_pages(file_name: str, frontier_pages: PriorityQueue):
    """
    Save the frontier_pages PriorityQueue to a file using joblib.
    Exceptions need to be handled when calling the method and are not taken care of within.
    :param file_name: File path to save the frontier_pages.
    :param frontier_pages: PriorityQueue containing frontier pages.
    """
    dump(frontier_pages.to_list(), file_name)


def save_similarity_hash(simhash_path: str, similarity_hash_dict: Dict[str, str]):
    """
    Save a dictionary containing URL keys and their corresponding similarity hash values.
    Exceptions need to be handled when calling the method and are not taken care of within.
    :param similarity_hash_dict: Dictionary containing URL keys and similarity hash values.
    :param simhash_path: File path to save the similarity hash dictionary.
    """
    dump(similarity_hash_dict, simhash_path)


# Methods for loading files
def load_json(file_name: str):
    """
    Load data from a JSON file with comprehensive error handling.
    :param file_name: File path to load data from.
    :return: The loaded data from the JSON file.
    """
    # Check if it is a .json file
    if not file_name.endswith('.json'):
        raise ValueError("Invalid file format. Only JSON files are supported.")

    # Raise exception when the file_name does not exist
    if not os.path.isfile(file_name):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), file_name)
    # If it exists, then open it.
    try:
        with open(file_name, 'r') as file:
            visited_pages = json.load(file)
        return visited_pages
    # When the file is corrupt, it throws an exception
    except json.JSONDecodeError as e:
        print(f"Error loading {file_name}: \n{e}")
        raise json.JSONDecodeError(f"An error occurred in decoding {file_name}", file_name, 0)


def load_file(file_name: str, error_message: str):
    """
    Load data from a JSON file with proper error handling.
    :param file_name: File path to load data from.
    :param error_message: Custom error message to display if loading fails.
    :return: The loaded data from the JSON file.
    """
    try:
        file = load_json(file_name)
    except FileNotFoundError as file_not_found_err:
        print(file_not_found_err)
        sys.exit(f"{file_name} does not exist. {error_message}")
    except json.JSONDecodeError as dec_err:
        print(dec_err)
        sys.exit("An error occurred in decoding the JSON file.")

    return file


def load_visited_pages():
    """
    Method that loads the visited pages for a crawling process that has already been started.
    When the file does not exist, an error is returned and the system exits.
    :return: Set with the visited pages of the previous crawl process.
    """
    file_name = "visited_pages.json"
    error_message = "Try giving the Web_Crawler object a frontier to create an empty frontier or construct it newly."
    return set(load_file(os.path.join("data_files", file_name), error_message))


def load_frontier():
    """
    Loads the frontier fron the frontier_pages.joblib file in data_files
    If the file is not found or any error occurs, a warning will be printed.
    :return: PriorityQueue containing frontier pages.
    """
    file_name = "frontier_pages.joblib"
    frontier_list = load(os.path.join("data_files", file_name))
    frontier_pq = PriorityQueue()
    for (priority, url) in frontier_list:
        frontier_pq.put((priority, url))
    return frontier_pq


def load_index(path: str):
    """
    Loads the index dictionary from the given path.
    :param path: File path to load the index dictionary.
    :return: The loaded index dictionary.
    """
    index = load(path)
    return index


def load_similarity_hash(simhash_path):
    """
    Load a dictionary containing URL keys and their corresponding similarity hash values.
    If it cannot be loaded, an empty dictionary is returned.
    :param simhash_path: File path to the saved similarity hash dictionary.
    :return: Dictionary containing URL keys and similarity hash values.
    """
    try:
        similarity_hash_dict = load(simhash_path)
        return similarity_hash_dict
    except FileNotFoundError:
        print(f"Error: File not found at {simhash_path}")
        return {}
    except Exception as e:
        print(f"Error: Unable to load data from {simhash_path}. Reason: {e}")
        return {}
