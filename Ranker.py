from File_loader import load_index
from Embedder import Embedder
from utils import preprocessing
import numpy as np
import os
import re
from difflib import SequenceMatcher
from typing import Tuple
#Class for doing the ranking of the documents

class Ranker:
    """
    Class using different ranking methods to return the top k documents for a given query
    """
    def __init__(self, index_path: str, inverted_index_path:str, embedding_index_path: str, results_path: str, relevant_docs_count: int=100, ):
        self.index_db = load_index(index_path)
        self.inverted_index_db = load_index(inverted_index_path)
        self.doc_count = len(self.index_db)
        if self.doc_count < relevant_docs_count:
            self.relevant_docs_count = self.doc_count
        else:
            self.relevant_docs_count = relevant_docs_count
        self.load_embedding_index(embedding_index_path)
        if not self.check():
            raise ValueError("The indecies do not have the same length")
        self.embedder = Embedder('bert-base-uncased') #must be set to the same model as the one used for the embedding index
        self.rank_method = "BM25" 
        self.results_path = results_path
        self.b = 0.75
        self.k1 = 1.2
        self.calculate_avgdl() 
        self.rank_methods = ["BM25", "TF-IDF", "Feature_embedding", "Pseudo_relevance_feedback", "Merge"]


    def calculate_avgdl(self):
        """
        Calculate the average document length
        """
        avgdl = 0
        for key, value in self.index_db.items():
            avgdl += value[1]
        self.avgdl = avgdl/len(self.index_db)


    def check(self)-> bool:
        """
        Check if all the indecies are loaded correctly
        :return: True if the index is loaded, False otherwise
        """
        index_len = len(self.index_db)
        embedding_number = self.embeddings.shape[0]
        ids = len(self.id)
        print("Index length: ", index_len)
        print("Embedding number: ", embedding_number)
        print("ID length: ", ids)
        if index_len == embedding_number and index_len == ids:
            return True
        else:
            return False

    def rank(self, query: str) -> list:
        """
        Rank the documents for the given query using the set ranking method
        :param query: the query string
        :return: the list of the top k documents as a list of tuples (url, score)
        """

        if self.rank_method == "Feature_embedding":
            sorted_docs = self.embedding_ranking(query)
        else:
            relevant_docs = self.query_union(query)
            if self.rank_method == "Merge":
                sorted_docs_tf_idf = self.TF_IDF(query, relevant_docs)
                sorted_docs_BM25 = self.BM25(query, relevant_docs)
                merged_list = sorted_docs_tf_idf[:5]
                sorted_docs_tf_idf.extend(sorted_docs_BM25[:5])
                sorted_docs_pseudo_relevance = self.pseudo_relevance_feedback_embedding(merged_list, mode='distance')
                sorted_docs = self.merge_rankings(sorted_docs_tf_idf, sorted_docs_BM25, sorted_docs_pseudo_relevance)
            if self.rank_method == "Pseudo_relevance_feedback":
                sorted_docs_tf_idf = self.TF_IDF(query, relevant_docs)[:5]
                sorted_docs_BM25 = self.BM25(query, relevant_docs)[:5]
                sorted_docs_tf_idf.extend(sorted_docs_BM25)
                sorted_docs = self.pseudo_relevance_feedback_embedding(sorted_docs_tf_idf, mode='distance')
            elif self.rank_method == "BM25":
                sorted_docs = self.BM25(query, relevant_docs)
            elif self.rank_method == "TF-IDF":
                sorted_docs = self.TF_IDF(query, relevant_docs)
        self.save_results(sorted_docs, query)
        return [self.index_db[result[0]][0] for result in sorted_docs]
        #TODO return ordere list of links

    #Helper functions

    @staticmethod
    def sequence_similarity(url1, url2):
        matcher = SequenceMatcher(None, url1, url2)
        similarity_ratio = matcher.ratio()
        return similarity_ratio
    
    @staticmethod
    def get_url(url):
        regex_pattern = r"https?://([^/]+)/"
        url_domain = re.search(regex_pattern, url).group(1)
        return url_domain

    def gain(self,url, urls, websites):
        """
        Calculate the gain if an url is added to the resultlist
        :param url: the url to check
        :param urls: the list of websites already in the resultlist
        :param websites: the dictionary of websites and their urls
        """
        url_domain = self.get_url(url)
        if url_domain not in urls:
            gain = 10 #TODO finetuning
        else:
            #check other urls from website
            other_website_urls = websites[url_domain]
            similarities = map(lambda x: self.sequence_similarity(url, x), other_website_urls)
            average_similarity = sum(similarities)/len(other_website_urls)
            gain = (1-average_similarity) * 10/len(other_website_urls) #TODO finetuning
        return gain
    
    def find_new_documents(self, sorted_docs:list, pointer:int, added_pages:list) -> Tuple[int, str]:
        """
        Find pointer for a new document that is not already in the list
        :param sorted_docs: the list of the top k documents
        :param pointer: the current pointer
        :param added_pages: the list of already added pages
        :return: the new pointer and the url of the new document
        """
        while self.index_db[sorted_docs[pointer][0]][0] in added_pages:
            pointer += 1
        return pointer, self.index_db[sorted_docs[pointer][0]][0]        


    def merge_rankings(self, sorted_docs1: list, sorted_docs2: list, sorted_docs3: list) -> list:
        """
        Merge the rankings of the different methods
        :param sorted_docs1: the list of the top k documents for the first method (TF-IDF)
        :param sorted_docs2: the list of the top k documents for the second method (BM25)
        :param sorted_docs3: the list of the top k documents for the third method (pseudo relevance feedback)
        :return: the list of the top k documents after the merging
        """
        merged_ranking = []
        urls = []
        websites ={}
        added_pages =[]
        tf_idf_factor = 0.3
        bm25 =1
        pseudo_relevance_factor = 10
        list1_pointer = 0
        list2_pointer = 0
        list3_pointer = 0
        # naive merge take one with biggest increase
        for i in range(self.relevant_docs_count):
            #need to calculate the gain of each potenzial document
            list1_pointer, url1 = self.find_new_documents(sorted_docs1, list1_pointer, added_pages)
            list2_pointer, url2 = self.find_new_documents(sorted_docs2, list2_pointer, added_pages)
            list3_pointer, url3 = self.find_new_documents(sorted_docs3, list3_pointer, added_pages)
            list1_gain = self.gain(url1, urls, websites) + tf_idf_factor * sorted_docs1[list1_pointer][1]
            list2_gain = self.gain(url2, urls, websites) + bm25 * sorted_docs2[list2_pointer][1]
            list3_gain = self.gain(url3, urls, websites) + pseudo_relevance_factor / sorted_docs3[list3_pointer][1]
            if list1_gain > list2_gain and list1_gain > list3_gain:
                merged_ranking.append(sorted_docs1[list1_pointer])
                list1_pointer += 1
                url = self.get_url(url1)
                added_pages.append(url1)
                if url not in urls:
                    urls.append(url)
                if url not in websites.keys():
                    websites[url] = [url1]
                else:
                    websites[url].append(url1)
            elif list2_gain > list1_gain and list2_gain > list3_gain:
                merged_ranking.append(sorted_docs2[list2_pointer])
                list2_pointer += 1
                url = self.get_url(url2)
                added_pages.append(url2)
                if url not in urls:
                    urls.append(url)
                if url not in websites.keys():
                    websites[url] = [url2]
                else:
                    websites[url].append(url2)
            else:
                merged_ranking.append(sorted_docs3[list3_pointer])
                list3_pointer += 1
                url = self.get_url(url3)
                added_pages.append(url3)
                if url not in urls:
                    urls.append(url)
                if url not in websites.keys():
                    websites[url] = [url3]
                else:
                    websites[url].append(url3)
        return merged_ranking

    
    def pseudo_relevance_feedback_embedding(self, sorted_docs:list, top_k:int=4, mode:str ="distance") -> list:
        """
        Use pseudo relevance feedback to improve the ranking
        :param sorted_docs: the list of the top k documents
        :param top_k: the number of top documents to use for the feedback
        :param mode: the mode to use for the feedback, either distance or cosine
        :return: the list of the top k documents after the feedback
        """
        #TODO
        feedback = []
        for sorted_doc in sorted_docs:
            feedback.append(self.embeddings[self.id.index(sorted_doc[0])])
        mean_feedback = np.mean(feedback, axis=0)
        if mode == "distance":
            distance = np.linalg.norm(self.embeddings - mean_feedback, axis=1) #Euclidian distance
            relevant_indecs = np.argsort(distance)
            sorted_docs = []
            for i in range(self.relevant_docs_count):
                id = relevant_indecs[i]
                sorted_docs.append([self.id[id], distance[id]])
        elif mode == "cosine":
            sorted_docs =self.cosine_similarity(mean_feedback)
        return sorted_docs


    def load_embedding_index(self, embedding_index_path: str):
        """
        Load the embedding index
        :param embedding path
        """
        #TODO check if embedding was encoded with the some model
        embedding_index = load_index(embedding_index_path)
        # convert the embedding index to a numpy array
        self.id = []
        embedding = []
        for key, value in embedding_index.items():
            self.id.append(key)
            embedding.append(value)
        self.embeddings = np.stack(embedding, axis=0)

    def document_selection(self, query: str) -> list:
        """
        Select the documents that are relevant for the query
        :param query: the query string
        :return: the list of relevant documents
        """
        query = preprocessing(query)
        query = query.split()
        relevant_docs = []
        for word in query:
            if word in self.inverted_index_db.keys():
                relevant_docs.append(self.inverted_index_db[word])
        return relevant_docs


    def query_union(self,query: str) -> list:
        """
        Select the documents that are relevant for the query
        :param query: the query string
        :return: the list of relevant documents
        """
        query = preprocessing(query)
        query = query.split()
        relevant_words = []
        for word in query:
            if word in self.inverted_index_db.keys():
                relevant_words.append(self.inverted_index_db[word])
        relevant_docs = []
        for word in relevant_words:
            for doc in word:
                relevant_docs.append(doc[0])
        return sorted(set(relevant_docs))


    def listintersection(self,lista, listb):
        pointerA=0
        pointerB=0
        matches =[]
        print(lista)
        print(listb)
        while pointerA< len(lista) and pointerB< len(listb):
            if lista[pointerA][0]==listb[pointerB][0]:
                matches.append(lista[pointerA][0])
                pointerB+=1
                pointerA+=1
            elif lista[pointerA][0]<listb[pointerB][0]:
                pointerA+=1
            elif lista[pointerA][0]>listb[pointerB][0]:
                pointerB+=1
        return matches 

    def embedding_ranking(self,query: str) -> list:
        query_embedding = self.embedder.embed(query)
        return self.cosine_similarity(query_embedding)


    def cosine_similarity(self, query_embedding) -> list:
        query_norm = np.linalg.norm(query_embedding)
        embeddings_norm = np.linalg.norm(self.embeddings, axis=1)
        normalized_query = query_embedding / query_norm
        normalized_dataset = self.embeddings / embeddings_norm[:, np.newaxis]
        cosine_similarity = np.dot(normalized_dataset, normalized_query)
        relevant_indecs = np.flip(np.argsort(cosine_similarity))
        sorted_docs = []
        for i in range(self.relevant_docs_count):
            id = relevant_indecs[i]
            sorted_docs.append([self.id[id], cosine_similarity[id]])
        return sorted_docs

    def save_results(self, results: list, query: str):
        """
        Save the results in the given path
        :param results: the list of results
        :param query: the query string
        """
        file_name = query.replace(" ", "_") + "_"+ self.rank_method+ ".txt"
        file_path = os.path.join(self.results_path, file_name)
        with open(file_path, "w") as f:
            for i,result in enumerate(results):
                f.write(str(i+1) + "\t"+ str(self.index_db[result[0]][0]) + "\t" + str(result[1]) + "\n")
        


    def BM25(self, query: str, relevant_docs : list) -> list:
        """
        Calculate the BM25 score for the relevant documents
        :param query: the query string
        :param relevant_docs: the list of relevant documents
        :return: the top 100 documents with the highest BM25 score as a list of tuples (doc_id, score)
        """
        #TODO
        if len(relevant_docs) < self.relevant_docs_count:
            necessary_docs = len(relevant_docs)
        else:
            necessary_docs = self.relevant_docs_count
        query = preprocessing(query)
        query = query.split()
        IDF = np.zeros(len(query))
        for i in range(len(query)):
            n_q = len(self.inverted_index_db[query[i]])
            IDF[i] = np.log(((self.doc_count-n_q+0.5)/(n_q+0.5))+1)
        TF = np.zeros((len(relevant_docs), len(query)))
        for i,v in enumerate(query):
            docs = self.inverted_index_db[v]
            for doc in docs:
                doc_id = doc[0]
                if doc_id in relevant_docs:
                    f = len(doc[1])
                    D = self.index_db[doc_id][1] # TODO
                    TF[relevant_docs.index(doc_id), i] = (f * (self.k1+1))/(f+self.k1*(1-self.b+self.b*(D/self.avgdl)))
        BM25 = TF @ IDF # for all the relevant documents
        relevant_indecs = np.flip(np.argsort(BM25))
        sorted_docs = []
        for i in range(necessary_docs):
            id = relevant_indecs[i]
            sorted_docs.append([relevant_docs[id], BM25[id]])
        return sorted_docs

    def TF_IDF(self, query: str, relevant_docs : list) -> list:
        """
        Calculate the TF-IDF score for the relevant documents
        :param query: the query string
        :param relevant_docs: the list of relevant documents
        :return: the top 100 documents with the highest TF-IDF score as a list of tuples (doc_id, score)
        """
        if len(relevant_docs) < self.relevant_docs_count:
            necessary_docs = len(relevant_docs)
        else:
            necessary_docs = self.relevant_docs_count
        query = preprocessing(query)
        query = query.split()
        IDF = np.zeros(len(query))
        for i in range(len(query)):
            IDF[i] = np.log10(self.doc_count/len(self.inverted_index_db[query[i]]))
        TF = np.zeros((len(relevant_docs), len(query)))
        for i,v in enumerate(query):
            docs = self.inverted_index_db[v]
            for doc in docs:
                doc_id = doc[0]
                if doc_id in relevant_docs:
                    TF[relevant_docs.index(doc_id), i] = len(doc[1])
        TF_IDF = TF @ IDF # for all the relevant documents
        relevant_indecs = np.flip(np.argsort(TF_IDF))
        sorted_docs = []
        for i in range(necessary_docs):
            id = relevant_indecs[i]
            sorted_docs.append([relevant_docs[id], TF_IDF[id]])
        return sorted_docs
    

if __name__ == "__main__":
    path = 'data_files_bert_4'
    index = os.path.join(path, 'forward_index.joblib')
    index_inverted = os.path.join(path, 'inverted_index.joblib')
    index_embedding = os.path.join(path, 'embedding_index.joblib')
    result_path = os.path.join(path, 'results')
    ranker = Ranker(index, index_inverted, index_embedding, result_path, 100)
    # ["BM25", "TF-IDF", "Feature_embedding", "Pseudo_relevance_feedback", "Merge"]
    for method in ["Merge"]:
        ranker.rank_method = method
        ranker.rank("food and drinks")
        ranker.rank("tübingen attractions")
    #ranker.rank_method = "TF-IDF"
    #ranker.rank("food and drinks")
    #ranker.rank("tübingen attractions")
    #ranker.rank_method = "BM25"
    #ranker.rank("food and drinks")
    #ranker.rank("tübingen attractions")
    #ranker.rank_method = "embedding"
    #ranker.rank("food and drinks")
    #ranker.rank("tübingen attractions")
    #ranker.rank_method = "TF-IDF"
    #ranker.rank("food and drinks")
    #ranker.rank("tübingen attractions")

    #ranker.rank_method = "TF-IDF"
    #ranker.rank("food and drinks")
    