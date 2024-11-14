"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from collections import Counter, defaultdict
import numpy as np
from sentence_transformers import CrossEncoder
from indexing import InvertedIndex
import torch


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    # TODO: This class is responsible for returning a list of sorted relevant documents.
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int, str] = None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number
                 of top-ranked documents to be used in the query,
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        # TODO (HW4): If the user has indicated we should use feedback,
        #  create the pseudo-document from the specified number of pseudo-relevant results.
        #  This document is the cumulative count of how many times all non-filtered words show up
        #  in the pseudo-relevant documents. See the equation in the write-up. Be sure to apply the same
        #  token filtering and normalization here to the pseudo-relevant documents.

        # TODO (HW4): Combine the document word count for the pseudo-feedback with the query to create a new query
        # NOTE (HW4): Since you're using alpha and beta to weight the query and pseudofeedback doc, the counts
        #  will likely be *fractional* counts (not integers) which is ok and totally expected.

        """
        # 1. Tokenize query
        tokens = self.tokenize(query)
        query_word_counts = {}
        for token in tokens:
            if token not in self.stopwords:
                query_word_counts[token] = query_word_counts.get(token, 0) + 1
            else:
                # weird testcase considering the length of query before filtering
                query_word_counts['$$STOPWORD$$'] = query_word_counts.get('$$STOPWORD$$', 0) + 1

        # 2. Fetch a list of possible documents from the index
        possible_docs = set()
        for token in query_word_counts.keys():
            if token in self.index.vocabulary:
                for doc in self.index.index[token].keys():
                    possible_docs.add(doc)
        possible_docs = list(possible_docs)

        # 3. Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes)
        doc_score = []
        for docid in possible_docs:
            doc_word_counts = {}
            for token in query_word_counts.keys():
                if token in self.index.vocabulary:
                    cnt = self.index.get_token_doc_freq(token, docid)
                    if cnt > 0:
                        doc_word_counts[token] = cnt
            score = self.scorer.score(docid, doc_word_counts, query_word_counts)
            doc_score.append((docid, score))

        # 4. Return **sorted** results as format [(100, 0.5), (10, 0.2), ...]
        doc_score = sorted(doc_score, key=lambda x:-x[1])

        if pseudofeedback_num_docs <= 0:
            return doc_score
        else:
            pseudo_docs = [docid for docid, _ in doc_score[:pseudofeedback_num_docs]]
            pseudo_query = ' '.join([self.raw_text_dict[docid] for docid in pseudo_docs])
            pseudo_query = self.tokenize(pseudo_query)
            pseudo_query_word_counts = {}
            for token in pseudo_query:
                if token not in self.stopwords:
                    pseudo_query_word_counts[token] = pseudo_query_word_counts.get(token, 0) + 1
                else:
                    pseudo_query_word_counts['$$STOPWORD$$'] = pseudo_query_word_counts.get('$$STOPWORD$$', 0) + 1
            new_query_word_counts = {}
            for token in set(query_word_counts.keys()).union(set(pseudo_query_word_counts.keys())):
                new_query_word_counts[token] = pseudofeedback_alpha * query_word_counts.get(token, 0) + \
                                               pseudofeedback_beta * pseudo_query_word_counts.get(token, 0)/pseudofeedback_num_docs
            doc_score = []
            possible_docs = set()
            for token in new_query_word_counts.keys():
                if token in self.index.vocabulary:
                    for doc in self.index.index[token].keys():
                        possible_docs.add(doc)
            possible_docs = list(possible_docs)
            for docid in possible_docs:
                doc_word_counts = {}
                for token in new_query_word_counts.keys():
                    if token in self.index.vocabulary:
                        cnt = self.index.get_token_doc_freq(token, docid)
                        if cnt > 0:
                            doc_word_counts[token] = cnt
                score = self.scorer.score(docid, doc_word_counts, new_query_word_counts)
                doc_score.append((docid, score))
            doc_score = sorted(doc_score, key=lambda x: -x[1])
            return doc_score

class RelevanceScorer:
    """
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    """
    # NOTE: Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index: InvertedIndex, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        raise NotImplementedError


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        return 10


# TODO: Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        valid_words = [word for word in query_word_counts.keys() if word in doc_word_counts.keys() and word in self.index.vocabulary]
        dot_product = 0
        for word in valid_words:
            doc_count = doc_word_counts.get(word, 0)
            query_count = query_word_counts.get(word, 0)
            dot_product += doc_count * query_count

        # 2. Return the score
        return dot_product


# TODO Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters
        self.corpus_len = self.index.get_statistics()["total_token_count"]
        self.mu = self.parameters['mu']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        doc_len = self.index.get_doc_metadata(docid)['length']

        # 2. Compute additional terms to use in algorithm
        valid_words = [word for word in query_word_counts.keys() if word in doc_word_counts.keys() and word in self.index.vocabulary]
        query_len = sum(query_word_counts.values())
        total_score = query_len * np.log(self.mu / (doc_len + self.mu))

        # 3. For all query_parts, compute score
        for word in valid_words:
            query_count = query_word_counts[word]
            pwc = self.index.get_term_metadata(word)['term_count'] / self.corpus_len
            doc_term_count = doc_word_counts.get(word, 0)
            word_score = np.log(1 + doc_term_count / (self.mu * pwc))
            total_score += query_count * word_score
        
        # 4. Return the score
        return total_score



# TODO Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self.N = self.index.get_statistics()['number_of_documents']
        self.avg_doc_len = self.index.get_statistics()["mean_document_length"]
    
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index
        doc_len = self.index.get_doc_metadata(docid)['length']

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query
        valid_words = [word for word in query_word_counts.keys() if word in doc_word_counts.keys() and word in self.index.vocabulary]
        k1 = self.k1
        k3 = self.k3
        b = self.b
        avg_doc_len = self.avg_doc_len
        N = self.N

        total_score = 0
        # 3. For all query parts, compute the TF and IDF to get a score
        for word in valid_words:
            cd = doc_word_counts[word]
            cq = query_word_counts[word]
            doc_part = (k1 + 1) * cd / (k1 * (1 - b + b * doc_len / avg_doc_len) + cd)
            query_part = (k3 + 1) * cq / (k3 + cq)
            term_metadata = self.index.get_term_metadata(word)
            n = term_metadata['doc_frequency']
            idf = np.log((N - n + 0.5) / (n + 0.5))
            total_score += doc_part * query_part * idf

        # 4. Return score
        return total_score


# TODO (HW4): Implement Personalized BM25
class PersonalizedBM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, relevant_doc_index: InvertedIndex,
                 parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        """
        Initializes Personalized BM25 scorer.

        Args:
            index: The inverted index used to use for computing most of BM25
            relevant_doc_index: The inverted index of only documents a user has rated as relevant,
                which is used when calculating the personalized part of BM25
            parameters: The dictionary containing the parameter values for BM25

        Returns:
            The Personalized BM25 score
        """
        self.index = index
        self.relevant_doc_index = relevant_doc_index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self.N = self.index.get_statistics()['number_of_documents']
        self.R = self.relevant_doc_index.get_statistics()['number_of_documents']
        self.avg_doc_len = self.index.get_statistics()["mean_document_length"]

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # TODO (HW4): Implement Personalized BM25
        doc_len = self.index.get_doc_metadata(docid)['length']
        valid_words = [word for word in query_word_counts.keys() if word in doc_word_counts.keys() and word in self.index.vocabulary]
        k1 = self.k1
        k3 = self.k3
        b = self.b
        avg_doc_len = self.avg_doc_len
        N = self.N
        R = self.R
        total_score = 0
        for word in valid_words:
            cd = doc_word_counts[word]
            cq = query_word_counts[word]
            r = self.relevant_doc_index.get_term_metadata(word)['doc_frequency']
            doc_part = (k1 + 1) * cd / (k1 * (1 - b + b * doc_len / avg_doc_len) + cd)
            query_part = (k3 + 1) * cq / (k3 + cq)
            term_metadata = self.index.get_term_metadata(word)
            n = term_metadata['doc_frequency']
            idf = np.log((r+0.5)*(N-n-R+r+0.5)/((n-r+0.5)*(R-r+0.5)))
            total_score += doc_part * query_part * idf

        # 4. Return score
        return total_score



# TODO: Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']
        self.N = self.index.get_statistics()['number_of_documents']
        self.avg_doc_len = self.index.get_statistics()["mean_document_length"]

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index
        doc_len = self.index.get_doc_metadata(docid)['length']

        # 2. Compute additional terms to use in algorithm
        valid_words = [word for word in query_word_counts.keys() if word in doc_word_counts.keys() and word in self.index.vocabulary]
        b = self.b
        length_ratio = (1 - b + b * doc_len / self.avg_doc_len)
        total_score = 0

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
        for word in valid_words:
            cd = doc_word_counts.get(word, 0)
            tf = (1 + np.log(1 + np.log(cd))) / length_ratio
            cq = query_word_counts[word]
            term_metadata = self.index.get_term_metadata(word)
            n = term_metadata['doc_frequency']
            idf = np.log((self.N + 1) / n)
            total_score += tf * idf * cq

        # 4. Return the score
        return total_score



# TODO Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        self.N = self.index.get_statistics()['number_of_documents']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        N = self.N
        # 2. Compute additional terms to use in algorithm
        valid_words = [word for word in query_word_counts.keys() if word in doc_word_counts.keys() and word in self.index.vocabulary]
        total_score = 0
        for word in valid_words:
            cd = doc_word_counts.get(word, 0)
            tf = np.log(1 + cd)
            term_metadata = self.index.get_term_metadata(word)
            n = term_metadata['doc_frequency']
            idf = np.log(N / n) + 1
            total_score += tf * idf

        # 4. Return the score
        return total_score
    



class CrossEncoderScorer:
    '''
    A scoring object that uses cross-encoder to compute the relevance of a document for a query.
    '''
    def __init__(self, raw_text_dict: dict[int, str], 
                 cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model
        """
        # TODO: Save any new arguments that are needed as fields of this class
        self.raw_text_dict = {k: ' '.join(v.split(' ')[:500]) for k, v in raw_text_dict.items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self.model = CrossEncoder(cross_encoder_model_name, max_length=512, device=self.device)


    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.
        
        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        # NOTE: Do not forget to handle an edge case
        # (e.g., docid does not exist in raw_text_dict or empty query, both should lead to 0 score)
        if docid not in self.raw_text_dict or not query:
            return 0
        doc_text = self.raw_text_dict[docid]
        if len(doc_text) == 0 or len(query) == 0:
            return 0
        encoded_input = self.model.tokenizer(
            query,
            text_pair=doc_text,
            truncation=True,
            max_length=512,
            return_tensors='pt',
            padding='max_length'
        )
        encoded_input = {key: value.to(self.device) for key, value in encoded_input.items()}

        # NOTE: unlike the other scorers like BM25, this method takes in the query string itself,
        # not the tokens!

        # TODO (HW3): Get a score from the cross-encoder model
        #             Refer to IR_Encoder_Examples.ipynb in Demos folder on Canvas if needed
        with torch.no_grad():
            logits = self.model.model(**encoded_input).logits.to('cpu')
            score = logits[0][0]
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return score.item()