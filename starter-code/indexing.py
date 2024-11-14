'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
DO NOT use the pickle module.
'''

from enum import Enum
from document_preprocessor import Tokenizer, SplitTokenizer, RegexTokenizer, SpaCyTokenizer
from collections import Counter, defaultdict
import os
import time

try:
    import ujson as json
except ImportError:
    import json

TQDM = True
try:
    from tqdm import tqdm
except ImportError:
    TQDM = False

class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex, PositionalIndex
    PositionalIndex = 'PositionalIndex'
    BasicInvertedIndex = 'BasicInvertedIndex'
    SampleIndex = 'SampleIndex'
    InvertedIndex = 'InvertedIndex'


class InvertedIndex:
    """
    This class is the basic implementation of an in-memory inverted index. This class will hold the mapping of terms to their postings.
    The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
    and documents in the index. These metadata will be necessary when computing your relevance functions.
    """

    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        self.statistics = {}   # the central statistics of the index
        self.statistics['vocab'] = Counter() # token count
        self.vocabulary = set()  # the vocabulary of the collection
        self.document_metadata = {} # metadata like length, number of unique tokens of the documents

        self.index = defaultdict(defaultdict)  # the index 

    
    # NOTE: The following functions have to be implemented in the two inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        raise NotImplementedError

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        raise NotImplementedError
    
    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        raise NotImplementedError

    def save(self) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        self.statistics['number_of_documents'] = 0
        self.statistics['total_token_count'] = 0
    
    def add_doc(self, docid: int, tokens: list[str]) -> None:
        doc_counter = Counter(tokens)
        token_positions = {}
        for i, token in enumerate(tokens):
            if token not in token_positions:
                token_positions[token] = []
            else:
                token_positions[token].append(i)
        self.document_metadata[docid] = {
            'unique_tokens': len(doc_counter),
            'length': len(tokens),
            'unique_token_list': list(doc_counter.keys())
        }
        for token in doc_counter:
            self.index[token][docid] = doc_counter[token]
            self.statistics['vocab'][token] += doc_counter[token]
            self.vocabulary.add(token)
        self.statistics['number_of_documents'] += 1
        self.statistics['total_token_count'] += len(tokens)
        
    def remove_doc(self, docid: int) -> None:
        index_del = []
        for token in self.index:
            if docid in self.index[token]:
                self.statistics['vocab'][token] -= self.index[token][docid]
                del self.index[token][docid]
                if self.statistics['vocab'][token] == 0:
                    index_del.append(token)
        for token in index_del:
            del self.index[token]
            self.vocabulary.discard(token)
        del self.document_metadata[docid]
        self.statistics['number_of_documents'] -= 1
        
    def get_postings(self, term: str) -> list:
        postings = []
        for docid, freq in self.index[term].items():
            postings.append((docid, freq))
        return postings

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        return {
            'unique_tokens': self.document_metadata[doc_id]['unique_tokens'],
            'length': self.document_metadata[doc_id]['length']
        }
    
    def get_token_doc_freq(self, token: str, docid: int) -> int:
        if token not in self.index:
            return 0
        return self.index[token].get(docid, 0)
    
    def get_term_metadata(self, term: str) -> dict[str, int]:
        if term not in self.vocabulary:
            return {
                'term_count': 0,
                'doc_frequency': 0
            }
        else:
            return {
                'term_count': self.statistics['vocab'][term],
                'doc_frequency': len(self.index[term])
            }
        
    def get_statistics(self) -> dict[str, int]:
        if self.statistics['number_of_documents'] == 0:
            return {
                "unique_token_count": len(self.vocabulary),
                "total_token_count": self.statistics['total_token_count'],
                "stored_total_token_count": 0,
                "number_of_documents": self.statistics['number_of_documents'],
                "mean_document_length": 0
            }
        else:
            return {
                "unique_token_count": len(self.vocabulary),
                "total_token_count": self.statistics['total_token_count'],
                "stored_total_token_count": sum(self.statistics['vocab'].values()),
                "number_of_documents": self.statistics['number_of_documents'],
                "mean_document_length": self.statistics['total_token_count']/self.statistics['number_of_documents']
            }
    
    def save(self, index_directory_name:str) -> None:
        os.makedirs(index_directory_name, exist_ok=True)
        total_dict = {
            'statistics': {
                'vocab': dict(self.statistics['vocab']),
                'number_of_documents': self.statistics['number_of_documents'],
                'total_token_count': self.statistics['total_token_count']
            },
            'vocabulary': list(self.vocabulary),
            'document_metadata': self.document_metadata,
            'index': self.index
        }
        with open((index_directory_name+'/index.json'), 'w') as f:
            json.dump(total_dict, f)
        print('Saved!')

    def load(self, index_directory_name:str) -> None:
        with open((index_directory_name+'/index.json'), 'r') as f:
            total_dict = json.load(f)
        self.statistics['vocab'] = Counter(total_dict['statistics']['vocab'])
        self.statistics['number_of_documents'] = total_dict['statistics']['number_of_documents']
        self.statistics['total_token_count'] = total_dict['statistics']['total_token_count']
        self.vocabulary = set(total_dict['vocabulary'])
        self.document_metadata = {}
        for key, value in total_dict['document_metadata'].items():
            self.document_metadata[int(key)] = value
        for token, doc_dict in total_dict['index'].items():
            self.index[token] = defaultdict(int)
            for docid, freq in doc_dict.items():
                self.index[token][int(docid)] = freq
        print('Loaded!')

    def remove_token(self, token: str) -> None:
        if token in self.vocabulary:
            for docid in self.index[token].keys():
                self.document_metadata[docid]['unique_tokens'] -= 1
            del self.index[token]
            del self.statistics['vocab'][token]
            self.vocabulary.discard(token)

    def filter_frequency(self, min_freq: int) -> None:
        tokens_to_remove = []
        for token in self.vocabulary:
            if self.statistics['vocab'][token] < min_freq:
                tokens_to_remove.append(token)
        for token in tokens_to_remove:
            self.remove_token(token)
    

        

class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__()
        
    def add_doc(self, docid: int, tokens: list[str]) -> None:
        doc_counter = Counter(tokens)
        token_positions = {}
        for i, token in enumerate(tokens):
            token_positions[token] = token_positions.get(token, []) + [i]
        self.document_metadata[docid] = {
            'unique_tokens': len(doc_counter),
            'length': len(tokens),
        }
        for token in doc_counter:
            self.index[token][docid] = [doc_counter[token], token_positions[token]]
            self.statistics['vocab'][token] += doc_counter[token]
            self.vocabulary.add(token)
        self.statistics['number_of_documents'] += 1
        self.statistics['total_token_count'] += len(tokens)


    def get_postings(self, term: str) -> list:
        postings = []
        for docid, info in self.index[term].items():
            postings.append((docid, info[0], info[1]))
        return postings

    

class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''

    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1, doc_augment_dict: dict[int, list[str]] | None = None) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the entire corpus at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
            doc_augment_dict: An optional argument; This is a dict created from the doc2query.csv where the keys are
                the document id and the values are the list of queries for a particular document.

        Returns:
            An inverted index

        '''
         # TODO (HW3): This function now has an optional argument doc_augment_dict; check README.md
       
        # HINT: Think of what to do when doc_augment_dict exists, how can you deal with the extra information?
        #       How can you use that information with the tokens?
        #       If doc_augment_dict doesn't exist, it's the same as before, tokenizing just the document text
          
        # TODO: Implement this class properly. This is responsible for going through the documents
        #       one by one and inserting them into the index after tokenizing the document

        # TODO: Figure out what type of InvertedIndex to create.
        #       For HW3, only the BasicInvertedIndex is required to be supported

        # TODO: If minimum word frequencies are specified, process the collection to get the
        #       word frequencies

        # NOTE: Make sure to support both .jsonl.gz and .jsonl as input
                      
        # TODO: Figure out which set of words to not index because they are stopwords or
        #       have too low of a frequency

        # TODO: Read the collection and process/index each document.
        #       Only index the terms that are not stopwords and have high-enough frequency

        inv_index = InvertedIndex()
        if index_type == IndexType.BasicInvertedIndex:
            inv_index = BasicInvertedIndex()
        elif index_type == IndexType.PositionalIndex:
            inv_index = PositionalInvertedIndex()
        else:
            inv_index = BasicInvertedIndex()
        total_lines = 0
        with open(os.path.normpath(dataset_path), 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
        progress_bar = tqdm(total=min(max_docs if max_docs != -1 else total_lines, total_lines), desc="Indexing documents")
        try:
            if TQDM:
                with open(os.path.normpath(dataset_path), 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= max_docs and max_docs != -1:
                            break
                        doc = json.loads(line)
                        docid = doc['docid']
                        text = doc[text_key]
                        if doc_augment_dict is not None:
                            if docid in doc_augment_dict:
                                text += ' ' + ' '.join(doc_augment_dict[docid])
                        tokens = document_preprocessor.tokenize(text)
                        inv_index.add_doc(docid, tokens)
                        progress_bar.update(1)
            else:
                with open(os.path.normpath(dataset_path), 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= max_docs and max_docs != -1:
                            break
                        doc = json.loads(line)
                        docid = doc['docid']
                        text = doc[text_key]
                        if doc_augment_dict is not None:
                            if docid in doc_augment_dict:
                                text += ' ' + ' '.join(doc_augment_dict[docid])
                        tokens = document_preprocessor.tokenize(text)
                        inv_index.add_doc(docid, tokens)
        except FileNotFoundError:
            print(f'File not found at {os.path.normpath(dataset_path)}')
        stop_set = set()
        if stopwords is not None:
            stop_set = set(stopwords)
        if minimum_word_frequency > 0:
            for token in inv_index.vocabulary:
                if inv_index.statistics['vocab'][token] < minimum_word_frequency:
                    stop_set.add(token)
        for token in stop_set:
            if token in inv_index.vocabulary:
                inv_index.remove_token(token)
        return inv_index


'''
The following class is a stub class with none of the essential methods implemented. It is merely here as an example.
'''


class SampleIndex(InvertedIndex):
    '''
    This class does nothing of value
    '''

    def add_doc(self, docid, tokens):
        """Tokenize a document and add term ID """
        for token in tokens:
            if token not in self.index:
                self.index[token] = {docid: 1}
            else:
                self.index[token][docid] = 1

    def save(self):
        print('Index saved!')
