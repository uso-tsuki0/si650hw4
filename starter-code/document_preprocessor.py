"""
This is the template for implementing the tokenizer for your search engine.
You will be testing some tokenization techniques.
"""
import re
import nltk
import spacy
import spacy.cli
import time
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration



class TrieNode:
    def __init__(self):
        self.children = {}
        self.legal = False  # Marks the end of a valid phrase



class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, words: list):
        if not words:
            return
        ptr = self.root
        for word in words:
            if word in ptr.children:
                ptr = ptr.children[word]
            else:
                new_node = TrieNode()
                ptr.children[word] = new_node
                ptr = new_node
        ptr.legal = True

    def insert_multi_list(self, multi_list: list):
        if not multi_list:
            return
        for phrase in multi_list:
            words = phrase.split(' ')
            self.insert(words)
        


class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """
        # TODO: Save arguments that are needed as fields of this class
        self.lowercase = lowercase
        if lowercase and multiword_expressions:
            self.multiword_expressions = [expression.lower() for expression in multiword_expressions]
        else:
            self.multiword_expressions = multiword_expressions
    
    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and multi-word-expression handling. After that, return the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition
        """
        # TODO: Add support for lower-casing and multi-word expressions
        if self.lowercase:
            input_tokens = [token.lower() for token in input_tokens]
        if self.multiword_expressions:
            trie = Trie()
            trie.insert_multi_list(self.multiword_expressions)
            result = []
            index = 0
            while index < len(input_tokens):
                node = trie.root
                matched_index = -1
                current_index = index
                while current_index < len(input_tokens) and input_tokens[current_index] in node.children:
                    node = node.children[input_tokens[current_index]]
                    if node.legal:
                        matched_index = current_index
                    current_index += 1
                if matched_index != -1:
                    phrase = ' '.join(input_tokens[index:matched_index + 1])
                    result.append(phrase)
                    index = matched_index + 1
                else:
                    result.append(input_tokens[index])
                    index += 1
            return result
        return input_tokens 
        
    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # You should implement this in a subclass, not here
        raise NotImplementedError('tokenize() is not implemented in the base class; please use a subclass')



class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses the split function to tokenize a given string.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)


    def tokenize(self, text: str) -> list[str]:
        """
        Split a string into a list of tokens using whitespace as a delimiter.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        tokens = text.split()
        return self.postprocess(tokens)



class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses the split function to tokenize a given string.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)


    def tokenize(self, text: str) -> list[str]:
        """
        Split a string into a list of tokens using whitespace as a delimiter.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        tokens = text.split()
        return self.postprocess(tokens)


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = r'\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        # TODO: Save a new argument that is needed as a field of this class
        if multiword_expressions:
            escaped_expressions = [re.escape(expr) for expr in self.multiword_expressions]
            special_expr = '|'.join(escaped_expressions)
            self.token_regex = f'{special_expr}|{token_regex}'
        else:
            self.token_regex = token_regex

        # TODO: Initialize the NLTK's RegexpTokenizer 
        self.tokenizer = nltk.RegexpTokenizer(self.token_regex)


    def tokenize(self, text: str) -> list[str]:
        """
        Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # TODO: Tokenize the given text and perform postprocessing on the list of tokens
        #       using the postprocess function
        if self.lowercase:
            tokens = self.tokenizer.tokenize(text.lower())
        else:
            tokens = self.tokenizer.tokenize(text)
        return tokens
        


class SpaCyTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Use a spaCy tokenizer to convert named entities into single words. 
        Check the spaCy documentation to learn about the feature that supports named entity recognition.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        self.nlp = spacy.load("en_core_web_sm")
        for expression in multiword_expressions:
            self.nlp.tokenizer.add_special_case(expression, [{"ORTH": expression}])
        self.nlp.add_pipe("merge_entities")

    def tokenize(self, text: str) -> list[str]:
        """
        Use a spaCy tokenizer to convert named entities into single words.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        doc = self.nlp(text)
        return self.postprocess([item.text for item in doc])
    

class Doc2QueryAugmenter:
    """
    This class is responsible for generating queries for a document.
    These queries can augment the document before indexing.

    MUST READ: https://huggingface.co/doc2query/msmarco-t5-base-v1

    OPTIONAL reading
        1. Document Expansion by Query Prediction (Nogueira et al.): https://arxiv.org/pdf/1904.08375.pdf
    """
    
    def __init__(self, doc2query_model_name: str = 'doc2query/msmarco-t5-base-v1') -> None:
        """
        Creates the T5 model object and the corresponding dense tokenizer.
        
        Args:
            doc2query_model_name: The name of the T5 model architecture used for generating queries
        """
        #self.device = torch.device('cuda')  # Do not change this unless you know what you are doing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # TODO (HW3): Create the dense tokenizer and query generation model using HuggingFace transformers
        self.tokenizer = T5Tokenizer.from_pretrained(doc2query_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(doc2query_model_name).to(self.device)

    def get_queries(self, document: str, n_queries: int = 5, prefix_prompt: str = '') -> list[str]:
        """
        Steps
            1. Use the dense tokenizer/encoder to create the dense document vector.
            2. Use the T5 model to generate the dense query vectors (you should have a list of vectors).
            3. Decode the query vector using the tokenizer/decode to get the appropriate queries.
            4. Return the queries.
         
            Ensure you take care of edge cases.
         
        OPTIONAL (DO NOT DO THIS before you finish the assignment):
            Neural models are best performing when batched to the GPU.
            Try writing a separate function which can deal with batches of documents.
        
        Args:
            document: The text from which queries are to be generated
            n_queries: The total number of queries to be generated
            prefix_prompt: An optional parameter that gets added before the text.
                Some models like flan-t5 are not fine-tuned to generate queries.
                So we need to add a prompt to instruct the model to generate queries.
                This string enables us to create a prefixed prompt to generate queries for the models.
                See the PDF for what you need to do for this part.
                Prompt-engineering: https://en.wikipedia.org/wiki/Prompt_engineering
        
        Returns:
            A list of query strings generated from the text
        """
        # Note: Feel free to change these values to experiment
        document_max_token_length = 400  # as used in OPTIONAL Reading 1
        top_p = 0.85

        # NOTE: See https://huggingface.co/doc2query/msmarco-t5-base-v1 for details

        # TODO (HW3): For the given model, generate a list of queries that might reasonably be issued to search
        #       for that document
        # NOTE: Do not forget edge cases
        if not document or len(document) == 0:
            return []
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prefix_prompt+document, max_length=document_max_token_length, truncation=True, return_tensors='pt').to(self.device)
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=64,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=n_queries
            )
        results = []
        for i in range(len(outputs)):
            query = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            results.append(query)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return results



# Don't forget that you can have a main function here to test anything in the file
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    file_path = 'wikipedia_200k_dataset.jsonl'
    multi_word_path = 'data/multi_word_expressions.txt'
    limit = 1000
    docs = []
    multi_words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            data = json.loads(line)
            text = data['text']
            docs.append(text)
    with open(multi_word_path, 'r', encoding='utf-8') as f:
        for line in f:
            multi_words.append(line.strip())

    tokenizer_map = {
        'SplitTokenizer': SplitTokenizer(lowercase=True, multiword_expressions=multi_words),
        'RegexTokenizer': RegexTokenizer(token_regex=r'\w+', lowercase=True, multiword_expressions=multi_words),
        'SpaCyTokenizer': SpaCyTokenizer(lowercase=True, multiword_expressions=multi_words)
    }
    tokenizer_scores = {}
    for name, tokenizer in tokenizer_map.items():
        start_time = time.time()
        for doc in docs:
            tokenizer.tokenize(doc)
        end_time = time.time()
        elapsed_time = end_time - start_time
        tokenizer_scores[name] = elapsed_time
        print(f'{name} took {elapsed_time} seconds to tokenize {limit} documents')
    bars = sns.barplot(x=list(tokenizer_scores.keys()), y=list(tokenizer_scores.values()))
    plt.ylabel('Time (s)')
    plt.xlabel('Tokenizers')
    bars.bar_label(bars.containers[0])
    plt.show()
