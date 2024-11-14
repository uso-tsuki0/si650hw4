import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../implementation'))
import unittest
import json
import numpy as np
from sentence_transformers import SentenceTransformer
# modules being tested
from vector_ranker import VectorRanker


def assertScoreLists(self, exp_list, res_list):
    self.assertEqual(len(exp_list), len(
        res_list), f'Expected length {len(exp_list)} but actual list length {len(res_list)}')
    for idx in range(len(res_list)):
        self.assertEqual(exp_list[idx][0], res_list[idx][0],
                         f'Expected document not at index {idx}')
        self.assertAlmostEqual(exp_list[idx][1], res_list[idx][1], places=4,
                               msg=f'Expected score differs from actual score at {idx}')


class TestVectorRanker(unittest.TestCase):
    def setUp(self) -> None:
        self.model_name = 'sentence-transformers/msmarco-MiniLM-L12-cos-v5'
        self.transformer = SentenceTransformer(self.model_name)
        self.doc_embeddings = []
        self.doc_ids = []
        with open('./dataset_2.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                self.doc_embeddings.append(self.transformer.encode(data['text']))
                self.doc_ids.append(data['docid'])
        self.doc_embeddings = np.array(self.doc_embeddings)

    def test_query_without_feedback(self):
        exp_list = [(2, 0.5097077488899231), (1, 0.38314518332481384), (3, 0.282781183719635)]
        ranker = VectorRanker(self.model_name, self.doc_embeddings, self.doc_ids)
        query = 'What is the second document?'
        res_list = ranker.query(query)
        assertScoreLists(self, exp_list, res_list)

    def test_query_with_feedback(self):
        # Tests the result when we use one document for pseudo feedback
        exp_list = [(2, 0.6077660918235779), (1, 0.4529471695423126), (3, 0.37983667850494385)]
        ranker = VectorRanker(self.model_name, self.doc_embeddings, self.doc_ids)
        query = 'What is the second document?'
        res_list = ranker.query(query, 1)  # Use the default alpha and beta
        assertScoreLists(self, exp_list, res_list)

    def test_weighting_parameters(self):
        # Tests the result when both alpha and beta are set to 0.5
        exp_list = [(2, 0.7548538446426392), (1, 0.5576502680778503), (3, 0.525420069694519)]
        ranker = VectorRanker(self.model_name, self.doc_embeddings, self.doc_ids)
        query = 'What is the second document?'
        res_list = ranker.query(query, 1, 0.5, 0.5)
        assertScoreLists(self, exp_list, res_list)

    def test_large_alpha_one_doc(self):
        # Tests the result when (1) alpha is set to one and (2) we use one document for pseudo feedback
        exp_list = [(2, 0.5097077488899231), (1, 0.38314518332481384), (3, 0.282781183719635)]
        ranker = VectorRanker(self.model_name, self.doc_embeddings, self.doc_ids)
        query = 'What is the second document?'
        res_list = ranker.query(query, 1, 1, 0)
        assertScoreLists(self, exp_list, res_list)

    def test_large_alpha_all_docs(self):
        # Tests the result when (1) alpha is set to one and (2) we use all documents for pseudo feedback
        exp_list = [(2, 0.5097077488899231), (1, 0.38314518332481384), (3, 0.282781183719635)]
        ranker = VectorRanker(self.model_name, self.doc_embeddings, self.doc_ids)
        query = 'What is the second document?'
        res_list = ranker.query(query, 3, 1, 0)
        assertScoreLists(self, exp_list, res_list)

    def test_large_beta_one_doc(self):
        # Tests the result when (1) beta is set to one and (2) we use one document for pseudo feedback
        exp_list = [(2, 0.9999998807907104), (3, 0.768058717250824), (1, 0.7321551442146301)]
        ranker = VectorRanker(self.model_name, self.doc_embeddings, self.doc_ids)
        query = 'What is the second document?'
        res_list = ranker.query(query, 1, 0, 1)
        assertScoreLists(self, exp_list, res_list)

    def test_large_beta_all_docs(self):
        # Tests the result when (1) beta is set to one and (2) we use all documents for pseudo feedback
        exp_list = [(2, 0.8334047198295593), (3, 0.8331509828567505), (1, 0.8211831450462341)]
        ranker = VectorRanker(self.model_name, self.doc_embeddings, self.doc_ids)
        query = 'What is the second document?'
        res_list = ranker.query(query, 3, 0, 1)
        assertScoreLists(self, exp_list, res_list)

    
if __name__ == '__main__':
    unittest.main()
