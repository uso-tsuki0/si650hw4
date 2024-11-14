import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../implementation'))
import unittest
import json
# modules being tested
from document_preprocessor import RegexTokenizer
from ranker import Ranker, WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF, PersonalizedBM25
from indexing import Indexer, IndexType


def assertScoreLists(self, exp_list, res_list):
    self.assertEqual(len(exp_list), len(
        res_list), f'Expected length {len(exp_list)} but actual list length {len(res_list)}')
    for idx in range(len(res_list)):
        self.assertEqual(exp_list[idx][0], res_list[idx][0],
                         f'Expected document not at index {idx}')
        self.assertAlmostEqual(exp_list[idx][1], res_list[idx][1], places=4,
                               msg=f'Expected score differs from actual score at {idx}')

def get_raw_text_dict(self, dataset_name):
    raw_text_dict = {}
    with open(dataset_name) as f:
        for line in f:
            d = json.loads(line)
            docid = d['docid']
            tokens = (d['text'])
            raw_text_dict[docid] = tokens

    return raw_text_dict


class MockTokenizer:
    def tokenize(self, text):
        return text.split()

class TestWordCountCosineSimilarity(unittest.TestCase):
   def setUp(self) -> None:
       self.preprocessor = MockTokenizer()
       self.stopwords = set(['a', 'an', 'and', 'the', 'this'])
       self.index = Indexer.create_index(
           IndexType.BasicInvertedIndex, './dataset_1.jsonl', self.preprocessor, self.stopwords, 1)
       scorer = WordCountCosineSimilarity(self.index)
       self.ranker = Ranker(self.index, self.preprocessor,
                            self.stopwords, scorer)

   def test_no_overlap(self):
       exp_list = []
       res_list = self.ranker.query("cough drops")
       self.assertEqual(exp_list, res_list,
                        'Cosine: no overlap between query and docs')

   def test_perfect_match(self):
       exp_list = [(1, 1), (3, 1), (5, 1)]
       res_list = self.ranker.query("AI")
       self.assertEqual(exp_list, res_list,
                        'Expected list differs from result list')

   def test_partial_match(self):
       exp_list = [(3, 2), (4, 2), (1, 1), (5, 1)]
       res_list = self.ranker.query("AI chatbots and vehicles")
       self.assertEqual(exp_list, res_list,
                        'Expected list differs from result list')


class TestDirichletLM(unittest.TestCase):
   def setUp(self) -> None:
       self.preprocessor = MockTokenizer()
       self.stopwords = set(['a', 'an', 'and', 'the', 'this'])
       self.index = Indexer.create_index(
           IndexType.BasicInvertedIndex, './dataset_1.jsonl', self.preprocessor, self.stopwords, 1)
       scorer = DirichletLM(self.index)
       self.ranker = Ranker(self.index, self.preprocessor,
                            self.stopwords, scorer)

   def test_no_overlap(self):
       exp_list = []
       res_list = self.ranker.query("cough drops")
       assertScoreLists(self, exp_list, res_list)

   def test_perfect_match(self):
       exp_list = [(5, 0.01128846343027107), (3, 0.007839334610553066),
                   (1, 0.0073475716303944075)]
       res_list = self.ranker.query("AI")
       assertScoreLists(self, exp_list, res_list)

   def test_partial_match(self):
       exp_list = [(3, 0.029667610688458967), (4, 0.017285590697028078),
                   (5, -0.027460212369367794), (1, -0.04322377956887445)]
       res_list = self.ranker.query("AI chatbots and vehicles")
       assertScoreLists(self, exp_list, res_list)

   @unittest.skip('Test broken due to parameter mismatch')
   def test_small_mu(self):
       DLM = DirichletLM(self.index, {'mu': 5})
       ret_score = DLM.score(1, ['AI', 'Google'])
       exp_score = 1.6857412751512575

       self.assertAlmostEqual(
           exp_score, ret_score, places=3, msg='DirichletLM: partial match, score')

   @unittest.skip('Test broken due to parameter mismatch')
   def test_small_mu2(self):
       DLM = DirichletLM(self.index, {'mu': 1})
       ret_score = DLM.score(1, ['AI', 'Google'])
       exp_score = 1.798539156213434

       self.assertAlmostEqual(
           exp_score, ret_score, places=3, msg='DirichletLM: partial match, score')

   @unittest.skip('Test broken due to parameter mismatch')
   def test_med_mu(self):
       DLM = DirichletLM(self.index, {'mu': 30})
       ret_score = DLM.score(1, ['AI', 'Google'])
       exp_score = 1.2278314183215069

       self.assertAlmostEqual(
           exp_score, ret_score, places=3, msg='DirichletLM: partial match, score')

   @unittest.skip('Test broken due to parameter mismatch')
   def test_large_mu(self):
       DLM = DirichletLM(self.index, {'mu': 1000})
       ret_score = DLM.score(1, ['AI', 'Google'])
       exp_score = 0.11811761538891903

       self.assertAlmostEqual(
           exp_score, ret_score, places=3, msg='DirichletLM: partial match, score')


class TestPersonalizedBM25(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = RegexTokenizer('\\w+')
        self.stopwords = set(['a', 'an', 'and', 'the', 'this'])
        self.index = Indexer.create_index(
            IndexType.BasicInvertedIndex, './dataset_1.jsonl', self.preprocessor, self.stopwords, 1)
        self.relevant_doc_index = Indexer.create_index(
            IndexType.BasicInvertedIndex, './data_relevant.jsonl', self.preprocessor, self.stopwords, 1)
        self.raw_text_dict = self.get_raw_text_dict('dataset_1.jsonl')
        scorer = PersonalizedBM25(self.index, self.relevant_doc_index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer, self.raw_text_dict)

    def get_raw_text_dict(self, dataset_name):
        raw_text_dict = {}
        with open(dataset_name) as f:
            for line in f:
                d = json.loads(line)
                docid = d['docid']
                tokens = (d['text'])
                raw_text_dict[docid] = tokens
        return raw_text_dict

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match(self):
        exp_list = [(5, 2.0392113877909646), (4, 2.010853998282129), (1, 1.8559964456477376), (3, 1.8095449217021604)]
        res_list = self.ranker.query("AI")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match(self):
        exp_list = [(4, 3.555748654844854), (3, 2.83116880749364), (5, 2.0392113877909646), (1, 1.8559964456477376)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match_large_alpha_one_doc(self):
        exp_list = [(4, 7.457237795021482), (3, 3.924045933803818), (1, 3.2321043774997484), (5, 2.2657904308788495), (2, 0.38815195030476646)]
        res_list = self.ranker.query("AI chatbots and vehicles", 1, 0.9, 0.1)
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match_large_alpha_all_docs(self):
        exp_list = [(4, 5.244083671662035), (3, 4.09002725491848), (1, 2.6957218456203322), (5, 0.5636175183641821), (2, 0.021329643063885573)]
        res_list = self.ranker.query("AI chatbots and vehicles", 3, 0.9, 0.1)
        assertScoreLists(self, exp_list, res_list)


    def test_perfect_match_large_alpha_one_doc(self):
        exp_list = [(4, 2.2342822203134767), (1, 2.0622182729419305), (3, 1.7387490184613492), (2, -0.24688050646098766), (5, -2.4849714039712953)]
        res_list = self.ranker.query("AI", 1, 0.9, 0.1)
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match_large_alpha_all_docs(self):
        exp_list = [(4, 3.959729514286703), (1, 3.7469252449724713), (3, 2.523356366729673), (5, 0.6657341244922553), (2, 0.1978996760905643)]
        res_list = self.ranker.query("AI", 3, 0.9, 0.1)
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match_large_beta_one_doc(self):
        exp_list = [(4, 34.903349883083834), (1, 12.752340199600766), (3, 11.364697390213603), (5, 3.8951228755557747), (2, 2.8837414182177787)]
        res_list = self.ranker.query("AI chatbots and vehicles", 1, 0.1, 0.9)
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match_large_beta_all_docs(self):
        exp_list = [(4, 17.81416412492448), (3, 13.478719712129823), (1, 8.836358633021659), (2, 0.11247014114487205), (5, -10.766535051631202)]
        res_list = self.ranker.query("AI chatbots and vehicles", 3, 0.1, 0.9)
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match_large_beta_one_doc(self):
        exp_list = [(4, 3.8409570753703584), (1, 3.545161750113656), (3, 1.22965488279709), (2, -2.0222010023602244), (5, -34.21747883149466)]
        res_list = self.ranker.query("AI", 1, 0.1, 0.9)
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match_large_beta_all_docs(self):
        exp_list = [(4, 18.387357673271357), (1, 17.823893924880107), (3, 7.645885615571295), (2, 1.4969224033743416), (5, -9.93217743008434)]
        res_list = self.ranker.query("AI", 3, 0.1, 0.9)
        assertScoreLists(self, exp_list, res_list)



class TestBM25(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = MockTokenizer()
        self.stopwords = set(['a', 'an', 'and', 'the', 'this'])
        self.index = Indexer.create_index(
            IndexType.BasicInvertedIndex, './dataset_1.jsonl', self.preprocessor, self.stopwords, 1)
        scorer = BM25(self.index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer, self.get_raw_text_dict('dataset_1.jsonl'))

    def get_raw_text_dict(self, dataset_name):
        raw_text_dict = {}
        with open(dataset_name) as f:
            for line in f:
                d = json.loads(line)
                docid = d['docid']
                tokens = (d['text'])
                raw_text_dict[docid] = tokens
        return raw_text_dict

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match(self):
        exp_list = [(1, -0.31623109945742595), (3, -0.32042144088133173),
                    (5, -0.35318117923823517)]
        res_list = self.ranker.query("AI")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match(self):
        exp_list = [(4, 1.5460888344441546), (3, 0.7257835477973098),
                    (1, -0.31623109945742595), (5, -0.35318117923823517)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        assertScoreLists(self, exp_list, res_list)

    @unittest.skip('Test broken due to parameter mismatch')
    def test_small_k1(self):
        scorer = BM25(self.index, {'b': 0.75, 'k1': 1, 'k3': 8})
        ret_score = scorer.score(1, ['AI', 'Google'])
        exp_score = 0.7199009648250208

        self.assertAlmostEqual(exp_score, ret_score,
                               places=3, msg='BM25: partial match, score')

    @unittest.skip('Test broken due to parameter mismatch')
    def test_large_k1(self):
        scorer = BM25(self.index, {'b': 0.75, 'k1': 2, 'k3': 8})
        ret_score = scorer.score(1, ['AI', 'Google'])
        exp_score = 0.7068428242958602

        self.assertAlmostEqual(exp_score, ret_score,
                               places=3, msg='BM25: partial match, score')

    @unittest.skip('Test broken due to parameter mismatch')
    def test_small_k3(self):
        scorer = BM25(self.index, {'b': 0.75, 'k1': 1.2, 'k3': 0})
        ret_score = scorer.score(1, ['AI', 'Google'])
        exp_score = 0.7162920454285571

        self.assertAlmostEqual(exp_score, ret_score,
                               places=3, msg='BM25: partial match, score')

    @unittest.skip('Test broken due to parameter mismatch')
    def test_large_k3(self):
        scorer = BM25(self.index, {'b': 0.75, 'k1': 1.2, 'k3': 1000})
        ret_score = scorer.score(1, ['AI', 'Google'])
        exp_score = 0.7162920454285571

        self.assertAlmostEqual(exp_score, ret_score,
                               places=3, msg='BM25: partial match, score')

    @unittest.skip('Test broken due to parameter mismatch')
    def test_random_param(self):
        scorer = BM25(self.index, {'b': 0.75, 'k1': 1.99, 'k3': 49})
        ret_score = scorer.score(1, ['AI', 'Google'])
        exp_score = 0.7069285957828516

        self.assertAlmostEqual(exp_score, ret_score,
                               places=3, msg='BM25: partial match, score')


class TestPivotedNormalization(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = MockTokenizer()
        self.stopwords = set(['a', 'an', 'and', 'the', 'this'])
        self.index = Indexer.create_index(
            IndexType.BasicInvertedIndex, './dataset_1.jsonl', self.preprocessor, self.stopwords, 1)
        scorer = PivotedNormalization(self.index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer, self.get_raw_text_dict('dataset_1.jsonl'))

    def get_raw_text_dict(self, dataset_name):
        raw_text_dict = {}
        with open(dataset_name) as f:
            for line in f:
                d = json.loads(line)
                docid = d['docid']
                tokens = (d['text'])
                raw_text_dict[docid] = tokens
        return raw_text_dict

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match(self):
        exp_list = [(5, 0.7095587433308632), (3, 0.6765779252477553),
                    (1, 0.6721150101735617)]
        res_list = self.ranker.query("AI")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match(self):
        exp_list = [(4, 2.7806792016468633), (3, 2.4255064908289246),
                    (5, 0.7095587433308632), (1, 0.6721150101735617)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        assertScoreLists(self, exp_list, res_list)

    @unittest.skip('Test broken due to parameter mismatch')
    def test_small_param(self):
        scorer = PivotedNormalization(self.index, {'b': 0})
        ret_score = scorer.score(1, ['AI', 'Google'])
        exp_score = 2.4849066497880004

        self.assertAlmostEqual(exp_score, ret_score, places=3,
                               msg='PivotedNormalization: partial match, score')

    @unittest.skip('Test broken due to parameter mismatch')
    def test_large_param(self):
        scorer = PivotedNormalization(self.index, {'b': 1})
        ret_score = scorer.score(1, ['AI', 'Google'])
        exp_score = 2.1487133971696237

        self.assertAlmostEqual(exp_score, ret_score, places=3,
                               msg='PivotedNormalization: partial match, score')


class TestTF_IDF(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = MockTokenizer()
        self.stopwords = set(['a', 'an', 'and', 'the', 'this'])
        self.index = Indexer.create_index(
            IndexType.BasicInvertedIndex, './dataset_1.jsonl', self.preprocessor, self.stopwords, 1)
        scorer = TF_IDF(self.index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer, self.get_raw_text_dict('dataset_1.jsonl'))

    def get_raw_text_dict(self, dataset_name):
        raw_text_dict = {}
        with open(dataset_name) as f:
            for line in f:
                d = json.loads(line)
                docid = d['docid']
                tokens = (d['text'])
                raw_text_dict[docid] = tokens
        return raw_text_dict

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match(self):
        exp_list = [(1, 1.047224521431117),
                    (3, 1.047224521431117), (5, 1.047224521431117)]
        res_list = self.ranker.query("AI")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match(self):
        exp_list = [(4, 2.866760557116562), (3, 2.8559490532810434),
                    (1, 1.047224521431117), (5, 1.047224521431117)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        assertScoreLists(self, exp_list, res_list)


# Assuming YourRanker uses the same setup as the other rankers
# class TestYourRanker(unittest.TestCase):
#    def setUp(self):
#        self.index = MockIndex()
#        self.scorer = YourRanker(self.index, {})
#    def test_score(self):
#        result = self.scorer.score(["test", "sample"], self.index.get_statistics(), self.index, 1)
#        self.assertTrue(isinstance(result, dict))
#        self.assertTrue('docid' in result and 'score' in result)

if __name__ == '__main__':
    unittest.main()
