import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../implementation'))
import unittest
import pandas as pd
# modules being tested
from network_features import NetworkFeatures

class TestNetworkFeatures(unittest.TestCase):

    def test_load_network(self):
        '''Test loading the network using load_network (with CSV file)'''

        nf = NetworkFeatures()

        network = nf.load_network('toy-network.csv', 80)

        self.assertEqual(len(network.names), 20, 'Expected 20 nodes in graph')

    def test_load_network_gz(self):
        '''Test loading the network using load_network (with GZ file)'''

        nf = NetworkFeatures()

        network = nf.load_network('toy-network.csv.gz', 80)

        self.assertEqual(len(network.names), 20, 'Expected 20 nodes in graph')

    def test_calculate_page_rank(self):
        '''Test calculating the Pagerank scores'''

        nf = NetworkFeatures()
        network = nf.load_network('toy-network.csv', 80)
        
        pr_scores = [0.05125961, 0.05951749, 0.04892111, 0.05118604, 0.08388691,
                     0.05019047, 0.06019026, 0.06067016, 0.0379761 , 0.02879423,
                     0.0707906 , 0.05905573, 0.04016703, 0.06474813, 0.02881887,
                     0.04909074, 0.01851644, 0.02785495, 0.04900831, 0.05935682 ]
        
        est_pr_scores = list(nf.calculate_page_rank(network, weights=None))

        for pr, est_pr in zip(pr_scores, est_pr_scores):
            self.assertAlmostEqual(est_pr, pr, places=3,
                                   msg='PageRank scores do not match')
        
    def test_calculate_page_rank_random_weights(self):
        '''Test calculating the Pagerank scores'''

        nf = NetworkFeatures()
        network = nf.load_network('toy-network.csv', 80)

        weights   = {i: w for i, w in enumerate([0.07652791, 0.01072198, 0.04272346, 0.03483903, 0.04443277, 
                     0.0312399,  0.05502874, 0.07246399, 0.08816051, 0.04774878,
                     0.06816517, 0.04907787, 0.03601704, 0.00551931, 0.08229834, 
                     0.04100174, 0.04849918, 0.02559986, 0.08956767, 0.05036674])}
        
        pr_scores = [0.05612454686480185, 0.056494157789259536, 0.046790378707990125, 0.05104379887970217, 0.08315660100037643, 0.044171890101144085, 0.06006861330328686, 0.06633070998798647, 0.043596423179655085, 0.029186414900649642, 0.07281049180621953, 0.05659696045616512, 0.03756240538535933, 0.054581314468071525, 0.034467799667375926, 0.04699753604491001, 0.016553694892361682, 0.024252414895602224, 0.05644482079546731, 0.06276902687361513]
        
        est_pr_scores = list(nf.calculate_page_rank(network, weights=weights))

        print(est_pr_scores)

        for pr, est_pr in zip(pr_scores, est_pr_scores):
            self.assertAlmostEqual(est_pr, pr, places=3,
                                   msg='PageRank scores do not match')

    def test_calculate_page_rank_same_weights(self):
        '''Test calculating the Pagerank scores'''

        nf = NetworkFeatures()
        network = nf.load_network('toy-network.csv', 80)
        
        weights =   {i: w for i, w in enumerate([0.05, 0.05, 0.05, 0.05, 0.05,
                     0.05, 0.05, 0.05, 0.05, 0.05,
                     0.05, 0.05, 0.05, 0.05, 0.05, 
                     0.05, 0.05, 0.05, 0.05, 0.05])}
    
        pr_scores = [0.05125961, 0.05951749, 0.04892111, 0.05118604, 0.08388691,
                     0.05019047, 0.06019026, 0.06067016, 0.0379761 , 0.02879423,
                     0.0707906 , 0.05905573, 0.04016703, 0.06474813, 0.02881887,
                     0.04909074, 0.01851644, 0.02785495, 0.04900831, 0.05935682 ]
    
        est_pr_scores = list(nf.calculate_page_rank(network, weights=weights))

        for pr, est_pr in zip(pr_scores, est_pr_scores):
            self.assertAlmostEqual(est_pr, pr, places=3,
                                   msg='PageRank scores do not match')

    def test_calculate_page_rank_skewed_weights_1(self):
        '''Test calculating the Pagerank scores'''

        nf = NetworkFeatures()
        network = nf.load_network('toy-network.csv', 80)
        
#        weights =   [0.5, 0.025, 0.025, 0.025, 0.025,
#                     0.025, 0.025, 0.025, 0.025, 0.025,
#                     0.025, 0.025, 0.025, 0.025, 0.025, 
#                     0.025, 0.025, 0.025, 0.025, 0.025]
#
        weights =   {i: w for i, w in enumerate([1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 
                     0, 0, 0, 0, 0, 
                     0, 0, 0, 0, 0])}

    
        pr_scores = [0.21981578899953225, 0.0783977574624923, 0.028772664727135945, 0.0374250277636735, 0.07229897425480257, 0.036289801139554774, 0.03569990034560637, 0.031537859718283975, 0.030271703156105355, 0.055489959643626074, 0.056643415574798114, 0.04070903264292321, 0.06237519508651592, 0.032835595724852756, 0.01331423245986829, 0.02602895841834492, 0.005582060592793992, 0.016803607292944087, 0.07167936641480786, 0.04802909858133772]
    
        est_pr_scores = list(nf.calculate_page_rank(network, weights=weights))

        for pr, est_pr in zip(pr_scores, est_pr_scores):
            self.assertAlmostEqual(est_pr, pr, places=3,
                                   msg='PageRank scores do not match')

    def test_calculate_page_rank_skewed_weights_2(self):
        '''Test calculating the Pagerank scores'''

        nf = NetworkFeatures()
        network = nf.load_network('toy-network.csv', 80)
        
        weights =   {i: w for i, w in enumerate([0.5, 0.025, 0.025, 0.025, 0.025,
                     0.025, 0.025, 0.025, 0.025, 0.025,
                     0.025, 0.025, 0.025, 0.025, 0.025, 
                     0.025, 0.025, 0.025, 0.025, 0.025])}
    
        pr_scores = [0.13349818765961569, 0.06865351445939771, 0.039037913972059884, 0.04454415223946969, 0.0783951817407677, 0.043456674872473004, 0.04817133393884983, 0.04654870562596495, 0.03419901588867618, 0.04173387707803748, 0.06396970726527865, 0.05014454428527253, 0.050914633443870295, 0.049203105281571374, 0.02122504424747971, 0.037810896271968646, 0.012210691112710038, 0.022427906270967787, 0.05995860159649107, 0.05389631274907781]
    
        est_pr_scores = list(nf.calculate_page_rank(network, weights=weights))

        for pr, est_pr in zip(pr_scores, est_pr_scores):
            self.assertAlmostEqual(est_pr, pr, places=3,
                                   msg='PageRank scores do not match')


    def test_calculate_hits(self):
        '''Test calculating the HITS scores'''

        nf = NetworkFeatures()
        network = nf.load_network('toy-network.csv', 80)

        actual_hub_scores, actual_authority_scores = nf.calculate_hits(network)

        # hub == auth for the toy network
        expected_scores = [0.16546851, 0.27122355, 0.23465519, 0.18502592, 0.32562661,
                      0.19676417, 0.29508263, 0.25060985, 0.19212819, 0.1095624,
                      0.31498321, 0.27880198, 0.12209237, 0.24641294, 0.09718952,
                      0.23205709, 0.05497426, 0.14291885, 0.2388066, 0.26433229]
        
        est_hub_scores, est_auth_scores = nf.calculate_hits(network)
        est_hub_scores = list(est_hub_scores)
        est_auth_scores = list(est_auth_scores)

        for hub, est_hub, est_auth in zip(expected_scores, est_hub_scores, est_auth_scores):
            self.assertAlmostEqual(est_hub, hub, places=3,
                                   msg='Hub scores do not match')
            self.assertAlmostEqual(est_auth, hub, places=3,
                                   msg='Auh scores do not match')
        
    def test_get_all_network_statistics(self):
        nf = NetworkFeatures()
        expected_df = pd.read_csv('network_stats.csv')
        network = nf.load_network('toy-network.csv', 80)
        actual_df = nf.get_all_network_statistics(network)
        pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False, check_exact=False)



if __name__ == '__main__':
    unittest.main()
