import pandas as pd
import math
TQDM = True
try:
    from tqdm import tqdm
except ImportError:
    TQDM = False

"""
NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (marginally relevant) and 2 (very relevant). When you calculate MAP, treat 1s and 2s are relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant. Thus, we have 
three relevance levels: 0 (non-relevant), 1 (marginally relevant), and 2 (very relevant). 
"""

def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    search_result_relevances = search_result_relevances[:cut_off]
    num_relevant = 0
    precision_sum = 0.0
    for i, relevance in enumerate(search_result_relevances):
        if relevance == 1:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            precision_sum += precision_at_i
    if num_relevant == 0:
        return 0.0
    return precision_sum / len(search_result_relevances)


    


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_of: int = 10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # TODO: Implement NDCG
    dcg = 0
    idcg = 0
    for i in range(cut_of):
        if i > 0:
            dcg += (search_result_relevances[i]) / math.log2(i+1)
            idcg += (ideal_relevance_score_ordering[i]) / math.log2(i+1)
        else:
            dcg += search_result_relevances[i]
            idcg += ideal_relevance_score_ordering[i]
    if idcg == 0:
        return 0.0
    return dcg / idcg



def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # TODO: Load the relevance dataset
    rev_df = pd.read_csv(relevance_data_filename, encoding='ISO-8859-1')
    score_map = {}
    for index, row in rev_df.iterrows():
        query = row['query']
        docid = row['docid']
        rel = row['rel']
        score_map[query] = score_map.get(query, {})
        score_map[query][docid] = rel


    # TODO: Run each of the dataset's queries through your ranking function
    result_map = {}
    for query in score_map.keys():
        ranked = ranker.query(query)
        docids = []
        map_rev = []
        ndcg_rev = []
        for tuple in ranked:
            docids.append(tuple[0])
            if int(tuple[0]) in score_map[query]:
                map_rev.append(int(score_map[query][int(tuple[0])] > 3))
                ndcg_rev.append(score_map[query][int(tuple[0])])
            else:
                map_rev.append(0)
                ndcg_rev.append(1)
        result_map[query] = {'docids': docids, 'map_rev': map_rev, 'ndcg_rev': ndcg_rev}
        

    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out
    map_scores = []
    ndcg_scores = []
    map_sum = 0
    ndcg_sum = 0
    if TQDM:
        for query in tqdm(result_map.keys()):
            m_score = map_score(result_map[query]['map_rev'])
            n_score = ndcg_score(result_map[query]['ndcg_rev'], sorted(result_map[query]['ndcg_rev'], reverse=True))
            map_scores.append(m_score)
            ndcg_scores.append(n_score)
            map_sum += m_score
            ndcg_sum += n_score
    else:
        for query in result_map.keys():
            m_score = map_score(result_map[query]['map_rev'])
            n_score = ndcg_score(result_map[query]['ndcg_rev'], sorted(result_map[query]['ndcg_rev'], reverse=True))
            map_scores.append(m_score)
            ndcg_scores.append(n_score)
            map_sum += m_score
            ndcg_sum += n_score


    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance 
    #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.

    # NOTE: NDCG can use any scoring range, so no conversion is needed.
  
    # TODO: Compute the average MAP and NDCG across all queries and return the scores
    # NOTE: You should also return the MAP and NDCG scores for each query in a list
    return {'map': map_sum/len(map_scores), 'ndcg': ndcg_sum/len(ndcg_scores), 'map_list': map_scores, 'ndcg_list': ndcg_scores}



if __name__ == '__main__':
    pass