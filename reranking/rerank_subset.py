import os
import pickle
import time
import builtins
import torch

import numpy as np
import statistics

from copy import deepcopy
from math import floor
from collections import Counter

import datetime

from recbole.config import Config
from recbole.evaluator.evaluator import Evaluator

import warnings
warnings.simplefilter("ignore")

def print(*args, **kwargs):
    with open(f'reranking/log/log_{filename_prefix}_{exp_name}.txt', 'a+') as f:
        return builtins.print(*args, file=f, **kwargs)

def load_dataset_model(dataset: str, model: str, k: int, top_k_extend: int):
    # print(f"Loading {dataset} -- {model}")

    list_file = os.listdir("cluster/best_struct/")
    file_for_dataset_model = [x for x in list_file if dataset in x and model in x]
    assert len(file_for_dataset_model) == 1

    with open("cluster/best_struct/"+file_for_dataset_model[0],"rb") as f:
        struct = pickle.load(f)
    
    rec_score = deepcopy(struct.get("rec.score"))

    rec_topk = deepcopy(struct.get("rec.topk"))
    pos_item = struct.get("data.pos_items")

    #get top k extended items
    if k == top_k_extend:
        rec_items = deepcopy(struct.get("rec.items")[:,:k])
    else:
        rec_items = rec_score[:,1:]\
                                .sort(descending=True, stable=True)\
                                .indices[:,:top_k_extend]+1 #because item index starts from 1
    
    return struct, rec_score, rec_items, rec_topk, pos_item

def compute_coverage(rec_items, k: int, top_k_extend: int):
    item_id, num_times = torch.unique(rec_items.flatten(), return_counts=True)
    coverage = Counter(dict(map(lambda tup: (x.item() for x in tup), zip(item_id, num_times))))
    if k != top_k_extend:
        item_id, num_times_outside_top_k = torch.unique(rec_items[:,k:].flatten(), return_counts=True)
        coverage_outside_top_k = Counter(dict(map(lambda tup: (x.item() for x in tup), zip(item_id, num_times_outside_top_k))))
        coverage.subtract(coverage_outside_top_k)
    return coverage

def update_param(coverage, beta=0):
    vals = coverage.values()

    cov_avg = statistics.mean(vals)
    cov_max = max(vals)
    # cov_min = min(vals)

    cov_sup = min(floor(cov_avg + beta*cov_max), cov_max)
    cov_inf = cov_sup - beta*cov_max

    #Generate all items to be replaced and generate candidates for replacement
    c_plus = {x: count for x, count in coverage.items() if count > cov_sup}
    c_min =  {x: count for x, count in coverage.items() if count < cov_inf}

    #special case
    case1 = len(c_plus) == 0
    case2 = len(c_min)== 0
    if case1 or case2:
        c0 = {x: count for x, count in coverage.items() if x not in c_plus and x not in c_min}

        if case1:
            c_plus = {x: count for x, count in c0.items() if count != cov_inf}
        elif case2:
            c_min = {x: count for x, count in c0.items() if count != cov_inf}

    return c_plus, c_min

def greedy_sub(rec_items, rec_score, rec_topk, pos_item, coverage, c_plus, c_min, k, beta):

    gs_rec_items = deepcopy(rec_items) # width is according to extended k
    gs_rec_score = deepcopy(rec_score)
    gs_rec_topk = deepcopy(rec_topk)

    full_rec_mat = gs_rec_score[:,1:].sort(descending=True, stable=True).indices +1 #+1 because the 0-th column is -inf (decoy)

    thresh = round(0.25 * gs_rec_items[:,:10].numel())

    list_to_replace = []

    for i in c_plus:
        for i_prime in c_min:

            #find all users with i in the top k
            user_with_i = torch.where(gs_rec_items[:,:k]==i)[0]
            
            #filter user with i, selecting those without i prime in the top k
            user_with_i_prime = torch.where(gs_rec_items[:,:k]==i_prime)[0]
            mask = ~torch.isin(user_with_i, user_with_i_prime)
            user_with_i = user_with_i[mask]

            #create p(u,i) and p(u,i') matrices
            p_ui = gs_rec_score[user_with_i][:,i]
            p_ui_prime = gs_rec_score[user_with_i][:,i_prime]

            min_val = (p_ui - p_ui_prime).min()
            u_to_replace = user_with_i[(p_ui - p_ui_prime).argmin()]
            i_to_replace = i
            i_replacement = i_prime

            list_to_replace.append((u_to_replace, i_to_replace, i_replacement, min_val))

    sorted_list = sorted(list_to_replace, key=lambda x: x[-1])
    sorted_list_thresh = sorted_list[:thresh]

    for rep_id, tup in enumerate(sorted_list_thresh):
        u_to_replace = tup[0]
        i_to_replace = tup[1]
        i_replacement = tup[2] 
        min_val = tup[3]

        print(f"{rep_id}: replace {i_to_replace} with {i_replacement} for user {u_to_replace}, relevance loss: {min_val}")
        #get the original indices first, THEN replace i in recommendation list of user u with item i'
        
        #ensure idx_i_in_u not empty:
        idx_i_in_u_helper = torch.where(gs_rec_items[u_to_replace]==i_to_replace)[0]
        if len(idx_i_in_u_helper) > 0:
            idx_i_in_u = idx_i_in_u_helper[0]
        else:
            continue
        for_idx_i_replacement_in_u = torch.where(gs_rec_items[u_to_replace]==i_replacement)[0]

        gs_rec_items[u_to_replace][idx_i_in_u] = i_replacement

        if len(for_idx_i_replacement_in_u) > 0:
        #this should not happen when only top k is reranked

            idx_i_replacement_in_u = for_idx_i_replacement_in_u[0]
        
            if idx_i_replacement_in_u < k:
                print("Replacement item exist in the top k of the same user, swapping position")
                gs_rec_items[u_to_replace][idx_i_replacement_in_u] = i_to_replace
                
                #update the relevance value too
                if i in pos_item[u_to_replace]:
                    gs_rec_topk[u_to_replace][idx_i_replacement_in_u] = 1
                else:
                    gs_rec_topk[u_to_replace][idx_i_replacement_in_u] = 0
        else:
            idx_i_replacement_in_u = torch.where(full_rec_mat[u_to_replace]==i_replacement)[0][0]

        
        #also in full recmat, but we for sure have to switch in the full rec mat, otherwise one item recommended twice and another disappear
        full_rec_mat[u_to_replace][idx_i_in_u] = i_replacement
        full_rec_mat[u_to_replace][idx_i_replacement_in_u] = i_to_replace

        #update relevance
        if i_prime in pos_item[u_to_replace]: 
            gs_rec_topk[u_to_replace][idx_i_in_u] = 1
        else:
            gs_rec_topk[u_to_replace][idx_i_in_u] = 0

    return gs_rec_items, gs_rec_topk, full_rec_mat

def rerank_greedy_sub(rec_score, rec_items, rec_topk, pos_item, k, top_k_extend, beta):
    coverage = compute_coverage(rec_items, k, top_k_extend)
    c_plus, c_min = update_param(coverage, beta)
    rec_items, rec_topk, full_rec_mat = greedy_sub(rec_items, rec_score, rec_topk, pos_item, coverage, c_plus, c_min, k, beta)
    rec_items = rec_items[:, :k] #only take top k as final recommendation
    rec_topk = torch.cat((rec_topk[:,:k], rec_topk[:,-1:]), axis=1)
    return rec_items, rec_topk, full_rec_mat

def get_score_per_user(item, recommendation, score, mode="borda"):
    #to be used with borda and combnz
    match mode:
        case "borda":
            return torch.where(recommendation==item, score, 0).sum(1)
        case "combmnz":
            #only until k to see if the item is recommended at top k given the old/new ranking
            return torch.where(recommendation[:,:k]==item, score[:,:k], 0).sum(1)

def rerank_borda(rec_score, rec_items, rec_topk, pos_item, k, top_k_extend):
    coverage = compute_coverage(rec_items, k, top_k_extend)
    gs_rec_score = deepcopy(rec_score)
    borda_rec_topk = deepcopy(rec_topk)
    borda_rec_topk = torch.cat((borda_rec_topk[:,:k], borda_rec_topk[:,-1:]), axis=1)

    #within the extended top-k, get ranking based on relevance (ori_rec_items)
    ori_rec_items = deepcopy(rec_items)

    #within the extended top-k, get ranking based on coverage (new_rec_items)
    least_to_most_coverage = sorted(coverage, key=lambda i: coverage.get(i))
    for_indices = deepcopy(rec_items)
    for_indices.apply_(lambda x: least_to_most_coverage.index(x)) #inplace
    new_rec_items = for_indices.sort().values.apply_(lambda x: least_to_most_coverage[x])

    #point matrix
    num_user = new_rec_items.shape[0]
    points = torch.tensor([range(top_k_extend-1,-1,-1)] * num_user)

    #sum up the points
    dict_point_ori_rec_items = {i: get_score_per_user(i, ori_rec_items, points) for i in coverage.keys()}
    dict_point_new_rec_items = {i: get_score_per_user(i, new_rec_items, points) for i in coverage.keys()}
    total_point = {key: dict_point_ori_rec_items[key] + dict_point_new_rec_items[key] for key in coverage.keys()}

    borda_score = torch.zeros_like(gs_rec_score, dtype=torch.int64) #coverage is integer
    for key, val in total_point.items():
        borda_score[:,key] = val


    borda_full_rec_mat = borda_score[:,1:]\
                                .sort(descending=True, stable=True)\
                                .indices + 1 #first column is dummy
    borda_rec_item = borda_full_rec_mat[:,:k]

    #update relevance value
    for u in range(len(borda_rec_item)):
        new_indicator = torch.isin(borda_rec_item[u], torch.from_numpy(pos_item[u])).int()
        borda_rec_topk[u,:-1] = new_indicator

    borda_rec_topk = torch.cat((borda_rec_topk[:,:k], borda_rec_topk[:,-1:]), axis=1)
    return borda_rec_item, borda_rec_topk, borda_full_rec_mat

def combmnz(dict_score_rec_items, dict_score_new_rec_items, coverage):
    total_score = {}

    for key in coverage.keys():
        val_ori = dict_score_rec_items[key]
        val_new = dict_score_new_rec_items[key]

        stacked = np.stack([val_ori, val_new], axis=1)

        multiplier = np.count_nonzero(stacked, axis=1)

        total_score[key] = stacked.sum(1) * multiplier

    return total_score

def rerank_combmnz(rec_score, rec_items, rec_topk, pos_item, k, top_k_extend):
    coverage = compute_coverage(rec_items, k, top_k_extend)
    ori_pred_rel = deepcopy(rec_score)
    combmnz_rec_topk = deepcopy(rec_topk)
    combmnz_rec_topk = torch.cat((combmnz_rec_topk[:,:k], combmnz_rec_topk[:,-1:]), axis=1)

    #within the extended top-k, get ranking based on relevance (ori_rec_items)
    ori_rec_items = deepcopy(rec_items)
    ori_pred_rel_extended_top_k = ori_pred_rel\
                                                .sort(descending=True, stable=True)\
                                                .values[:,:top_k_extend]

    #min-max normalise relevance score
    max_pred_rel = ori_pred_rel_extended_top_k.max()
    min_pred_rel = np.nanmin(ori_pred_rel_extended_top_k[ori_pred_rel_extended_top_k != -np.inf])
    normalised_rel_items = (ori_pred_rel_extended_top_k - min_pred_rel) / (max_pred_rel-min_pred_rel)

    #within the extended top-k, get ranking based on coverage (new_rec_items)
    least_to_most_coverage = sorted(coverage, key=lambda i: coverage.get(i))
    for_indices = deepcopy(rec_items)
    for_indices.apply_(lambda x: least_to_most_coverage.index(x)) #inplace
    new_rec_items = for_indices.sort().values.apply_(lambda x: least_to_most_coverage[x])

    #get score from coverage of items
    coverage_rec_items = deepcopy(new_rec_items)
    coverage_rec_items.apply_(lambda x: coverage[x]) #inplace

    #min-max normalise coverage score, and calculate 1-score (to promote least covered item)
    max_cov = coverage_rec_items.max()
    min_cov = coverage_rec_items.min()

    normalised_coverage_items = 1 - (coverage_rec_items-min_cov) / (max_cov - min_cov)

    dict_score_rec_items = {i: get_score_per_user(i, ori_rec_items, normalised_rel_items, "combmnz") for i in coverage.keys()}
    dict_score_new_rec_items = {i: get_score_per_user(i, new_rec_items, normalised_coverage_items, "combmnz") for i in coverage.keys()}

    total_score = combmnz(dict_score_rec_items, dict_score_new_rec_items, coverage)

    mnz_score = torch.zeros_like(rec_score, dtype=torch.float64)

    for key, val in total_score.items():
        mnz_score[:,key] = torch.from_numpy(val)


    combmnz_full_rec_mat = mnz_score[:,1:]\
                                .sort(descending=True, stable=True)\
                                .indices + 1 #first column is dummy
    combmnz_rec_item = combmnz_full_rec_mat[:,:k]

    #update relevance value
    for u in range(len(combmnz_rec_item)):
        new_indicator = torch.isin(combmnz_rec_item[u], torch.from_numpy(pos_item[u])).int()
        combmnz_rec_topk[u,:-1] = new_indicator

    combmnz_rec_topk = torch.cat((combmnz_rec_topk[:,:k], combmnz_rec_topk[:,-1:]), axis=1)

    return combmnz_rec_item, combmnz_rec_topk, combmnz_full_rec_mat

def evaluate(ori_struct, rec_items, rec_topk, full_rec_mat):
    #update the recommended items, true relevance value of item at the top k, and insert the full rec mat to evaluate IAA just for reranking
    start_time = time.time()
    updated_struct = deepcopy(ori_struct)
    updated_struct.set("rec.items", rec_items)
    updated_struct.set("rec.topk", rec_topk)
    updated_struct.set("rec.all_items", full_rec_mat)

    result = evaluator.evaluate(updated_struct)
    print(result)
    print("Time taken to evaluate: ", time.time() - start_time)
        
    return result, updated_struct

def save_result_struct(exp_name, curr_result, struct):
    with open(f"reranking/result/{filename_prefix}_{exp_name}.pickle","wb") as f:
        pickle.dump(
            curr_result,
            f, 
            pickle.HIGHEST_PROTOCOL
        )
    with open(f"reranking/rerank_struct/{filename_prefix}_{exp_name}.pickle","wb") as f:
        pickle.dump(
            struct,
            f, 
            pickle.HIGHEST_PROTOCOL
        )                     

#start main
list_k = [10]
list_beta = [0.05]

list_dataset = [
                "Amazon-lb", 
                "Lastfm", 
                "Jester", 
                "ML-10M",
                "ML-20M",
                "QK-video", 
                ]

list_model = [
            "ItemKNN",
              "MultiVAE",
              "BPR",
              "NCL",
              ]


list_extended_k = [25]

for dataset in list_dataset:

    for model in list_model:

        
        for k in list_k:

            config = Config(
                    model=model, 
                    dataset=dataset, 
                    config_file_list=["RecBole/recbole/properties/overall.yaml"], 
                    config_dict={"topk":k,
                                "metrics":[ 
                                            "RelMetrics", 
                                            "FairWORel",
                                            "IAArerank",
                                            "IIF_AIF",
                                            "IBO",
                                            "MME"
                                            ]
                                }
                    )
            evaluator = Evaluator(config)


            for extended_k in list_extended_k:
                filename_prefix = f"{dataset}_{model}_at{k}_rerank{extended_k}"
                ori_struct, rec_score, rec_items, rec_topk, pos_item = load_dataset_model(dataset, model, k, extended_k)

                # COMBMNZ and BORDA will only cause change in fairness scores if we rerank outside the top-k, because they are used for user-wise reranking
                if extended_k != k:
                    #reranking with COMBMNZ
                    exp_name = "combmnz"

                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(now)
                    print("Starting combmnz")
                    combmnz_rec_items, combmnz_rec_topk, combmnz_full_rec_mat = rerank_combmnz(rec_score, rec_items, rec_topk, pos_item, k, extended_k)

                    combmnz_result, combnz_struct = evaluate(ori_struct, combmnz_rec_items, combmnz_rec_topk, combmnz_full_rec_mat)
                    save_result_struct(exp_name, combmnz_result, combnz_struct)
                

                    #reranking with BORDACOUNT
                    exp_name = "borda"
                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(now)
                    print("Starting borda")
                    borda_rec_items, borda_rec_topk, borda_full_rec_mat = rerank_borda(rec_score, rec_items, rec_topk, pos_item, k, extended_k)

                    borda_result, borda_struct = evaluate(ori_struct, borda_rec_items, borda_rec_topk, borda_full_rec_mat)
                    save_result_struct(exp_name, borda_result, borda_struct)


                #reranking with greedy substitution
                for beta in list_beta:
                    exp_name = f"GS-subset-{beta}"
                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(now)
                    print(f"Starting GS with beta={beta}")
                    start_time = time.time()
                    
                    gs_rec_items, gs_rec_topk, gs_full_rec_mat = rerank_greedy_sub(rec_score, rec_items, rec_topk, pos_item, k, extended_k, beta=beta)

                    print("Time taken to do greedy substitution: ", time.time() - start_time)

                    gs_result, gs_struct = evaluate(ori_struct, gs_rec_items, gs_rec_topk, gs_full_rec_mat)
            
                    save_result_struct(exp_name, gs_result, gs_struct)

        