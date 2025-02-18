import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from math import ceil
import os
import time

import builtins
import datetime
import copy
import warnings
warnings.simplefilter("ignore")

from recbole.config import Config
from recbole.evaluator.evaluator import Evaluator
    
import torch

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter




# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", default="Amazon-lb", help="Name of dataset")
args = vars(parser.parse_args())

# Set up parameters
dataset = args["dataset"]
dataset = "new_" + dataset

k = 10
path = "pareto/result"



def print(*args, **kwargs):
    with open(f'{path}/log_{dataset}_at{k}.txt', 'a+') as f:
        return builtins.print(*args, file=f, **kwargs)

def timenow(status=False):
    if not status:
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def load_dataset(dataset):

    with open(f"train_val_test/{dataset}_train.pickle","rb") as f:
        data = pickle.load(f)
    train = pd.DataFrame(data)

    with open(f"train_val_test/{dataset}_valid.pickle","rb") as f:
        data = pickle.load(f)
    val = pd.DataFrame(data)

    with open(f"train_val_test/{dataset}_test.pickle","rb") as f:
        data = pickle.load(f)
    test = pd.DataFrame(data)

    config = Config(
                model="Pop", 
                dataset=dataset, 
                config_file_list=["RecBole/recbole/properties/overall.yaml"], 
                config_dict={"topk":k,
                             "metrics":[ 
                                        "RelMetrics", 
                                        "FairWORel",
                                        ]
                             }
                )
    evaluator = Evaluator(config)
    item_id = config.final_config_dict["ITEM_ID_FIELD"]

    train = train.groupby("user_id")\
        .agg(lambda x: [x for x in x])\
        [item_id]

    val = val.groupby("user_id")\
        .agg(lambda x: [x for x in x])\
        [item_id]

    test = test.groupby("user_id")\
        .agg(lambda x: [x for x in x])\
        [item_id]

    df = pd.DataFrame()
    df["train"] = train.apply(set)
    df["valid"] = val.apply(set)
    df["pure_test"] = test.apply(set)

    df_test = df[~df.pure_test.isna()]

    df = df.applymap(lambda x: set() if type(x) == float else x)
    df_test = df_test.applymap(lambda x: set() if type(x) == float else x)
    return df, df_test, test, evaluator


def update_all_measure(results):
    rec = torch.tensor(df_test.recommendation.to_list(), dtype=torch.int64)
    struct.set("rec.items",rec)
    struct.set("rec.topk",torch.cat([torch.Tensor(df_test.rel.to_list()), torch.Tensor(df_test.num_rel.to_list()).reshape(-1,1)], axis=1).int())

    result = evaluator.evaluate(struct)
    results.append(result)
    if len(results) % 200 == 0:
        print(f"{timenow()}: {(time.time() - start_time)/60} mins has elapsed")
        print(f"{timenow()}: Saving the {len(results)}-th result")
        save_result(results, df_test, status="temp")


def fast_counter(series_to_count):
    my_counter = Counter()
    for row in series_to_count:
        my_counter.update(row)
    return my_counter

def fast_set_items(series_to_set):
    set_of_items = set()

    for row in series_to_set:
        set_of_items.update(set(row))
    
    return set_of_items

def get_frequency_count(recommendation):
    if all(recommendation.apply(type)) == set:
        all_rec_item = recommendation.apply(list)
    else:
        all_rec_item = recommendation

    all_rec_item_count = fast_counter(all_rec_item)
    return all_rec_item_count

def get_most_pop_item(recommendation):
    all_rec_item_count = get_frequency_count(recommendation)
    most_pop_item_in_rec, count = all_rec_item_count.most_common(1)[0]
    return most_pop_item_in_rec, count

def get_least_pop_item(recommendation):
    freq_count = get_frequency_count(recommendation)
    least_common, count = freq_count.most_common()[-1]
    return least_common, count

def get_least_pop_filtered_item(recommendation, user_id):
    
    recommended_items = fast_set_items(recommendation.recommendation)

    avail_item = recommended_items - df.loc[user_id,"train"] - df.loc[user_id,"valid"] - df.loc[user_id,"pure_test"]

    #and not in their current recommendation list
    avail_item = avail_item - set(recommendation.loc[user_id, "recommendation"])

    if len(avail_item) == 0:
        user_with_no_item_to_recommend.append(user_id)
    else:
        freq_count = get_frequency_count(recommendation.recommendation)
        #filter those that are not available
        for key in freq_count.keys():
            if key not in avail_item:
                freq_count[key] = 0
        freq_count = +freq_count
        least_common = freq_count.most_common()[-1][0]

    return least_common


def save_oracle(recommendation, status, all_item_not_in_test=None):


    with open(f"{path}/pareto_{dataset}_oracle_at{k}_{status}.pickle","wb") as f:
        pickle.dump(
            recommendation,
            f, 
            pickle.HIGHEST_PROTOCOL
        )

    if "final" in status and "ML" in dataset:
        all_item_not_in_test = list(all_item_not_in_test)
        with open(f"{path}/pareto_{dataset}_allitemnotintest_at{k}_{status}.pickle","wb") as f:
            pickle.dump(
                all_item_not_in_test,
                f, 
                pickle.HIGHEST_PROTOCOL
            )
        return all_item_not_in_test


def init_recommendation(df):
    #users with exactly k relevant items
    exactly_k = df[df.pure_test.apply(len)==k]
    recommendation = exactly_k.copy()
    recommendation["recommendation"] = recommendation.pure_test.apply(list)

    assert all(recommendation.recommendation.apply(len) == k)
    print(f"{timenow()}: All users with exactly {k} relevant items have been allocated recommendations")
    save_oracle(recommendation, f"{timenow(True)}_exact_{k}")

    #take care of users who have >k rel items
    if len(exactly_k) > 0:
        curr_item_count = get_frequency_count(recommendation.recommendation)
    else:
        curr_item_count = Counter()
    max_k = df.num_rel.max()

    for curr_k in range(k+1, max_k+1):
        user_w_k_rel_item = df.loc[df.pure_test.apply(len)==curr_k]

        if len(user_w_k_rel_item) == 0:
            continue

        #check if their items are already in the curr item count
        taken_item = user_w_k_rel_item.pure_test\
                                        .apply(list)\
                                        .apply(lambda x: [item for item in x if item in curr_item_count])
                                        
        weighted_num_taken_item = taken_item\
                                        .apply(lambda x: [curr_item_count[item] for item in x])\
                                        .apply(lambda x: sum(x) if len(x)!=0 else 0)

        user_w_k_rel_item["weighted_num_taken_item"] = weighted_num_taken_item
        user_w_k_rel_item["taken_item"] = taken_item

        #sort user by least number of items that have been used in the curr item count
        user_w_k_rel_item = user_w_k_rel_item.sort_values("weighted_num_taken_item")
        user_w_k_rel_item["recommendation"] = user_w_k_rel_item\
                                                        .apply(lambda x: [item for item in x.pure_test if item not in x.taken_item], axis=1)\
                                                        .apply(lambda x: x[:k]) #just in case there are more untaken item than required
        
        freq_count_update = get_frequency_count(user_w_k_rel_item.recommendation)

        curr_item_count.update(freq_count_update)

        for idx, row in user_w_k_rel_item.iterrows():
            if len(row.recommendation) == k:
                continue
            else:
                taken_item_count = [curr_item_count[item] for item in row.taken_item]

                num_item_to_add = k - len(row.recommendation)
                sorted_idx = np.argsort(taken_item_count, kind="stable")[:num_item_to_add]
                selected_arr = np.array(row.taken_item)[sorted_idx]
                selected = selected_arr.tolist()
                
                row.recommendation.extend(selected) #update recommendation
                assert len(row.recommendation) == k
                curr_item_count.update(Counter(selected)) #update freq count

        recommendation = pd.concat([recommendation, user_w_k_rel_item[recommendation.columns]])
        print(f"{timenow()}: All users with exactly {curr_k} relevant items have been allocated recommendations")
        save_oracle(recommendation, f"{timenow(True)}_exact_{curr_k}")

    #take care of users who have <k rel items
    lt_k = df.loc[df.pure_test.apply(len)<k].copy()

    if lt_k.shape[0] > 0:
        lt_k["recommendation"] = lt_k.pure_test.apply(list)

        #For candidate items to pad the recommendaitons, take all items but remove those that have appeared in the recommendation.
        #We need to calculate this from recommendation.recommendation instead of df_test.pure_test, because in the latter there are some discarded relevant items from users with >k rel items
        #if we exclude those discarded items, they will never appear in recommendation
        
        item_not_in_test_excl_lt_k = set_all_items.copy()
        if len(recommendation) > 0:
            recommended_items = fast_set_items(recommendation.recommendation)
            item_not_in_test_excl_lt_k = item_not_in_test_excl_lt_k - recommended_items
        
        lt_k_recommendation = fast_set_items(lt_k.recommendation)
        all_item_not_in_test = item_not_in_test_excl_lt_k - lt_k_recommendation

        #recommend unexposed items first
        
        num_lt_k = len(lt_k)

        cnt_idx = 0
        for _, row in tqdm(lt_k.iterrows()):
            while len(row.recommendation) < k and len(all_item_not_in_test) > 0:
                for item in all_item_not_in_test:
                    if len(row.recommendation) >= k:
                        break

                    if item in lt_k.loc[row.name,"train"] or item in lt_k.loc[row.name,"valid"]:
                        #item has already been interacted by that user
                        continue
                    else:
                        lt_k.loc[row.name, "recommendation"].append(item)
                        all_item_not_in_test = all_item_not_in_test - {item} #remove that item
            if cnt_idx % 100 == 0:
                save_oracle(lt_k, f"{timenow(True)}_lt_{k}")
                print(f"{timenow()}: Saved checkpoint for oracle, users with less than k items, {cnt_idx+1}/{num_lt_k}")
            cnt_idx +=1
    
        # if we ran out of unexposed items but some users don't have k items, then recommend the least frequent item currently in the recommendation list
        if len(all_item_not_in_test) == 0:
            print(f"{timenow()}: No more unexposed item")
        
        if any(lt_k.recommendation.apply(len) < k):
            print(f"{timenow()}: /!\ We ran out of unexposed item, but some users still don't have k recommended items")
            recommendation = pd.concat([recommendation, lt_k.loc[lt_k.recommendation.apply(len)==k, recommendation.columns]])
            lt_k = lt_k[lt_k.recommendation.apply(len)<k]
            user_with_no_item_to_recommend = []

            num_lt_k = len(lt_k)

            cnt_idx = 0
            for _, row in lt_k.iterrows():
                while len(row.recommendation) < k:
                #recommend least popular item in the current recommendation but not in their train-val and not in their ground truth (avail item)
                #technically, there should be no more items that are relevant to the user (because the user has <k relevant items to begin with)
                    curr_rec = pd.concat([recommendation, lt_k[recommendation.columns]])
                    least_common = get_least_pop_filtered_item(curr_rec, row.name)
                    
                    lt_k.loc[row.name, "recommendation"].append(least_common)
                if cnt_idx % 100 == 0:
                    save_oracle(lt_k, f"{timenow(True)}_lt_{k}")
                    print(f"{timenow()}: Saved checkpoint for oracle, users with less than k items, {cnt_idx+1}/{num_lt_k}")
                cnt_idx +=1
            assert len(user_with_no_item_to_recommend) == 0
        
        recommendation = pd.concat([recommendation, lt_k[recommendation.columns]])

    assert recommendation.shape[0] == df.shape[0]
    assert all(recommendation.recommendation.apply(len) == k)
    all_item_not_in_test = save_oracle(recommendation, f"{timenow(True)}_final", all_item_not_in_test)

    return recommendation, all_item_not_in_test

def sort_by_index_most_pop(df, most_pop_item):
    df.loc[:,"index_most_pop"] = df.loc[:,"recommendation"].apply(lambda x: x.index(most_pop_item) if most_pop_item in x else np.inf)
    df = df.loc[df.index_most_pop != np.inf]
    df = df.sort_values("index_most_pop", ascending=False)
    return df

def update_df_test_with_item(row, item):
    int_row_index_most_pop = int(row.index_most_pop)

    df_test.loc[row.name, "recommendation"][int_row_index_most_pop] = item
    df_test.loc[row.name, "rel"][int_row_index_most_pop] = int(item in df.loc[row.name,"pure_test"])

    #after changing relevance value above, need to reorder such that all the relevant items are in front.
    if  1 in df_test.loc[row.name, "rel"][int_row_index_most_pop+1:]:
        df_test.at[row.name, "recommendation"] = list(np.array(df_test.loc[row.name, "recommendation"])[np.argsort(df_test.loc[row.name,"rel"], kind="stable")[::-1]])
        df_test.at[row.name, "rel"] = list(np.sort(df_test.loc[row.name,"rel"],kind="stable")[::-1])

def check():
    #check that there is no intersection between the items in train_val of a user with their recommendation list
    check = pd.DataFrame()
    check["train"] = df.train
    check["valid"] = df.valid
    check["curr_test"] = df_test.recommendation.apply(set)
    check = check.dropna()
    assert check.apply(lambda x: x.train.intersection(x.curr_test), axis=1).apply(len).value_counts().shape[0] == 1
    assert check.apply(lambda x: x.valid.intersection(x.curr_test), axis=1).apply(len).value_counts().shape[0] == 1

    #ensure the k recommended items are unique
    assert all(check.curr_test.apply(len) == k) 

def oracle2fair(all_item_not_in_test, results):
    #get mostpop item
    most_pop_item_in_rec, count = get_most_pop_item(df_test.recommendation)
    new_most_pop_item_in_rec, count = get_most_pop_item(df_test.recommendation)

    #get users who got recommended the most popular item in test, prioritizing users who have these items at the back of the recommendation list first.
    df_test["index_most_pop"] = pd.Series(dtype="float64")

    selected_df = sort_by_index_most_pop(df_test, most_pop_item_in_rec)


    #recommend items that have not appeared in the recommendation set yet.
    for item in tqdm(all_item_not_in_test):
        if count == 1:
            df_test_recommendation = fast_counter(df_test.recommendation)

            assert df_test_recommendation.most_common(1)[0][1] == 1 #check that most popular item is 1
            print(f"{timenow()}: All items in the recommendation list are equally popular now")
            break

        #there is a new most pop item
        if most_pop_item_in_rec != new_most_pop_item_in_rec:
            # i = 0
            most_pop_item_in_rec = new_most_pop_item_in_rec
            selected_df = sort_by_index_most_pop(df_test, most_pop_item_in_rec)

        #for each of those users, replace the most popular item with the least popular item
        if new_most_pop_item_in_rec == most_pop_item_in_rec:

             #from those users, select only users that do not have the item in the train or val (has not interacted with the items)
            not_in_train_val = df.loc[selected_df.index, ["train","valid"]].applymap(lambda x: item not in x).all(axis=1) 
            not_in_train_val = not_in_train_val.loc[not_in_train_val]

            #technically, we should only select users that has not had that item in the current recommendation yet
            #but here the candidate replacement item is "not in recommendation"

            candidate_users = df.loc[not_in_train_val.index]
            candidate_users_find_relevant = candidate_users["pure_test"].apply(lambda x: item in x)

             #prioritise if there are candidate users that would find this item relevant (item exist in pure_ set)
            if candidate_users_find_relevant.any():
                row_idx = candidate_users[candidate_users_find_relevant].index[0]
                row = selected_df.loc[row_idx]

            else:
                row_idx = candidate_users.index[0]
                row = selected_df.loc[row_idx]

            selected_df = selected_df.drop(row.name)

            update_df_test_with_item(row, item)
            try:
                all_item_not_in_test = all_item_not_in_test-{item}  
            except:
                all_item_not_in_test.remove(item)

            #update measure count, including fairness
            update_all_measure(results)

            #get new most pop item
            new_most_pop_item_in_rec, count_most = get_most_pop_item(df_test.recommendation)
                            
        else:
            raise Exception(f'{timenow()}: Edge case encountered: debug needed')
                

    else: #we ran out of unexposed item but we have not reached max fairness yet
        #get mostpop item
        most_pop_item_in_rec, _ = get_most_pop_item(df_test.recommendation)
        new_most_pop_item_in_rec, count_most = get_most_pop_item(df_test.recommendation)

        #get leastpop item
        item, count_least = get_least_pop_item(df_test.recommendation)

        #get users who got recommended the most popular item in test, prioritizing users who have these items at the back of the recommendation list first.
        #because items at the back are less likely to be relevant, so the relevance can be maintained
        df_test["index_most_pop"] = pd.Series(dtype="float64")
        selected_df = sort_by_index_most_pop(df_test, most_pop_item_in_rec)

        m = df_test.shape[0] 
        n = num_set_all_items

        while count_most > ceil(k*m/n):

            #there is a new most pop item
            if most_pop_item_in_rec != new_most_pop_item_in_rec:
                # i=0
                most_pop_item_in_rec = new_most_pop_item_in_rec
                selected_df = sort_by_index_most_pop(df_test, most_pop_item_in_rec)

            #for each of those users, replace the most popular item with the least popular item
            if new_most_pop_item_in_rec == most_pop_item_in_rec:

                #from those users, select only users that do not have the item in the train or val (has not interacted with the items)
                not_in_train_val = df.loc[selected_df.index, ["train","valid"]].applymap(lambda x: item not in x).all(axis=1) 
                not_in_train_val = not_in_train_val.loc[not_in_train_val]

                #and further select only users that has not had that item in the current recommendation yet
                not_in_rec = df_test.loc[not_in_train_val.index, "recommendation"].apply(lambda x: item not in x)
                not_in_rec = not_in_rec.loc[not_in_rec]

                candidate_users = df.loc[not_in_rec.index]
                candidate_users_find_relevant = candidate_users["pure_test"].apply(lambda x: item in x)

                #prioritise if there are candidate users that would find this item relevant (item exist in pure_ set)
                if candidate_users_find_relevant.any():
                    row_idx = candidate_users[candidate_users_find_relevant].index[0]
                    row = selected_df.loc[row_idx]
                    
                    
                else:
                    row_idx = candidate_users.index[0]
                    row = selected_df.loc[row_idx]

                selected_df = selected_df.drop(row.name)

                update_df_test_with_item(row, item)

                #update measure count, including fairness
                update_all_measure(results)

                #get new most pop item
                new_most_pop_item_in_rec, count_most = get_most_pop_item(df_test.recommendation)
                item, _ = get_least_pop_item(df_test.recommendation)
                            
            else:
                raise Exception(f'{timenow()}: Edge case encountered: debug needed')

    check()

    save_result(results, df_test, "oraclefair")


def save_result(fair_results, df_test, status):

    with open(f"{path}/pareto_{dataset}_{status}_at{k}.pickle","wb") as f:
        pickle.dump(
            pd.DataFrame(fair_results),
            f, 
            pickle.HIGHEST_PROTOCOL
        )

    with open(f"{path}/dftest_pareto_{dataset}_{status}_at{k}.pickle","wb") as f:
        pickle.dump(
            df_test, 
            f, 
            pickle.HIGHEST_PROTOCOL
        )

def save_time(the_time_now, tracked_time, status):
    with open(f"{path}/time_pareto_{dataset}_{status}_at{k}_{the_time_now}.pickle", "wb") as f:
        pickle.dump(
            tracked_time, 
            f, 
            pickle.HIGHEST_PROTOCOL
        )






isExist = os.path.exists(path)
if not isExist:

    # Create a new directory because it does not exist
    os.makedirs(path)
    print(f"{timenow()}: Creating directory of result")


print(f"{timenow()}: Doing {dataset} at {k}")

df, df_test, test, evaluator = load_dataset(dataset)

#number of relevant item, number of users that have that amount of relevant items

print(df_test.pure_test.apply(len).value_counts().sort_index())

df_test["num_rel"] = df_test.pure_test.apply(len)


#get number of items in the dataset
set_train = fast_set_items(df.train.dropna().apply(list))
set_valid = fast_set_items(df.valid.dropna().apply(list))
set_test = fast_set_items(df_test.pure_test.dropna().apply(list))

set_all_items = set_train | set_valid | set_test
num_set_all_items = len(set_all_items)
print(f"{timenow()}: Num unique item in dataset:", num_set_all_items)

#ORACLE
start_time = time.time()
if all(df_test["pure_test"].apply(len) == k):
    df_test["recommendation"] = df_test["pure_test"].apply(list)
    #Take all items that have appeared in train_val (of all users, not just ones with enough relevant items), 
    #but remove those that have appeared in the test set of selected users.
    
    all_item_not_in_test = set_train.union(set_valid) - set_test

else:
    df_test, all_item_not_in_test = init_recommendation(df_test)
end_time = time.time()

oracle_time = end_time - start_time
print(f"{timenow()}: total time taken for oracle (initial recommendation): {oracle_time}")
save_time(timenow(True), oracle_time, "oracle")

check()

#number of items not in test
print(f"{timenow()}: Num item not in test:", len(all_item_not_in_test))

df_test["rel"] = df_test.apply(lambda x: [1 if item in x.pure_test else 0 for item in x.recommendation], axis=1)

df_test = df_test[["recommendation", "rel", "num_rel"]]

list_file = os.listdir("struct/")
file_for_dataset = [x for x in list_file if dataset in x and "Pop" in x]
assert len(file_for_dataset) == 1

with open("struct/"+file_for_dataset[0],"rb") as f:
    struct = pickle.load(f)

rec = torch.tensor(df_test.recommendation.to_list(), dtype=torch.int64)
struct.set("rec.items",rec)
struct.set("rec.topk",torch.cat([torch.Tensor(df_test.rel.to_list()), torch.Tensor(df_test.num_rel.to_list()).reshape(-1,1)], axis=1).int())

result = evaluator.evaluate(struct)
results = [result]

start_time = time.time()
#ORACLE2FAIR: change recommended item to make it more fair, using the following scenario
oracle2fair(all_item_not_in_test, results)
end_time = time.time()

oracle2fair_time = end_time - start_time 
save_time(timenow(True), oracle2fair_time, "oracle2fair")

print(f"{timenow()}: total time taken for oracle2fair (the rest of the PF): {oracle2fair_time}")