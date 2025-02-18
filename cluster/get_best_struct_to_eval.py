import os
import pickle
import shutil
import pandas as pd
import torch


folder = "cluster/bestparam/"

def load_config(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    if torch.cuda.is_available():
        checkpoint = torch.load(model_file)
    else:
        checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    config = checkpoint['config']
    return config

def get_param_from_file(path):
    with open(path, "rb") as f:
        try:
            bestparam = pickle.load(f)
        except:
            f.seek(0)
            bestparam  = pickle.load(f)
    
    return bestparam


def get_model_file(dataset, model):

    #get param files
    files = sorted([f for f in os.listdir(folder) if "param" in f and dataset in f and model in f])
	
    if len(files) > 1:
        print("Found more than 1 file")
        best_file = ""
        max_score = -1
        for file in files:
            result = pd.read_pickle(folder+file.replace("param_","result_"))
            curr_score = result["best_valid_score"]

            if curr_score >= max_score: #take new one if there are two best scores
                best_file = file
                max_score = curr_score
        final_file = best_file

    else:
        final_file = files[0]
    
    bestparam = get_param_from_file(f"{folder}/{final_file}")

    #find which file
    #reverse to take the newest model
    candidate = reversed(sorted([f for f in os.listdir("cluster/saved") if model in f and dataset in f])) 

    for model_file in candidate:
        config  = load_config("cluster/saved/"+model_file)
        to_check = [config.final_config_dict[key] for key, _ in bestparam.items()]
        bestparam_val = list(bestparam.values())
        bestparam_val = [x if type(x)!=str else eval(x) for x in bestparam_val]
        if to_check == bestparam_val:
            print(f"found {dataset} - {model}")
            found = model_file
            shutil.copy(f"cluster/struct/struct_{model_file}", f"cluster/best_struct/struct_{model_file}")
            break
    return found


list_dataset = [
	        "Amazon-lb",
	        "Lastfm",
            "Jester",
	        "QK-video",
	        "ML-10M",
	        "ML-20M"
            ]

list_model = [
            "ItemKNN", "BPR", "MultiVAE", 
              "NCL"
              ]

for dataset in list_dataset:
    for model in list_model:
        print(f"Finding {dataset} - {model}")
        try:
            with open(f"cluster/results/filename_best_for_{model}_{dataset}.pickle","rb") as f:
                found = pickle.load(f)
                print("found existing best file")
                print(found)
        except:
            print("no existing file found")
            found = get_model_file(dataset, model)
            with open(f"cluster/results/filename_best_for_{model}_{dataset}.pickle","wb") as f:
                print("Saving new best file")
                pickle.dump(found, f, pickle.HIGHEST_PROTOCOL)
                print(found)