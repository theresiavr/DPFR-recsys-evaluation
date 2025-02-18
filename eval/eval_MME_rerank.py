import builtins
from recbole.config import Config
from recbole.evaluator.evaluator import Evaluator

import pickle


import warnings 
warnings.filterwarnings('ignore')

import os
import time

import datetime


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", default="Amazon-lb", help="Name of dataset")
args = vars(parser.parse_args())

# Set up parameters
dataset = args["dataset"]
    
list_model = [
    "BPR",
    "MultiVAE",
    "ItemKNN",
    "NCL"
]

def print(*args, **kwargs):
    with open(f'reranking/log/log_MME_{dataset}_{model_name}_{rerank}.txt', 'a+') as f:
        return builtins.print(*args, file=f, **kwargs)
    



for model_name in list_model:
    for rerank in ["borda", "combmnz", "GS-subset-0.05"]:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(now)

        print(f"Doing {dataset} - {model_name} - {rerank}")
        try:
            with open(f"reranking/MME/{dataset}_{model_name}_{rerank}_at10_rerank25.pickle","rb") as f:
                found = pickle.load(f)
                print("found existing evaluation result ")
                print(found)
        except:
            print(f"Cannot find existing result for {dataset} - {model_name}, proceed with eval")
    

            config = Config(
                    model=model_name, 
                    dataset="new_"+dataset, 
                    config_file_list=["RecBole/recbole/properties/overall.yaml"],

                    config_dict={"topk": [10], 
                                "metrics":["MME"]})

            evaluator = Evaluator(config)

            list_filename = [f for f in os.listdir("reranking/rerank_struct") if dataset in f and model_name in f and rerank in f and "rerank25" in f and "at10" in f]

            assert len(list_filename) == 1

            with open(f"reranking/rerank_struct/{list_filename[0]}","rb") as f:
                struct = pickle.load(f)

                start_time = time.time()
                result = evaluator.evaluate(struct)
                print("total time taken: ", time.time() - start_time)
                print(result)

                with open(f"reranking/MME/{dataset}_{model_name}_{rerank}_at10_rerank25.pickle","wb") as f:
                    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)