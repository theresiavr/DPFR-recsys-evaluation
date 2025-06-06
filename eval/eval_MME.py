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
    with open(f'eval/MME/log_{dataset}_{model_name}.txt', 'a+') as f:
        return builtins.print(*args, file=f, **kwargs)

for model_name in list_model:
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(now)

    print(f"Doing {dataset} - {model_name}")

    try:
        with open(f"eval/MME/base_{dataset}_{model_name}.pickle","rb") as f:
            found = pickle.load(f)
            print("found existing evaluation result ")
            print(found)
    except:
        print(f"Cannot find existing result for {dataset} - {model_name}, proceed with eval")
        config = Config(
            model=model_name, 
            dataset="new_"+dataset, 
            config_file_list=["RecBole/recbole/properties/overall.yaml"],

            config_dict={
                        "topk": [10], 
                        "metrics":[
                            "MME"
                            ]})

        evaluator = Evaluator(config)

        list_filename = [f for f in os.listdir("cluster/best_struct") if dataset in f and model_name in f]

        assert len(list_filename) == 1

        with open(f"cluster/best_struct/{list_filename[0]}","rb") as f:
            struct = pickle.load(f)

            start_time = time.time()
            result = evaluator.evaluate(struct)
            time_taken =  time.time() - start_time
            print(f"total time taken: {time_taken}")
            print(result)

            with open(f"eval/MME/time_base_{dataset}_{model_name}.pickle","wb") as f:
                pickle.dump(time_taken, f, pickle.HIGHEST_PROTOCOL)

            with open(f"eval/MME/base_{dataset}_{model_name}.pickle","wb") as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)