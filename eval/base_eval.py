import builtins
from recbole.config import Config
from recbole.evaluator.evaluator import Evaluator

import pickle


import warnings 
warnings.filterwarnings('ignore')

import os
import time

import datetime



def print(*args, **kwargs):
    with open(f'eval/base/log_{dataset}_{model_name}.txt', 'a+') as f:
        return builtins.print(*args, file=f, **kwargs)
    

list_dataset = [
                "Amazon-lb", 
                "Jester", 
                "Lastfm", 
                "QK-video",
                "ML-10M", 
                "ML-20M"
                ]
list_model = [
    "BPR",
    "MultiVAE",
    "ItemKNN",
    "NCL"
]

for dataset in list_dataset:

    for model_name in list_model:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(now)

        print(f"Doing {dataset} - {model_name}")

        try:
            with open(f"eval/base/base_{dataset}_{model_name}.pickle","rb") as f:
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
                                    "RelMetrics",
                                    "FairWORel",
                                    "IAA",
                                    "MME",
                                    "IIF_AIF",
                                    "IBO"
                                ]})

            evaluator = Evaluator(config)

            list_filename = [f for f in os.listdir("cluster/best_struct") if dataset in f and model_name in f]

            assert len(list_filename) == 1

            with open(f"cluster/best_struct/{list_filename[0]}","rb") as f:
                struct = pickle.load(f)

                start_time = time.time()
                result = evaluator.evaluate(struct)
                print("total time taken: ", time.time() - start_time)
                print(result)

                with open(f"eval/base/base_{dataset}_{model_name}.pickle","wb") as f:
                    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)