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


    
list_model = [
    "BPR",
    "MultiVAE",
    "ItemKNN",
    "NCL"
]

list_dataset = ["Amazon-lb", "Lastfm", "QK-video", "Jester", "ML-10M", "ML-20M"]

list_measure = ["IBO", "AIF", "IIF", "IAArerank"]

def print(*args, **kwargs):
    with open(f'reranking/log/log_joint_{dataset}_{model_name}_{rerank}.txt', 'a+') as f:
        return builtins.print(*args, file=f, **kwargs)
    


for dataset in list_dataset:
    for model_name in list_model:
        for rerank in ["borda", "combmnz", "GS-subset-0.05"]:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(now)

            print(f"Doing {dataset} - {model_name} - {rerank}")
            try:
                with open(f"reranking/joint/{measure}_{dataset}_{model_name}_{rerank}_at10_rerank25.pickle","rb") as f:
                    found = pickle.load(f)
                    print("found existing evaluation result ")
                    print(found)
            except:
                print(f"Cannot find existing result for {dataset} - {model_name}, proceed with eval")
        


                list_filename = [f for f in os.listdir("reranking/rerank_struct") if dataset in f and model_name in f and rerank in f and "rerank25" in f and "at10" in f]

                assert len(list_filename) == 1

                with open(f"reranking/rerank_struct/{list_filename[0]}","rb") as f:
                    struct = pickle.load(f)

                    for measure in list_measure:
                        print(f"Doing {measure}")
                        config = Config(
                            model=model_name, 
                            dataset="new_"+dataset, 
                            config_file_list=["RecBole/recbole/properties/overall.yaml"],

                            config_dict={
                                        "topk": [10], 
                                        "metrics":[
                                            measure
                                            ]})

                        evaluator = Evaluator(config)

                        start_time = time.time()
                        result = evaluator.evaluate(struct)
                        time_taken = time.time() - start_time
                        print(f"total time taken: {time_taken}")
                        print(result)

                        
                        with open(f"reranking/joint/time_{measure}_{dataset}_{model_name}_{rerank}_at10_rerank25.pickle","wb") as f:
                            pickle.dump(time_taken, f, pickle.HIGHEST_PROTOCOL)

                        with open(f"reranking/joint/{measure}_{dataset}_{model_name}_{rerank}_at10_rerank25.pickle","wb") as f:
                            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)