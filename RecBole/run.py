from recbole.quick_start import run_recbole

dataset = [
    "Lastfm",
    "Amazon-lb",
    "Jester",
    "QK-video",
    "ML-10M",
    "ML-20M"
]

models = [
    "Pop",
    ]

for data in dataset:
    for model in models:      
        curr_result = run_recbole(
                                model=model,
                                dataset=data,
                                )
