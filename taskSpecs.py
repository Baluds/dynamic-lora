TASK_SPECS = [
    # Commonsense / reasoning
    {"name": "CommonsenseQA",   "load_args": ("tau/commonsense_qa",),                 "split": "train"},
    {"name": "PIQA",            "load_args": ("piqa",),                               "split": "train"},
    {"name": "COPA",            "load_args": ("super_glue", "copa"),                  "split": "train"},
    {"name": "CosmosQA",        "load_args": ("cosmos_qa",),                          "split": "train"},
    {"name": "record",          "load_args": ("aps/super_glue", "record"),                             "split": "train"},
    # Sentiment
    {"name": "IMDB",            "load_args": ("imdb",),                               "split": "train"},
    {"name": "SST2",            "load_args": ("glue", "sst2"),                        "split": "train"},
    {"name": "YelpFull",        "load_args": ("yelp_review_full",),                   "split": "train"},
    {"name": "Sentiment140",    "load_args": ("sentiment140",),                       "split": "train"},
    # Reading‑comprehension
    {"name": "MultiRC",         "load_args": ("super_glue", "multirc"),               "split": "train"},
    {"name": "SQuADv1",         "load_args": ("squad",),                              "split": "train"},
    {"name": "BoolQ",           "load_args": ("super_glue", "boolq"),                 "split": "train"},
    {"name": "OpenBookQA",      "load_args": ("openbookqa", "main"),                  "split": "train"},
    # Paraphrase
    {"name": "PAWS",            "load_args": ("paws", "labeled_final"),               "split": "train"},
    {"name": "QQP",             "load_args": ("glue", "qqp"),                         "split": "train"},
    # NLI
    {"name": "RTE",             "load_args": ("super_glue", "rte"),                   "split": "train"},
    {"name": "CB",              "load_args": ("super_glue", "cb"),                    "split": "train"},
    {"name": "MNLI",            "load_args": ("glue", "mnli"),                        "split": "train"},
    {"name": "ANLI‑R3",         "load_args": ("anli",),                               "split": "train_r3"},
    {"name": "WNLI",            "load_args": ("nyu-mll/glue", "wnli"),                 "split": "train"}
]