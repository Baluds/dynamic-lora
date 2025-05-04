import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from taskSpecs import TASK_SPECS

client = chromadb.PersistentClient(path="./chroma_store") 
collection = client.get_or_create_collection(name="task_embeddings",
    metadata={"hnsw:space": "cosine"}
    )

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_text(dataset_name, example, index):
    """
    Return a representative text string for embedding.
    Extend this mapping as needed.
    """
    name = dataset_name.lower()

    # --- commonsense / reasoning ---
    if name == "commonsenseqa":
        return example["question"][index]
    if name == "piqa":
        return example["goal"][index]
    if name == "copa":
        return example["premise"][index]
    if name == "cosmosqa":
        return f"{example['question'][index]} {example['context'][index]}"
    if name == "record":
        return example["query"][index]

    # --- sentiment ---
    if name in {"imdb", "yelpfull", "sentiment140"}:
        return example["text"][index]
    if name == "sst2":
        return example["sentence"][index]

    # --- reading comprehension ---
    if name == "multirc":
        return f"{example['question'][index]} {example['paragraph'][index]}"
    if name == "squadv1":
        return example["question"][index]
    if name == "boolq":
        return f"{example['question'][index]} {example['passage'][index]}"
    if name == "openbookqa":
        return example["question_stem"][index]

    # --- paraphrase ---
    if name == "paws":
        return f"{example['sentence1'][index]} || {example['sentence2'][index]}"
    if name == "qqp":
        return f"{example['question1'][index]} || {example['question2'][index]}"

    # --- NLI ---
    if name in {"rte", "cb", "mnli", "anli‑r3"}:
        return f"{example['premise'][index]} -> {example['hypothesis'][index]}"
    if name in {"wnli",}:
        return f"{example['sentence1'][index]} -> {example['sentence2'][index]}"

    # fallback: stringify the whole example
    return str(example)


def load_and_sample_data(task_specs, total_samples=2000, seed=42):
    random.seed(seed)
    datasets = []
    for spec in task_specs:
        ds = load_dataset(*spec['load_args'], split=spec.get('split', 'train'))
        datasets.append((spec['name'], ds))
    n_tasks = len(datasets)
    base_n = total_samples // n_tasks
    extra = total_samples % n_tasks
    counts = [base_n + (1 if i < extra else 0) for i in range(n_tasks)]
    counts = [min(counts[i], len(ds)) for i, (_, ds) in enumerate(datasets)]
    sampled_data = []
    for (name, ds), count in zip(datasets, counts):

        ds_shuff = ds.shuffle(seed=seed)[:count]   
        for idx, example in enumerate(ds_shuff):
            text = get_text(name, ds_shuff,idx)
            meta = {"dataset": name}
            meta["text"] = text  # store the text itself as metadata for reference
            sampled_data.append((text, meta))
    return sampled_data

sampled_examples = load_and_sample_data(TASK_SPECS, total_samples=2000, seed=42)

texts = [text for text, meta in sampled_examples]
metadatas = [meta for text, meta in sampled_examples]

embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)


ids = [f"{meta['dataset']}_{i}" for i, meta in enumerate(metadatas)]
collection.add(embeddings=embeddings.tolist(),
               metadatas=metadatas,
               ids=ids)
print(f"Indexed {collection.count()} embeddings in Chroma.")
