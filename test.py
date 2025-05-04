import chromadb
import math
from collections import defaultdict
from sentence_transformers import SentenceTransformer

#todo maake it top 100 and also from the list calculate percentage of similarity for each dataset

def weigh_datasets(results, temp=1.0):
    """
    Turn Chroma query results into dataset-level weights.
    Uses a softmax over (‑distance/temp) within the top‑k.
    """
    sims_by_ds = defaultdict(float)

    ids         = results["ids"][0]
    metadatas   = results["metadatas"][0]
    distances   = results["distances"][0]

    #  exp(‑d/T) normalized
    exp_sim = [math.exp(d / temp) for d in distances]

    # add the distances for each dataset
    for sim, meta in zip(exp_sim, metadatas):
        ds_name = meta["dataset"]
        sims_by_ds[ds_name] += sim

    # normalize by dividing by sum
    Z = sum(sims_by_ds.values())
    weights = {ds: val / Z for ds, val in sims_by_ds.items()}
    weights_sorted = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    return weights_sorted

client = chromadb.PersistentClient(path="./chroma_store") 
collection = client.get_or_create_collection("task_embeddings")
model = SentenceTransformer("all-MiniLM-L6-v2")
query_text ="John is so short."
query_embedding = model.encode([query_text])  

results = collection.query(
    query_embeddings=query_embedding,
    n_results=100  
)

w = weigh_datasets(results, temp=0.5)
for rank, (ds, weight) in enumerate(w, 1):
    print(f"{rank:>2}. {ds:<15} {weight:.3f}")
print(sum(weight for _, weight in w))

# print(weigh_datasets(results=results,temp=0.3))
for i in range(len(results["ids"][0])):
    print(f"Rank {i+1}:")
    print("ID:      ", results["ids"][0][i])
    print("Text:    ", results["metadatas"][0][i]["text"])
    print("Dataset: ", results["metadatas"][0][i]["dataset"])
    print("Distance:", results["distances"][0][i])
    print("-" * 40)
