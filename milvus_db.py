from pymilvus import MilvusClient
import pandas as pd
from pymilvus import DataType, FieldSchema, CollectionSchema
import os


def create_df():
    all_data = []
    for file_name in os.listdir("data"):
        if file_name.endswith(".csv"):
            data = pd.read_csv(os.path.join("data", file_name))
            data.columns = data.columns.str.strip().str.lower()
            all_data.append(data)
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def init_db():
    client = MilvusClient("voice_recognition.db")

    if client.has_collection(collection_name="voice_collection"):
        client.drop_collection(collection_name="voice_collection")
    client.create_collection(
        collection_name="voice_collection",
        dimension=21,  # The vectors we will use in this demo has 21 dimensions
    )

    return client


client = init_db()
df = create_df()

# Define the schema for the collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=21),
    FieldSchema(name="label", dtype=DataType.STRING),
]
schema = CollectionSchema(fields, "Voice recognition collection")

labels = df[["label"]]
vectors = df.drop(columns=["label"])


data = [
    {"id": i, "vector": vectors.iloc[i].tolist(), "label": labels.iloc[i].tolist()}
    for i in range(len(vectors))
]

# Insert data into the collection
client.insert("voice_collection", data)
