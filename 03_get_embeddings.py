from sentence_transformers import SentenceTransformer
import yaml
from typing import Dict, Any
import pandas as pd
import polars as pl
import numpy as np
import os
import logging
import time

LOGGING_FORMAT = "%(asctime)s,%(msecs)03d %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def load_config():
    logger.info("Loading config file...")
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    logger.info("Config file loaded.")
    return config

def revise_embeddings() -> None:
    logger.info("Revising the embeddings...")
    if not os.path.exists("ids_to_delete.npy"):
        logger.info("No ids_to_delete.npy file found. No rows to delete from the embeddings. Skipping the deletion of rows from the embeddings.")
        return
    ids_to_delete = np.load("ids_to_delete.npy").tolist()
    logger.info(f"Deleting {len(ids_to_delete)} rows from the embeddings...")
    embeddings = np.load("embeddings.npy")
    embeddings = np.delete(embeddings, ids_to_delete, axis=0)
    logger.info(f"Rows deleted. Revised embeddings with {embeddings.shape[0]} rows.")
    os.remove("ids_to_delete.npy")
    logger.info("Saving the revised embeddings...")
    np.save("embeddings.npy", embeddings)
    logger.info("Revised embeddings saved as embeddings.npy.")

def get_embeddings(config: Dict[str, Any]) -> None:
    if not os.path.exists("embeddings.npy") or os.path.exists("to_add.parquet"):
        # Load the embedding model
        logger.info(f"Initializing SentenceTransformer model with {config['embedding_model']}...")
        if "matryoshka_dim" in config or "jina" in config['embedding_model']:
            model = SentenceTransformer(config['embedding_model'],
                                        trust_remote_code=True,
                                        truncate_dim=config['matryoshka_dim'])
        else:
            model = SentenceTransformer(config['embedding_model'],
                                        trust_remote_code=True)
        if "matryoshka_dim" in config or "jina" in config['embedding_model']:
            logger.info(f"Model {config['embedding_model']} initialized with Matryoshka dimension {config['matryoshka_dim']}.")
        else:
            logger.info(f"Model {config['embedding_model']} initialized with default dimension.")

        if "task" in config and "jina" in config['embedding_model']:
            task = config["task"]
            prompt_name = task
        else:
            task = "text-matching"
            prompt_name = None
        logger.info(f"Task set to {task} with prompt name {prompt_name}.")

    if not os.path.exists("embeddings.npy"):
        # Reading the data
        logger.info("Reading the data...")
        data = pd.read_parquet("transformed_data.parquet")
        # Check if 'title_abstract' column exists
        if 'title_abstract' not in data.columns:
            logger.error("Column 'title_abstract' not found in the data.")
            return
        logger.info(f"Title-abstracts data loaded with {data.shape[0]} rows.")

        logger.info("No embeddings.npy file found. Encoding the title-abstracts...")
        start_time = time.perf_counter()
        title_embeddings = model.encode(data['title'].values.tolist(), batch_size=512, task=task, prompt_name=prompt_name)
        abstract_embeddings = model.encode(data['abstract'].values.tolist(), batch_size=512, task=task, prompt_name=prompt_name)
        # Create a weighted average of the embeddings
        # Put the title with a weight of 0.5, abstract with 0.5
        embeddings = (title_embeddings + abstract_embeddings) / 2
        end_time = time.perf_counter()
        logger.info(f"Title-abstracts encoded with dimensions {embeddings.shape} in {((end_time - start_time) / 60):.2f} minutes.")

        # Save the embeddings as npy file
        logger.info("Saving the embeddings...")
        np.save("embeddings.npy", embeddings)
        logger.info("Embeddings saved as embeddings.npy.")
    else:
        if os.path.exists("to_add.parquet"):
            to_add = pd.read_parquet("to_add.parquet")
            os.remove("to_add.parquet")
            logger.info(f"Encoding {to_add.shape[0]} new title-abstracts...")
            start_time = time.perf_counter()
            embeddings = np.load("embeddings.npy")
            new_embeddings = model.encode(to_add['title_abstract'].values.tolist(), batch_size=512, task=task, prompt_name=prompt_name)
            end_time = time.perf_counter()
            embeddings = np.vstack((embeddings, new_embeddings))
            logger.info(f"New Title-abstracts encoded with dimensions {new_embeddings.shape} in {((end_time - start_time) / 60):.2f} minutes.")
            logger.info(f"Revised embeddings with dimensions {embeddings.shape}.")
            logger.info("Saving the revised embeddings...")
            np.save("embeddings.npy", embeddings)
            logger.info("Revised embeddings saved as embeddings.npy.")
        else:
            logger.info("No new title-abstracts found. Embeddings already exist.")

if __name__ == '__main__':
    config = load_config()
    revise_embeddings()
    get_embeddings(config)
