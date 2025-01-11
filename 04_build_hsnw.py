import faiss
import numpy as np
import logging
import yaml
from typing import Dict, Any

LOGGING_FORMAT = "%(asctime)s,%(msecs)03d %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    logger.info("Loading config file...")
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    logger.info("Config file loaded.")
    return config

def build_hsnw(config: Dict[str, Any]) -> None:
    logger.info("Building the HNSW index...")
    embeddings = np.load("embeddings.npy")
    index = faiss.IndexHNSWFlat(embeddings.shape[1], config['faiss_M'])
    # Use efConstruction here
    index.hnsw.efConstruction = config['faiss_efConstruction']
    # Build graph here
    index.add(embeddings)
    # Use efSearch here
    index.hnsw.efSearch = config['faiss_efSearch']
    logger.info(f"HNSW entry point: {index.hnsw.entry_point}")
    bincounts = np.bincount(faiss.vector_to_array(index.hnsw.levels)).tolist()
    distr_nodes = {f"Level {i}": bincounts[i] for i in range(len(bincounts))}
    logger.info(f"HNSW distribution: {distr_nodes}")
    logger.info("Saving the HNSW index...")
    faiss.write_index(index, "index.faiss")
    logger.info("HNSW index saved as index.faiss.")

if __name__ == "__main__":
    config = load_config()
    build_hsnw(config)