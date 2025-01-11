import logging
import kagglehub
import os
import polars as pl
import numpy as np
import time
import hashlib

LOGGING_FORMAT = "%(asctime)s,%(msecs)03d %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to convert the latest 'created' date from the 'versions' field to a specific format
def get_latest_time(element):
    return time.strftime("%d %b %Y", time.strptime(element[-1]['created'], "%a, %d %b %Y %H:%M:%S %Z"))

# Function to convert the 'update_date' field to a specific format
def get_latest_date(element):
    return time.strftime("%d %b %Y", time.strptime(element, "%Y-%m-%d"))

# Hashing function
def hash_row(row):
    return hashlib.md5(str(row).encode('utf-8')).hexdigest()

def main():
    # Download the data
    logger.info("Downloading the data...")
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    logger.info(f"Data downloaded at {path}.")

    # Delete old version of the data
    dirs = os.listdir("../.cache/kagglehub/datasets/Cornell-University/arxiv/versions")
    # Check if new version is uploaded
    if len(dirs) == 1:
        # Exit because there's no new versions
        logger.info("No new version uploaded.")
    # Sort the directories by file name
    dirs.sort()
    # Keep the latest version
    latest_version = dirs[-1]
    # Delete the old version
    for dir in dirs:
        if dir != latest_version:
            logger.info(f"Deleting old version {dir}...")
            os.system(f"rm -rf ../.cache/kagglehub/datasets/Cornell-University/arxiv/versions/{dir}")
            os.system(f"rm -rf ../.cache/kagglehub/datasets/Cornell-University/arxiv/{dir}.complete")
            logger.info(f"Old version {dir} deleted.")

    # Cache the data
    logger.info("Filtering data...")
    arxiv_metadata_json = f"../.cache/kagglehub/datasets/Cornell-University/arxiv/versions/{latest_version}/arxiv-metadata-oai-snapshot.json"
    arxiv_metadata = pl.read_ndjson(arxiv_metadata_json)
    cs_filtered_arxiv_metadata = arxiv_metadata.filter(pl.col("categories").str.contains(r"\b(?:cs\.(?:CV|LG|CL|AI|NE|RO))\b", strict=True))
    # Initializing a new column '_time' with default value 0
    cs_filtered_arxiv_metadata = cs_filtered_arxiv_metadata.with_columns(pl.lit(0, dtype=pl.Int64).alias('_time'))
    # Updating the column '_time' with the latest version data or update date
    cs_filtered_arxiv_metadata = cs_filtered_arxiv_metadata.with_columns(
        pl.when(cs_filtered_arxiv_metadata['versions'].is_not_null())
        .then(cs_filtered_arxiv_metadata['versions'].map_elements(get_latest_time, return_dtype=pl.Utf8))
        .otherwise(cs_filtered_arxiv_metadata['update_date'].map_elements(get_latest_date, return_dtype=pl.Utf8))
        .alias('_time')
    )
    # Columns to drop
    columns_to_drop = ['versions', 'authors_parsed', 'report-no', 'license', 'submitter']
    # Dropping the specified columns
    cs_filtered_arxiv_metadata = cs_filtered_arxiv_metadata.drop(columns_to_drop)
    # Hashing the rows
    cs_filtered_arxiv_metadata = cs_filtered_arxiv_metadata.with_columns(
        pl.struct(cs_filtered_arxiv_metadata.columns).map_elements(hash_row, return_dtype=pl.Utf8).alias('hash')
    )

    # # TEST: Randomly delete 10% of the rows
    # cs_filtered_arxiv_metadata = cs_filtered_arxiv_metadata.sample(fraction=0.95, seed=42)

    logger.info(f"Filtered data with {cs_filtered_arxiv_metadata.shape[0]} rows.")
    # Check if "cached_data.json" already exists
    if os.path.exists("cached_data.json"):
        logger.info("Checking for existing cached data...")
        # Load the existing DataFrame
        existing_data = pl.read_ndjson("cached_data.json")
        # Print file size of the existing DataFrame
        logger.info(f"Existing cached data loaded with {existing_data.shape[0]} rows.")
        # Check if there are changes in the 'hash' column
        logger.info("Checking for changes in the data...")
        # Check old vs new (to get deletions)
        deletions = existing_data.with_columns(
            existing_data['hash'].is_in(cs_filtered_arxiv_metadata['hash']).alias('is_in')
        )['is_in']
        # Deletions = false
        ids_to_delete = np.where(~deletions.to_numpy())[0].tolist()
        logger.info(f"Detected {len(ids_to_delete)} deletions.")
        # Check new vs old (to get additions, updates)
        additions = cs_filtered_arxiv_metadata.with_columns(
            cs_filtered_arxiv_metadata['hash'].is_in(existing_data['hash']).alias('is_in')
        )['is_in']
        logger.info(f"Detected {sum(~additions.to_numpy())} additions.")
        to_add = cs_filtered_arxiv_metadata.filter(~additions.to_numpy())
        # Retain = true
        existing_data = existing_data.filter(deletions.to_list())
        if len(to_add) > 0:
            logger.info(f"to_add: {to_add.shape[0]} rows \n {to_add.head(5)}")
            existing_data = pl.concat([existing_data, to_add])
        # Update the existing DataFrame
        existing_data.write_ndjson("cached_data.json")
        logger.info(f"Existing cached data updated with {existing_data.shape[0]} rows.")
        if len(ids_to_delete) > 0:
            # Export the 'ids_to_delete' to npy file
            logger.info(f"Saving {len(ids_to_delete)} deletions to ids_to_delete.npy.")
            np.save("ids_to_delete.npy", ids_to_delete)
        # Save 'to_add' DataFrame to a new NDJSON file
        if len(to_add) > 0:
            logger.info(f"Saving {to_add.shape[0]} additions to to_add.json.")
            to_add.write_ndjson("to_add.json")
    else:
        # Writing the processed DataFrame to a new NDJSON file
        logger.info("No cached data found. Writing the processed data...")
        cs_filtered_arxiv_metadata.write_ndjson("cached_data.json")
        logger.info(f"Processed data written to cached_data.json with {cs_filtered_arxiv_metadata.shape[0]} rows.")

if __name__ == '__main__':
    main()