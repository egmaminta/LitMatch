import logging
import polars as pl
import os

LOGGING_FORMAT = "%(asctime)s,%(msecs)03d %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

def transform_data():
    logger.info("Loading the existing cached data...")
    cached_df = pl.read_ndjson("cached_data.json")
    export_df = pl.DataFrame(
        {
            'id': cached_df['id'],
            'doi': cached_df['doi'],
            'authors': cached_df['authors'],
            'title': cached_df['title'],
            'abstract': cached_df['abstract'],
            'date': cached_df['_time'],
            'title_abstract': cached_df['title']+ "\n" + cached_df['authors'] + "\n\n" + cached_df['abstract'],
            'hash': cached_df['hash']
        }
    )
    export_df = export_df.to_pandas()
    logger.info("Transformed data ready.")
    export_df.to_parquet("transformed_data.parquet")
    logger.info("Transformed data saved as transformed_data.parquet.")

    if os.path.exists("to_add.json"):
        logger.info("Loading new data (data not found in existing cached data)...")
        to_add = pl.read_ndjson("to_add.json")
        to_add_df = pl.DataFrame(
            {
                'id': to_add['id'],
                'doi': to_add['doi'],
                'authors': to_add['authors'],
                'title': to_add['title'],
                'abstract': to_add['abstract'],
                'date': to_add['_time'],
                'title_abstract': to_add['title'] + "\n" + to_add['authors'] + "\n\n" + to_add['abstract'],
                'hash': to_add['hash']
            }
        )
        to_add_df = to_add_df.to_pandas()
        # Delete the file after loading
        os.remove("to_add.json")
        logger.info("New data loaded.")
        logger.info("Saving new data as to_add.parquet...")
        to_add_df.to_parquet("to_add.parquet")
        logger.info("New data saved as to_add.parquet.")

if __name__ == "__main__":
    transform_data()