import streamlit as st
import faiss
import yaml
import pandas as pd
from sentence_transformers import SentenceTransformer
import logging
import time
from flashrank import Ranker, RerankRequest
from bs4 import BeautifulSoup
import requests

LOGGING_FORMAT = "%(asctime)s,%(msecs)03d %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    logger.info("Loading config file...")
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    logger.info("Config file loaded.")
    return config

def load_model(config):
    logger.info(f"Initializing SentenceTransformer model with {config['embedding_model']}...")
    if "matryoshka_dim" in config and "jina" in config['embedding_model']:
        model = SentenceTransformer(config['embedding_model'],
                                    trust_remote_code=True,
                                    truncate_dim=config['matryoshka_dim'])
    else:
        model = SentenceTransformer(config['embedding_model'],
                                    trust_remote_code=True)
    if "matryoshka_dim" in config and "jina" in config['embedding_model']:
        logger.info(f"Model {config['embedding_model']} initialized with Matryoshka dimension {config['matryoshka_dim']}.")
    else:
        logger.info(f"Model {config['embedding_model']} initialized with default dimension.")
    return model

def load_index() -> faiss.Index:
    logger.info("Loading the HNSW index...")
    index = faiss.read_index("index.faiss")
    logger.info("HNSW index loaded.")
    return index

def return_task_prompt_name(config):
    if "task" in config and "jina" in config['embedding_model']:
        task = config["task"]
        prompt_name = task
    else:
        task = "text-matching"
        prompt_name = None
    logger.info(f"Task set to {task} with prompt name {prompt_name}.")
    return task, prompt_name

def return_data() -> pd.DataFrame:
    logger.info("Loading the data...")
    db = pd.read_parquet("transformed_data.parquet")
    logger.info(f"Data loaded with {db.shape[0]} rows.")
    return db

def load_resources():
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    # Load model and index only if they don't exist in session state
    if 'model' not in st.session_state:
        st.session_state.model = load_model(st.session_state.config)
    if 'index' not in st.session_state:
        st.session_state.index = load_index()
    # Retrieve task and prompt name for encoding
    if 'task' not in st.session_state:
        st.session_state.task, st.session_state.prompt_name = return_task_prompt_name(st.session_state.config)
    if 'db' not in st.session_state:
        st.session_state.db = return_data()
    # Initialize session state variables if they don't exist
    if 'query' not in st.session_state:
        st.session_state.query = ""
    if 'num_results' not in st.session_state:
        st.session_state.num_results = 15
    if 'refine' not in st.session_state:
        st.session_state.refine = False
    if 'ranker' not in st.session_state:
        st.session_state.ranker = Ranker(model_name=st.session_state.config['ranker'],
                                         max_length=8192)

load_resources()

def get_arxiv_data(arxiv_id):
    url = f"https://ar5iv.org/html/{arxiv_id}"
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        sections = soup.find_all('section')
        content = " ".join(section.text.strip() for section in sections)
        # Clean up the content
        content = content.replace('\r', ' ').replace('\t', ' ').replace('\n', ' ')
    except Exception as e:
        logger.error(f"Failed to retrieve data from {url}. Error: {e}")
        content = ""
    return content

# Set Streamlit page configuration with custom theme color
st.set_page_config(
    page_title="LitMatch",          # Set page title
    page_icon="⚡",                  # Set page icon
    layout="centered",              # Center the layout
    initial_sidebar_state="expanded",  # Sidebar state
)

def main() -> None:
    """
    Main function for running the LitMatch app, which performs academic search
    and information retrieval from the arXiv database.
    """
    # Streamlit UI setup, use #ff822d for the header color
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="display: inline; color: #ff822d; font-size: 3em;">⚡LitMatch⚡</h1>
            <p style="font-size: 16px; color: #555;">
                A <strong style="color: #ff822d;">Lit</strong>erature-<strong style="color: #ff822d;">Match</strong>ing Tool, 
                inspired by <strong style="color: #840000;">arxiv-sanity</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h3 style="text-align: center;">A Lightweight Semantic Search Tool for Related Literature in Computer Science from the arXiv Collection</h3>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align: center; font-size: 14px; color: #666;">
            <p>
                by <strong>Emmanuel Maminta</strong>, <strong>Myk Ogbinar</strong>, and <strong>Ted Peñas</strong><br>
                <em>University of the Philippines Diliman | Artificial Intelligence Program | Turing Batch</em>
            </p>
            <p>Copyright © 2024</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('---')

    # Query input and result settings
    st.session_state.query = st.text_area(
        "Enter your query here:", 
        value=st.session_state.query, 
        placeholder="e.g. Transformer-based models for NLP", 
        height=100
    )
    
    st.session_state.num_results = st.number_input(
        "Number of results to return:", 
        min_value=5, 
        max_value=30,
        value=st.session_state.num_results, 
        step=5
    )
    
    st.session_state.refine = st.checkbox("Include detailed sections from the web (experimental)", value=st.session_state.refine)

    # Trigger search when the button is pressed
    if st.button("Search"):
        if st.session_state.query.strip() == "":
            st.warning("Please enter a query to search.")
        else:
            start_time = time.perf_counter_ns()
            # Encode the query and search the index
            query_embedding = st.session_state.model.encode(
                [st.session_state.query], 
                batch_size=1, 
                show_progress_bar=False, 
                task=st.session_state.task, 
                prompt_name=st.session_state.prompt_name
            )
            distances, indices = st.session_state.index.search(query_embedding, st.session_state.num_results)

            # Retrieve results from the database and combine them with the scores
            results_df = st.session_state.db.loc[indices.squeeze(0), ['id', 'title', 'authors', 'abstract', 'date']].reset_index(drop=True)
            results_df['score'] = distances.squeeze(0)
            
            # Rerank the results (make df a list of dictionaries)
            # 2 passages: (1) title only, (2) abstract + content from ar5iv.org
            title_passages = pd.DataFrame(
                {
                    'id': results_df['id'],
                    'text': results_df['title'],
                    'meta': results_df[['title', 'authors', 'date']].to_dict(orient='records')
                }
            )
            abstract_content_passages = pd.DataFrame(
                {
                    'id': results_df['id'],
                    'text': results_df['abstract'],
                    'meta': results_df[['title', 'authors', 'date', 'abstract']].to_dict(orient='records')
                }
            )
            
            # Apply `get_arxiv_data` to each row in the DataFrame
            if st.session_state.refine:
                abstract_content_passages['text'] += '\n' + abstract_content_passages['id'].apply(get_arxiv_data)
            
            title_passages = title_passages.to_dict(orient='records')
            abstract_content_passages = abstract_content_passages.to_dict(orient='records')
            
            # Deletes the old results_df to avoid conflicts
            del results_df
            
            title_rerank_request = RerankRequest(query=st.session_state.query, passages=title_passages)
            title_results_df = st.session_state.ranker.rerank(title_rerank_request)

            abstract_content_rerank_request = RerankRequest(query=st.session_state.query, passages=abstract_content_passages)
            abstract_content_results_df = st.session_state.ranker.rerank(abstract_content_rerank_request)
            
            end_time = time.perf_counter_ns()
            elapsed_time = (end_time - start_time) / 1e6
            
            # Get the average score of the two rerankings (merge)
            title_results_df = pd.DataFrame(title_results_df)
            abstract_content_results_df = pd.DataFrame(abstract_content_results_df)
            
            results_df = pd.merge(title_results_df, abstract_content_results_df, on='id', suffixes=('_title', '_abstract_content'))
            results_df['score'] = results_df[['score_title', 'score_abstract_content']].mean(axis=1)
            # Sort the results by the average score
            results_df = results_df.sort_values(by='score', ascending=False).reset_index(drop=True)
            
            # Show only id, title, authors, and date
            meta = pd.DataFrame(results_df['meta_abstract_content'].tolist())
            
            # Combine the results with the metadata
            final_results = pd.concat([results_df[['score', 'id']], meta], axis=1)

            del results_df, meta

            st.markdown(
                f"""<p style="font-size: 14px; color: #666;">Found {final_results.shape[0]} results in {elapsed_time:.2f} ms.</p>""",
                unsafe_allow_html=True
            )
            
            # Pretty write the results (bulletin format)
            # Title by Authors (Date)
            # Abstract
            for idx, row in final_results.iterrows():
                # Format authors and handle accents
                try:
                    authors = row['authors'].encode('unicode_escape').decode('utf-8')
                except:
                    authors = row['authors'].encode('latin1').decode('unicode_escape')
                st.markdown(
                    f"""
                    <div style="padding: 10px; margin-bottom: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
                        <h3 style="color: #ff822d; margin-bottom: 5px;">{row['title']}</h3>
                        <p style="margin: 0; color: #555; font-size: 14px;"><strong>Authors:</strong> {authors} <br><strong>Date:</strong> {row['date']}</p>
                        <p style="margin-top: 10px; color: #333; font-size: 13px; line-height: 1.6;">{row['abstract']}</p>
                        <div style="margin-top: 10px;">
                            <a href="https://arxiv.org/abs/{row['id']}" target="_blank" 
                            style="color: #0066cc; text-decoration: none; font-weight: bold;">
                                Read more →
                            </a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("---")


            # Optional: Provide a message if no results are found
            if final_results.empty:
                st.info("No results found. Try modifying your query or adjust the number of results.")


if __name__ == "__main__":
    main()

# Run the app with:
# streamlit run 05_streamlit_app.py