"""Main Streamlit App."""
import sys

sys.path.append('src')

from pathlib import Path

import pandas as pd
import streamlit as st

from sentopic import SenTopic

# ERI theme colors
COLORS = {
    "mediumblue": "#005E7B",
    "darkred": "#D0073A",
    "lightblue": "#008CA5",
    "gray": "#777777",
    "yellow": "#FBC15E",
    "darkgreen": "#307F42",
    "pink": "#FFB5B8",
    "darkblue": "#063157",
    "brightred": "#EA0D49",
    "brown": "#603534",
    "lightgreen": "#70B73F",
    "orange": "#F7941D",
}

#constants
DATE_COLUMN = 'date/time'
BASE_URL = Path.cwd()
REVIEW_DATA_PATH = BASE_URL / Path(r'data\selected_reviews.csv')
OFFERING_DATA_PATH = BASE_URL / Path(r'data\selected_offerings.csv')
HOTEL_NAMES = ['Hotel A', 'Hotel B', 'Hotel C']
EMBEDDING_MODEL = 'all-MiniLM-L12-v2'
SENTIMENT_MODEL ='lxyuan/distilbert-base-multilingual-cased-sentiments-student'
UMAP_N_NEIGHBORS_PARAMS = (4,16, 10, 2) #min, max, start, step
HDBSCAN_MIN_CLUSTERS_RARAMS = (4, 16, 8, 2) #min, max, start, step


@st.cache_data
def load_data(filepath: Path) -> pd.DataFrame:
    """Load data from csv."""
    data = pd.read_csv(filepath)
    return data

@st.cache_data
def get_review_stats(reviews: pd.DataFrame, 
                     offering_id: int) -> dict:
    """Get review aggregate statistics for a given offering_id."""
    filtered_reviews = reviews[reviews['offering_id']==offering_id]
    
    average_rating = round(float(filtered_reviews['rating'].mean()),2)
    num_ratings = filtered_reviews['rating'].shape[0]
    rating_dist = 100*filtered_reviews['rating'].value_counts().sort_index(ascending=True)/num_ratings
    remainders = 100-rating_dist
    remainders = remainders.to_list()
    rating_dist = rating_dist.to_list()
    rating_df = pd.DataFrame({'rating_dist': rating_dist, 'remainder': remainders})
    #convert value counts to percentages
    
    return {'avg_rating': average_rating, 'num_ratings': num_ratings, 'rating_dist': rating_df}

@st.cache_data
def format_hotel_data(HOTEL_NAMES: list[str], offering_lookup: dict, reviews: pd.DataFrame) -> pd.DataFrame:
    """Format hotel data for display."""
    hotel_data = []
    hotel_to_offering_id = {}
    for i in range(len(HOTEL_NAMES)):
            offering_id = offering_lookup[i]['id']
            offering_style = offering_lookup[i]['hotel_style']
            hotel_to_offering_id[HOTEL_NAMES[i][-1]] = offering_id
            review_stats = get_review_stats(reviews, offering_id)
            hotel_data.append({
                'Hotel Name': HOTEL_NAMES[i],
                'Hotel Style': offering_style,
                'Number of Reviews': review_stats['num_ratings'],
                'Average Rating': review_stats['avg_rating'],
                'Ratings': review_stats['rating_dist']['rating_dist'].to_list()
            })
    hotel_table_df = pd.DataFrame(hotel_data)
    return hotel_table_df
    

def load_from_csv(filepath: Path) -> pd.DataFrame:
    """Load processed data from csv."""
    data = pd.read_csv(filepath)
    return data

@st.cache_data
def fit_sentopic_model(df: pd.DataFrame, n_neighbors: int, min_cluster_size: int) -> SenTopic:
    """Fit SenTopic model."""
    sentopic_model = SenTopic()
    sentopic_model.umap_model.set_params(n_neighbors = n_neighbors)
    sentopic_model.hdbscan_model.set_params(min_cluster_size = min_cluster_size)

    sentopic_model.fit_transform(df['chunks'].to_list(), 
                                embeddings = df[[col for col in df.columns if col.startswith('embed_')]].to_numpy()
                                )
    return sentopic_model

# @st.cache_data
# def open_ai_topic_representation(client: OpenAI, topics: list[str]) -> list[str]:
#     """Get topic representations using OpenAI."""

#     MODEL = "gpt-4.1-nano"
#     TEMPERATURE = 0.2
#     MAX_OUTPUT_TOKENS = 100
#     TOP_P = 0.5
#     topic_string = ', '.join(topics)
#     PROMPT = f"""
#     Keywords were obtained from related hotel reviews. 
#     Determine the most likely topic described by the given keywords: {topic_string} \n\n
#     First, identify the potential subject the keywords collectively point toward by finding their common themes or context. 
#     Only after thoughtful analysis, state your inferred topic as a concise phrase.\n\n
#     **Output format:**  \n- Your answer should be a short phrase (no more than 3 words).\n
#     - Do NOT explain your reasoning or add extra details—just provide the phrase.
#     """
#     response = client.responses.create(
#     model=MODEL,
#     input=[
#         {
#         "role": "user",
#         "content": [
#             {
#             "type": "input_text",
#             "text": PROMPT
#             }
#         ]
#         },
#     ],
#     text={
#         "format": {
#         "type": "text"
#         }
#     },
#     temperature=TEMPERATURE,
#     max_output_tokens=MAX_OUTPUT_TOKENS,
#     top_p=TOP_P
#     )

#     return response.output_text

def format_topics_for_display(topics: dict[int, dict]) -> list[tuple[int, str]]:
    """Format topics for display with emojis."""
    formatted_topics = []
    for topic_id, topic_info in topics.items():
        sentiment = topic_info['sentiment']
        keywords = topic_info['keywords']
        if sentiment == 'positive':
            emoji = '✅'
        elif sentiment == 'negative':
            emoji = '❌'
        else:
            emoji = ''

        if type(topic_info['keywords']) is list:
            topic_name = f"{emoji} {'_'.join(keywords[:3]).lower()}"
        else:
            topic_name = f"{emoji} {topic_info['keywords'].title()}"

        formatted_topics.append((topic_id, topic_name))
    return formatted_topics


def format_docs_as_snippets(docs: list[str]) -> list[str]:
    """Format documents as snippets."""
    #if first character is a lowercase letter, add ellipsis at the start
    docs = ['...'+ doc if doc[0] == doc[0].lower() else doc for doc in docs]

    #if last character is not a punctuation, add ellipsis at the end
    docs = [doc + '...' if doc[-1] not in ['.', '!', '?'] else doc for doc in docs]
    formatted_docs = [f'"{doc}"' for doc in docs]
    return formatted_docs

@st.fragment 
def topic_selection_fragment(topic_names: list[str], 
                             topic_names_to_id_map: dict, 
                             topic_df: pd.DataFrame,
                             rep_docs: pd.DataFrame) -> None:
    """Topic selection fragment."""

    topic_selection = st.pills("**Select a topic to learn more**",
                                topic_names,    
                                selection_mode="single",
                                help="Topic names are generated from extracted keywords. Topics marked with ✅ are generally positive, while those with ❌ are generally negative.",
                                default = topic_names[0]
                                )
    
    #display representative documents for the selected topic
    if topic_selection:

        topic_id = topic_names_to_id_map[topic_selection]
        percent_positive = round(topic_df[topic_df['Topic']==topic_id]['positive_ratio'].max()*100)
        percent_negative = round(topic_df[topic_df['Topic']==topic_id]['negative_ratio'].max()*100) 
        num_reviews = topic_df[topic_df['Topic']==topic_id]['num_reviews'].max()
        sample_docs = rep_docs[rep_docs['Topic']==topic_id]['Document'].to_list()

        # #topic info: num related reviews, topic name, num pos/neg, sample reviews
        st.write(f'{num_reviews} guests mentioned "{topic_selection.removeprefix("✅").removeprefix("❌").strip()}" in their reviews. ',
                f':green[{percent_positive}% positive] | :red[{percent_negative}% negative]')
        formatted_docs = format_docs_as_snippets(sample_docs[:5])
        st.info('\n\n'.join(formatted_docs))


@st.dialog("Notes and References")
def show_citation_dialog() -> None:
    """Show citation dialog."""
    st.caption("""
            This demo builds off the existing work of [BERTopic](https://maartengr.github.io/BERTopic/index.html),
            and [distilbert-base-multilingual-cased-sentiments-student](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student).
            All reviews are owned by their respective authors and used here for demonstration purposes only.
            """)

def main()-> None:
    """Create Main Streamlit app."""
    st.set_page_config(page_title="Hotel Review Analysis", 
                       page_icon="🏨",
                       layout="wide")
    
    st.title('Hotel Review Analysis - Sentiment Analysis on Topics')
    with st.spinner("Loading data..."):
        reviews = load_data(REVIEW_DATA_PATH)
        offerings = load_data(OFFERING_DATA_PATH)
        offering_lookup = offerings.to_dict('index')

    # Sidebar for feature selection
    st.sidebar.header("Topic Modeling Settings")
    st.sidebar.markdown("Adjust parameters to balance local vs global topics:")
    n_neighbors = st.sidebar.slider("n_neighbors (UMAP)", 
                      min_value = UMAP_N_NEIGHBORS_PARAMS[0], 
                      max_value= UMAP_N_NEIGHBORS_PARAMS[1],
                      value = UMAP_N_NEIGHBORS_PARAMS[2],
                      step = UMAP_N_NEIGHBORS_PARAMS[3],
                      help="Lower values focus on local structure, higher values on global structure.") 
    
    min_cluster_size = st.sidebar.slider("min_cluster_size (HDBSCAN)", 
                      min_value = HDBSCAN_MIN_CLUSTERS_RARAMS[0], 
                      max_value= HDBSCAN_MIN_CLUSTERS_RARAMS[1],
                      value = HDBSCAN_MIN_CLUSTERS_RARAMS[2],
                      step = HDBSCAN_MIN_CLUSTERS_RARAMS[3],
                      help="Lower values yield more clusters, higher values yield fewer clusters.") 

    st.header('1) Select a hotel')
    st.text('Explore three different hotels with different rating distributions.')

    #create a table with hotel name, number of reviews, 
    hotel_table_df = format_hotel_data(HOTEL_NAMES, offering_lookup, reviews)

    hotel_to_offering_id = {HOTEL_NAMES[i][-1]: offering_lookup[i]['id'] for i in range(len(HOTEL_NAMES))}

    column_configuration = {
        'Ratings': st.column_config.BarChartColumn(
            "Ratings",
            help="Distribution of ratings from 1 to 5 stars",
            width="small",
            y_min = 0,
            y_max = 100,
            color = COLORS['yellow']
        )
    }
    
    st.dataframe(hotel_table_df, 
                 column_config=column_configuration,
                 width='stretch', 
                 hide_index=True)

    option = st.selectbox(
    "**Select a hotel to analyze**",
    ("Hotel A", "Hotel B", "Hotel C"))
                
    st.header('2) Explore the topics in your reviews')
    st.text("""
            Use embeddings and sentiment analysis to understand what your customer's are saying about your hotel. Use the sidebar to adjust the settings for topic modeling.
            """)

    with st.spinner("Loading data..."):
        #check if processed data exists
        print('Loading processed data...')
        final_df = load_data(BASE_URL / Path(r'data\processed_chunks.csv'))

    with st.spinner("Performing topic modeling..."):
        print('Fitting SenTopic model...')
        hotel_df = final_df[final_df['offering_ids']==hotel_to_offering_id[option[-1]]]
        model = fit_sentopic_model(hotel_df, n_neighbors, min_cluster_size)
        results = model.get_topic_info(hotel_df).reset_index()

        results['num_chunks'] = results[['positive','negative','neutral']].sum(axis=1)
        #calculate overall positive percentage for each topic
        results['positive_ratio'] = results['positive']/results['num_chunks']
        results['negative_ratio'] = results['negative']/results['num_chunks']
        
        # get representative chunks for each topic
        rep_docs = model.topic_model.get_document_info(hotel_df['chunks'].to_list())

        topics = {}
        #get topic key works from top_n_words
        for i, row in enumerate(results[results["Topic"]!= -1][['Top_n_words', 'positive_ratio', 'negative_ratio']].to_numpy().tolist()):
            top_n_words = row[0]
            if row[1] >= 0.6:
                sentiment = 'positive'
            elif row[2] >= 0.6:
                sentiment = 'negative'
            else:
                sentiment ='neutral'

            words = [word.strip() for word in top_n_words.split('-')]

            # print('Retrieving topic representations from OpenAI...')
            # response = open_ai_topic_representation(client,topics=words[:5])

            topic_id = i
            topics[topic_id] = {'keywords': words,
                                'sentiment': sentiment}
            
    #display selected parameters
    st.write(f"Topics extracted using **n_neighbors: {n_neighbors}** and **min_cluster_size: {min_cluster_size}** ")

    #clustering results: number of topics, number of chunks
    # average chunks per topic, biggest topic size, smallest topic size, number of unclustered chunks
    summary_stats = pd.DataFrame(
        {
            "Number of topics": [results["Topic"].nunique()-1], #exclude -1 topic
            "Number of chunks": [hotel_df.shape[0]],
            "Number of unclustered chunks": [results[results['Topic']==-1]['num_reviews'].max()],
            "Average chunks per topic": [round(hotel_df.shape[0]/(results['Topic'].nunique()-1),1)],
            "Biggest topic size": [results[results['Topic']!=-1]['num_reviews'].max()],
            "Smallest topic size": [results[results['Topic']!=-1]['num_reviews'].min()],
        },
    )
    st.dataframe(summary_stats, hide_index=True)

    #display topics from keywords
    topic_names = [v for k,v in format_topics_for_display(topics)]
    #reverse map topic names to topic ids
    topic_names_to_id_map = {v:k for k,v in format_topics_for_display(topics)}
    topic_selection_fragment(topic_names, topic_names_to_id_map, results, rep_docs)

    st.button(
        "&nbsp;:small[:gray[:books: Notes and References]]",
        type="tertiary",
        on_click=show_citation_dialog,
    )

if __name__ == "__main__":
    main()


