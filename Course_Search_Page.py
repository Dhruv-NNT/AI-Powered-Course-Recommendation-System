import streamlit as st
import pandas as pd
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import math

# Set page layout to wide and browser tab name
st.set_page_config(page_title="AIPCR: Keyword Search", layout="wide")

# Load data with specified encoding
@st.cache_data
def load_data():
    df = pd.read_csv(
        "topic_modelling_output_bart_mnli_v3.csv",
        encoding='ISO-8859-1'
    )
    unnamed_columns = [col for col in df.columns if col.startswith('Unnamed:')]
    df = df.drop(columns=unnamed_columns, axis=1)
    return df

# Generate and store embeddings using LangChain's FAISS vector store
def generate_vectorstore(df):
    # Combine course_name and course_summary
    texts = (df['course_name'] + ' ' + df['course_summary']).tolist()
    
    # Initialize progress bar and status text
    my_bar = st.progress(0)
    status_text = st.empty()
    
    # Inform the user that the embedding model is being loaded
    status_text.text('Loading embedding model...')
    
    # Initialize the embedding model
    embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    
    # Update status text
    status_text.text('Generating embeddings...')
    
    # Initialize variables
    text_embeddings = []
    batch_size = 32
    total_texts = len(texts)
    total_batches = math.ceil(total_texts / batch_size)
    
    for idx, start_idx in enumerate(range(0, total_texts, batch_size)):
        batch_texts = texts[start_idx:start_idx+batch_size]
        batch_embeddings = embeddings_model.embed_documents(batch_texts)
        text_embeddings.extend(zip(batch_texts, batch_embeddings))  # Store tuples of (text, embedding)
        # Update progress bar
        my_bar.progress((idx + 1) / total_batches)
    
    # After embeddings are generated, create vectorstore
    vectorstore = FAISS.from_embeddings(text_embeddings, embeddings_model)
    
    # Remove progress bar and status text
    my_bar.empty()
    status_text.empty()
    
    return vectorstore

# Function to search courses by FAISS similarity using LangChain's FAISS vector store
def search_courses_by_faiss(query, vectorstore, df, threshold=0.5):
    # Perform similarity search with scores
    results_with_scores = vectorstore.similarity_search_with_score(query, k=10)
    
    # Filter results based on similarity threshold
    filtered_results = []
    for doc, score in results_with_scores:
        if score >= threshold:
            # Find the corresponding row in the DataFrame
            matching_row = df[df['course_name'] + ' ' + df['course_summary'] == doc.page_content]
            if not matching_row.empty:
                filtered_results.append({
                    'course_name': matching_row['course_name'].values[0],
                    'course_summary': matching_row['course_summary'].values[0],
                    'course_url': matching_row['course_url'].values[0] if 'course_url' in matching_row.columns else 'N/A',
                    'similarity_score': score
                })
    matching_courses = pd.DataFrame(filtered_results)
    return matching_courses

# Function to format course summary text for HTML display
def format_text(text):
    # Replace the non-standard bullet with a standard HTML bullet
    text = text.replace('', '•')
    
    # Split the text into lines, strip each line of extra spaces, and remove empty lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Join the cleaned lines with a single line break for HTML display
    cleaned_text = '<br>'.join(lines)
    
    # Return text wrapped in a paragraph tag
    return f"<p>{cleaned_text}</p>"

# Function to create clickable links in the course_url column
def make_clickable_links(df, url_column):
    df['course_url'] = df[url_column].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
    return df

# Function to reorder and select specific columns
def reorder_columns(df):
    columns_order = ['course_name', 'course_summary', 'course_url']
    return df[columns_order]

# Add maroon ribbon
with st.container():
    st.markdown(
        """
        <style>
        .ribbon {
            background-color: #181C62;
            color: white;
            padding: 10px;
            font-size: 24px;
            font-weight: bold;
            display: flex;
            justify-content: center; /* Center the title */
            align-items: center;
        }
        .ribbon .title {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create the ribbon layout
    st.markdown(
        '''
        <div class="ribbon">
            <span class="title">AIPCR: Course Search System(B)</span>
        </div>
        ''',
        unsafe_allow_html=True
    )

# Load and process data only if not already in session state
if 'df' not in st.session_state or 'vectorstore' not in st.session_state:
    # Load and process data
    df = load_data()
    vectorstore = generate_vectorstore(df)
    # Store in session state
    st.session_state['df'] = df
    st.session_state['vectorstore'] = vectorstore
else:
    # Retrieve from session state
    df = st.session_state['df']
    vectorstore = st.session_state['vectorstore']

# Add title
st.title('Find Relevant Courses')

# Starter courses list
starter_courses = [
    "MS7320",
    "MS7310",
    "MS7120",
    "MS7130",
    "MS7140"
]

def get_starter_courses(df):
    """Filter DataFrame to get starter courses."""
    return df[df['course'].isin(starter_courses)]

# Introductory message for new users
if 'intro_message_shown' not in st.session_state:
    st.markdown("""
        ## Welcome to the AIPCR Course Search System!
        If you're new to materials science or just getting started, we've curated some beginner-friendly courses to help you begin your journey. 
        You can search for specific topics using the search bar below or explore the recommended starter courses to get started.
    """)
    st.session_state['intro_message_shown'] = True

# Get keyword from user input
search_term = st.text_input('Enter a search term:')

# Perform search and display results
if search_term:
    # Search for courses using FAISS semantic search
    matching_courses = search_courses_by_faiss(search_term, vectorstore, df, threshold=0.5)

    if not matching_courses.empty:
        st.write("Top matching courses:")

        # Format the course summary text
        matching_courses['course_summary'] = matching_courses['course_summary'].apply(format_text)

        # Apply clickable links to the course_url column
        matching_courses = make_clickable_links(matching_courses, 'course_url')

        # Reorder columns and select specific columns to display
        matching_courses = reorder_columns(matching_courses)

        # Rename columns to remove underscores and capitalize words
        matching_courses.columns = [col.replace('_', ' ').title() for col in matching_courses.columns]

        # Display the DataFrame with clickable links using st.markdown
        st.markdown(
            matching_courses.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )
    else:
        st.write("No relevant courses found. Here are some recommended starter courses for beginners in materials science:")
        
        # Get and display starter courses
        starter_df = get_starter_courses(df)
        starter_df['course_summary'] = starter_df['course_summary'].apply(format_text)
        starter_df = make_clickable_links(starter_df, 'course_url')
        starter_df = reorder_columns(starter_df)
        starter_df.columns = [col.replace('_', ' ').title() for col in starter_df.columns]

        st.markdown(
            starter_df.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )
else:
    st.write("For those new to materials science, here are some starter courses to begin your journey:")

    # Get and display starter courses
    starter_df = get_starter_courses(df)
    starter_df['course_summary'] = starter_df['course_summary'].apply(format_text)
    starter_df = make_clickable_links(starter_df, 'course_url')
    starter_df = reorder_columns(starter_df)
    starter_df.columns = [col.replace('_', ' ').title() for col in starter_df.columns]

    st.markdown(
        starter_df.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )
