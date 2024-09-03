import streamlit as st
import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch

# Set page layout to wide and browser tab name
st.set_page_config(page_title="AIPCR: Keyword Search", layout="wide")

# Load transformer model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Function to get embedding for a text
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize embeddings
    return normalized_embeddings.squeeze().numpy()

# Load data with specified encoding
@st.cache_data
def load_data():
    df = pd.read_csv("topic_modelling_output_bart_mnli_v3.csv", encoding='ISO-8859-1')
    unnamed_columns = [col for col in df.columns if col.startswith('Unnamed:')]
    df = df.drop(columns=unnamed_columns, axis=1)
    return df

# Generate and store normalized embeddings for each course
@st.cache_data
def generate_course_embeddings(df):
    df['embedding'] = df['course_summary'].apply(get_embedding)
    embeddings = np.stack(df['embedding'].values)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Use IndexFlatIP for inner product (cosine similarity)
    index.add(embeddings)  # Add normalized embeddings to the index
    return df, index

# Function to search courses by FAISS similarity using cosine similarity
def search_courses_by_faiss(query, index, df):
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, 10)
    matching_rows = df.iloc[indices.flatten()].copy()
    matching_rows['faiss_score'] = distances.flatten()  # Distances are now effectively cosine similarities
    
    # Apply the threshold: keep only rows with faiss_score above a certain threshold
    threshold = 0.2
    matching_rows = matching_rows[matching_rows['faiss_score'] >= threshold]

    # Sort by faiss_score (cosine similarity)
    matching_rows = matching_rows.sort_values(by=['relevance','faiss_score'], ascending=False)
    
    return matching_rows

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
    columns_order = ['faiss_score', 'course', 'course_name', 'course_summary', 'course_url']
    return df[columns_order]

# Load and process data
df = load_data()
df, faiss_index = generate_course_embeddings(df)

# Add maroon ribbon
st.markdown(
    """
    <style>
    .ribbon {
        background-color: #181C62;
        color: white;
        padding: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    table {
        border-collapse: collapse;
        width: 100%;
    }
    th, td {
        border: 2px solid black;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
    }
    </style>
    <div class="ribbon">
        AIPCR: Course Search (B)
    </div>
    """,
    unsafe_allow_html=True
)

# Add title
st.title('Find Relevant Courses')

# Add a prompt
st.subheader('Enter a keyword to search for relevant courses')

# Get keyword from user input
search_term = st.text_input('Enter a keyword:')

# Perform search and display results
if search_term:
    # Search for courses using FAISS semantic search
    matching_courses = search_courses_by_faiss(search_term, faiss_index, df)
    
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
        st.write("No relevant courses found.")
