import streamlit as st
import pandas as pd
from transformers import pipeline
import numpy as np

# Set page layout to wide and browser tab name
st.set_page_config(page_title="AIPCR: Keyword Search", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("D:/MSAI Lectures and Documents/AIPCR Project/topic_modelling_output_bart_mnli_v2.csv")
    unnamed_columns = [col for col in df.columns if col.startswith('Unnamed:')]
    return df.drop(columns=unnamed_columns, axis=1)

# Function to create clickable links in the course_url column
def make_clickable_links(df, url_column):
    df['course_url'] = df[url_column].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
    return df

# Function to add a search relevance column
def add_relevance_column(df):
    df['search_relevance'] = range(1, len(df) + 1)
    return df

# Function to search courses by keyword
def search_courses_by_keyword(keyword, df):
    keyword = keyword.lower()
    matching_rows = df[df['keywords'].str.contains(keyword, case=False, na=False)]
    sorted_rows = matching_rows.sort_values(by='relevance', ascending=False)
    return sorted_rows

# Function to reorder and select specific columns
def reorder_columns(df):
    columns_order = ['course', 'course_name', 'course_summary', 'course_url', 'relevance', 'search_relevance']
    return df[columns_order]

# Function to calculate DCG
def dcg_at_k(relevances, k):
    relevances = np.asfarray(relevances)[:k]
    if relevances.size:
        return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))
    return 0.0

# Function to calculate nDCG
def ndcg_at_k(relevances, k):
    dcg_max = dcg_at_k(sorted(relevances, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(relevances, k) / dcg_max

# Function to calculate Average Precision
def average_precision(relevances, threshold=4.0):
    relevances = np.asarray(relevances) >= threshold
    out = [precision_at_k(relevances, k + 1) for k in range(relevances.size) if relevances[k]]
    if not out:
        return 0.0
    return np.mean(out)

# Function to calculate Precision at K
def precision_at_k(relevances, k):
    relevances = np.asarray(relevances)[:k]
    return np.mean(relevances)

# Function to calculate MAP
def mean_average_precision(relevances_list, threshold=4.0):
    return np.mean([average_precision(relevances, threshold) for relevances in relevances_list])

# Load data
df = load_data()

# Initialize the zero-shot classification pipeline lazily
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Add maroon ribbon
st.markdown(
    """
    <style>
    .ribbon {
        background-color: #181C62; /* Darker maroon color */
        color: white;
        padding: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    <div class="ribbon">
        AIPCR: Course Search
    </div>
    """,
    unsafe_allow_html=True
)

# Add title
st.title('Find Relevant Courses')

# Add a prompt
st.subheader('Enter a keyword to search for relevant courses')

# Get keyword from user input
search_term = st.text_input('')
search_words = search_term.split()

# Perform search and display results
if search_term:
    print(f"Search term: {search_term}")  # Debugging print
    matching_courses = search_courses_by_keyword(search_term, df)
    if not matching_courses.empty:
        print("Primary search found results.")  # Debugging print
        st.write("Top matching courses:")
        
        # Apply clickable links to the course_url column
        matching_courses = make_clickable_links(matching_courses, 'course_url')
        
        # Add the search relevance column
        matching_courses = add_relevance_column(matching_courses)
        
        # Reorder columns and select specific columns to display
        matching_courses = reorder_columns(matching_courses)
        
        # Calculate nDCG and MAP for primary search
        primary_relevances = matching_courses['relevance'].tolist()
        ndcg_primary = ndcg_at_k(primary_relevances, len(primary_relevances))
        map_primary = mean_average_precision([primary_relevances])
        
        # Print or log nDCG and MAP (these won't appear on the Streamlit app)
        print(f"Primary Search nDCG: {ndcg_primary}, MAP: {map_primary}")
        
        # Display the DataFrame with clickable links using st.markdown
        st.markdown(
            matching_courses.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )
    else:
        print("Primary search found no results, triggering fallback search.")  # Debugging print
        st.write("No direct keyword matches, attempting fallback search.")
        
        classifier = load_classifier()
        
        # List to store relevant courses
        relevant_courses = []
            
        # Iterate over each row in the dataframe
        for index, row in df.iterrows():
            # Get course description from the row
            keywords = row['keywords']
            
            # Check if course description is valid
            if keywords and isinstance(keywords, str):
                # Compute similarity score between search term and course description
                similarity_score = classifier(sequences=[search_term], candidate_labels=[keywords])
                score = similarity_score[0]['scores'][0]  # Access the first item in the list, then access 'scores'
                    
                # Check if similarity score is above threshold
                if score > 0.50:
                    # Append the row along with the score to the relevant courses list
                    relevant_courses.append((row, score))
        
        # Sort relevant courses by relevance score in descending order
        relevant_courses.sort(key=lambda x: x[1], reverse=True)
        
        # Extract rows from the sorted relevant courses
        sorted_relevant_courses = [row for row, _ in relevant_courses]
            
        # Convert list of relevant courses to DataFrame
        relevant_courses_df = pd.DataFrame(sorted_relevant_courses)

        if not relevant_courses_df.empty:
            print("Fallback search found results.")  # Debugging print
            # Calculate nDCG and MAP for fallback search
            fallback_relevances = relevant_courses_df['relevance'].tolist()
            ndcg_fallback = ndcg_at_k(fallback_relevances, len(fallback_relevances))
            map_fallback = mean_average_precision([fallback_relevances])
            
            # Print or log nDCG and MAP (these won't appear on the Streamlit app)
            print(f"Fallback Search nDCG: {ndcg_fallback}, MAP: {map_fallback}")
            
            # Apply clickable links to the course_url column
            relevant_courses_df = make_clickable_links(relevant_courses_df, 'course_url')
            
            # Add the search relevance column
            relevant_courses_df = add_relevance_column(relevant_courses_df)
            
            # Reorder columns and select specific columns to display
            relevant_courses_df = reorder_columns(relevant_courses_df)
            
            # Display the DataFrame with clickable links using st.markdown
            st.markdown(
                relevant_courses_df.to_html(escape=False, index=False),
                unsafe_allow_html=True
            )
        else:
            st.write("No relevant courses found.")
