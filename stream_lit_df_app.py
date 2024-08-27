import streamlit as st
import pandas as pd
from transformers import pipeline
import numpy as np

# Set page layout to wide and browser tab name
st.set_page_config(page_title="AIPCR: Keyword Search", layout="wide")

# Load data with specified encoding
@st.cache_data
def load_data():
    df = pd.read_csv("D:/MSAI Lectures and Documents/AIPCR Project/topic_modelling_output_bart_mnli_v3.csv", encoding='ISO-8859-1')
    unnamed_columns = [col for col in df.columns if col.startswith('Unnamed:')]
    return df.drop(columns=unnamed_columns, axis=1)

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
    columns_order = ['search_relevance', 'course', 'course_name', 'course_summary', 'course_url']
    return df[columns_order]

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
    
    /* Thicker borders for DataFrame */
    table {
        border-collapse: collapse;
        width: 100%;
    }
    th, td {
        border: 10px solid black; /* Thicker border */
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
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
search_term = st.text_input('Enter a keyword:')
search_words = search_term.split()

# Perform search and display results
if search_term:
    matching_courses = search_courses_by_keyword(search_term, df)
    if not matching_courses.empty:
        st.write("Top matching courses:")
        
        # Format the course summary text
        matching_courses['course_summary'] = matching_courses['course_summary'].apply(format_text)
        
        # Apply clickable links to the course_url column
        matching_courses = make_clickable_links(matching_courses, 'course_url')
        
        # Add the search relevance column
        matching_courses = add_relevance_column(matching_courses)
        
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
            # Format the course summary text
            relevant_courses_df['course_summary'] = relevant_courses_df['course_summary'].apply(format_text)
            
            # Apply clickable links to the course_url column
            relevant_courses_df = make_clickable_links(relevant_courses_df, 'course_url')
            
            # Add the search relevance column
            relevant_courses_df = add_relevance_column(relevant_courses_df)
            
            # Reorder columns and select specific columns to display
            relevant_courses_df = reorder_columns(relevant_courses_df)
            
            # Rename columns to remove underscores and capitalize words
            relevant_courses_df.columns = [col.replace('_', ' ').title() for col in relevant_courses_df.columns]
            
            # Display the DataFrame with clickable links using st.markdown
            st.markdown(
                relevant_courses_df.to_html(escape=False, index=False),
                unsafe_allow_html=True
            )
        else:
            st.write("No relevant courses found.")
