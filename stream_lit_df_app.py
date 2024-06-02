import streamlit as st
import pandas as pd
from transformers import pipeline

# Set page layout to wide and browser tab name
st.set_page_config(page_title="AIPCR: Keyword Search", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("D:/MSAI Lectures and Documents/AIPCR Project/topic_modelling_output_bart_mnli.csv")
    unnamed_columns = [col for col in df.columns if col.startswith('Unnamed:')]
    return df.drop(columns=unnamed_columns, axis=1)

# Function to search courses by keyword
def search_courses_by_keyword(keyword, df):
    keyword = keyword.lower()
    matching_rows = df[df['keywords'].str.contains(keyword, case=False, na=False)]
    return matching_rows.sort_values(by='relevance', ascending=False)

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
        background-color: #4B0082; /* Darker maroon color */
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
search_term = st.text_input('Keyword:')

# Perform search and display results
if search_term:
    matching_courses = search_courses_by_keyword(search_term, df)
    if not matching_courses.empty:
        st.write("Top matching courses:")
        
        # Apply CSS for text wrapping in description column
        st.markdown(
            """
            <style>
            .dataframe {
                width: 100%;
            }
            .dataframe th, .dataframe td {
                white-space: pre-wrap;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        st.dataframe(matching_courses)
    else:
        st.write("No direct keyword matches, but you may be interested in:")
        
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

        if 'relevance' in relevant_courses_df.columns:
            # Sort DataFrame by 'relevance' column in descending order
            relevant_courses_df.sort_values(by='relevance', ascending=False, inplace=True)
            
        if not relevant_courses_df.empty:
            st.markdown(
                """
                <style>
                .dataframe {
                    width: 100%;
                }
                .dataframe th, .dataframe td {
                    white-space: pre-wrap;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.dataframe(relevant_courses_df)
        else:
            st.write("No relevant courses found.")
