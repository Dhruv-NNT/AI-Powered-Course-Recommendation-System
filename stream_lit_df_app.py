import streamlit as st
import pandas as pd

# Set page layout to wide and browser tab name
st.set_page_config(page_title="AIPCR: Keyword Search", layout="wide")

# Load data
@st.cache_resource
def load_data():
    return pd.read_csv("Topic_Modelling_Predictions.csv")

# Function to search courses by keyword
def search_courses_by_keyword(keyword, df):
    keyword = keyword.lower()
    matching_rows = df[df['description'].str.contains(keyword, case=False, na=False)]
    matching_rows = matching_rows.sort_values(by='relevance', ascending=False)
    return matching_rows

# Load data
df = load_data()

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
        st.write("No matching courses found.")
