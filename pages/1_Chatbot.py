import streamlit as st
import os
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers.tokenization_utils_base')

# LangChain imports
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import Tool, initialize_agent
from langchain.callbacks.base import BaseCallbackHandler

# Set your Tavily and Groq API keys using environment variables
os.environ['TAVILY_API_KEY'] = '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'  # Replace with your Tavily API key
os.environ['GROQ_API_KEY'] = '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'      # Replace with your Groq API key

def create_chain(vectorstore):
    # Initialize the LLM (ChatGroq)
    llm_chain = st.session_state['llm']

    # Use the vector store to create a retriever
    retriever = vectorstore.as_retriever()

    # Initialize conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Set up the conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_chain,
        retriever=retriever,
        memory=memory,
    )

    return chain

def evaluator_agent(response, llm):
    evaluation_prompt = f"""
    As an evaluator, ensure that the following response should STRICTLY adhere to the guidelines:
    - The response should be educational (within the context of higher education such as master's or undergraduate-level courses), ethical, and promote learning.
    - It should not entertain inappropriate content, lewd references, or bad actors.
    - The response should encourage and facilitate learning in a safe and supportive manner.

    Response to evaluate:
    "{response}"

    Does the response meet these guidelines? Answer 'Yes' or 'No' and provide a brief justification.
    """
    evaluation_result = llm.predict(evaluation_prompt)
    if 'Yes' in evaluation_result:
        return True, response  # Response is appropriate
    else:
        return False, evaluation_result  # Response is not appropriate; return evaluation result

def classifier_router(question):
    """Router Function using a classifier and cosine similarity."""
    classifier = st.session_state['classifier']
    embedder = st.session_state['embedder']
    course_embeddings = st.session_state['course_embeddings']

    # Use cosine similarity with course embeddings
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    similarities = util.cos_sim(question_embedding, course_embeddings)
    max_similarity_score = similarities.max().item()
    similarity_threshold = 0.5  # Adjust as needed

    if max_similarity_score >= similarity_threshold:
        return 'vectorstore'
    else:
        # Use zero-shot classifier to check if the question is course-related
        candidate_labels = ['course-related', 'other']
        result = classifier(question, candidate_labels, hypothesis_template="This query is {}.")
        confidence_threshold = 0.6  # Adjust as needed
        if result['labels'][0] == 'course-related' and result['scores'][0] > confidence_threshold:
            return 'vectorstore'
        else:
            # Check if the query is educational but not course-related
            candidate_labels = ['educational', 'non-educational']
            result = classifier(question, candidate_labels, hypothesis_template="This query is {}.")
            if result['labels'][0] == 'educational':
                return 'web_search'
            else:
                return 'reject_query'

def web_search_tool(query):
    """Web Search Tool: Perform a web search using Tavily API."""
    # Placeholder for actual Tavily API integration
    results = f"Search results for '{query}' from Tavily (educational content related to materials science and engineering)."
    return results

def safe_groq_call(llm, inputs):
    try:
        return llm.predict(inputs)
    except Exception as e:
        return "The service is temporarily unavailable. Please try again later."

def create_primary_agent():
    # Create a retrieval tool using the conversation chain
    def recommend_courses(query):
        response = st.session_state['conversation_chain']({"question": query})
        return response["answer"]

    course_recommendation_tool = Tool(
        name="CourseRecommendationTool",
        func=recommend_courses,
        description="Provides course recommendations based on the user's query."
    )

    # Initialize the agent using initialize_agent
    primary_agent = initialize_agent(
        tools=[course_recommendation_tool],
        llm=st.session_state['llm'],
        agent="zero-shot-react-description",
        handle_parsing_errors=True,
        verbose=True,
    )

    return primary_agent

def create_augmentation_agent():
    # Create a tool that performs a web search and processes the results
    def augment_response(query):
        search_results = web_search_tool(query)
        augmented_response = st.session_state['llm'].predict(
            f"Based on the following search results, provide additional learning resources which the user can go through. "
            f"Tell the user explicitely about the additional resources retrieved outside the course database, begin your answer as 'Additionally, here are a few external learning resources you can explore:'\n\n{search_results}")
        return augmented_response

    augmentation_tool = Tool(
        name="AugmentationTool",
        func=augment_response,
        description="Enhances the response with additional learning resources."
    )

    # Initialize the agent using initialize_agent
    augmentation_agent = initialize_agent(
        tools=[augmentation_tool],
        llm=st.session_state['llm'],
        agent="zero-shot-react-description",
        handle_parsing_errors=True,
        verbose=True,
    )

    return augmentation_agent

def create_web_search_agent():
    # Create a tool that performs a web search
    def web_search(query):
        search_results = web_search_tool(query)
        return search_results

    web_search_tool_agent = Tool(
        name="WebSearchTool",
        func=web_search,
        description="Performs a web search to find information related to the user's query."
    )

    # Initialize the agent using initialize_agent
    web_search_agent = initialize_agent(
        tools=[web_search_tool_agent],
        llm=st.session_state['llm'],
        agent="zero-shot-react-description",
        handle_parsing_errors=True,
        verbose=True,
    )

    return web_search_agent

# Custom Callback Handler
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder

    def on_agent_action(self, action, **kwargs):
        """Called when the agent takes an action."""
        thought = f"**Agent's Thought:**\n{action.log}"
        self.placeholder.markdown(thought)

    def on_agent_finish(self, finish, **kwargs):
        """Called when the agent finishes."""
        # Clear the placeholder when done
        self.placeholder.empty()

# Streamlit app configuration
st.set_page_config(
    page_title="Chat with Course Database",
    page_icon="open_file_folder",
    layout="wide"
)

st.title("🗄️ Chat with Course Database")

# Check if 'vectorstore' and 'df' are in st.session_state
if 'vectorstore' not in st.session_state or 'df' not in st.session_state:
    st.error("Vector store and data are not available. Please go to the main page first.")
    st.stop()
else:
    df = st.session_state['df']
    vectorstore = st.session_state['vectorstore']

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize progress bar for agent initialization
progress_bar = st.progress(0)
status_text = st.empty()

# Initialize the LLM (ChatGroq)
if 'llm' not in st.session_state:
    status_text.text("Initializing LLM...")
    st.session_state['llm'] = ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
    progress_bar.progress(0.3)

# Set up the conversation chain
if 'conversation_chain' not in st.session_state:
    status_text.text("Setting up conversation chain...")
    st.session_state['conversation_chain'] = create_chain(vectorstore)
    progress_bar.progress(0.6)

# Initialize models and compute course embeddings (if not already done)
if 'classifier' not in st.session_state:
    status_text.text("Loading classifier model...")
    st.session_state['classifier'] = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    progress_bar.progress(0.7)

if 'embedder' not in st.session_state:
    status_text.text("Loading sentence embedding model...")
    st.session_state['embedder'] = SentenceTransformer("all-mpnet-base-v2")
    progress_bar.progress(0.8)

if 'course_embeddings' not in st.session_state:
    status_text.text("Computing course embeddings...")
    st.session_state['course_embeddings'] = st.session_state['embedder'].encode(df['course_summary'], convert_to_tensor=True)
    progress_bar.progress(0.9)

# Initialize agents
if 'primary_agent' not in st.session_state:
    status_text.text("Initializing primary agent...")
    st.session_state['primary_agent'] = create_primary_agent()
    progress_bar.progress(0.95)

if 'augmentation_agent' not in st.session_state:
    status_text.text("Initializing augmentation agent...")
    st.session_state['augmentation_agent'] = create_augmentation_agent()

if 'web_search_agent' not in st.session_state:
    status_text.text("Initializing web search agent...")
    st.session_state['web_search_agent'] = create_web_search_agent()

# Clear progress bar and status text after initialization
progress_bar.empty()
status_text.empty()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for chat
user_input = st.chat_input("Ask a question about courses or materials science...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Route the query
    route = classifier_router(user_input)

    # Create a placeholder for agent's thoughts
    thought_placeholder = st.empty()

    # Initialize the callback handler
    callback_handler = StreamlitCallbackHandler(thought_placeholder)
    callbacks = [callback_handler]

    if route == 'vectorstore':
        # Use the primary agent
        assistant_response = st.session_state['primary_agent'].run(user_input, callbacks=callbacks)
        
        # Enhance response using the augmentation agent
        augmented_response = st.session_state['augmentation_agent'].run(user_input, callbacks=callbacks)
        final_response = assistant_response + "\n\n" + augmented_response
    elif route == 'web_search':
        # Use the web search agent
        web_search_response = st.session_state['web_search_agent'].run(user_input, callbacks=callbacks)
        
        # Evaluate the response using the evaluator agent
        is_appropriate, evaluation_result = evaluator_agent(web_search_response, st.session_state['llm'])

        if is_appropriate:
            # Process the search results using the LLM to generate a coherent response
            final_response = safe_groq_call(st.session_state['llm'],
                                            f"Based on the following information, provide an informative response to the user's query:\n\n{web_search_response}")
        else:
            # If not appropriate, inform the user
            final_response = "The information found was not appropriate. Please ask about topics related to materials science and engineering."
    elif route == 'reject_query':
        # Inform the user that the query is not appropriate
        final_response = "I'm sorry, but I can only assist with educational queries related to materials science and engineering. Please ask a question related to these topics."
    else:
        # Handle unexpected routes
        final_response = "I'm sorry, but I was unable to process your request."

    # Clear the thought placeholder after the agent finishes
    thought_placeholder.empty()

    # Display assistant's response
    with st.chat_message("assistant"):
        st.markdown(final_response)

    # Add assistant's response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": final_response})
