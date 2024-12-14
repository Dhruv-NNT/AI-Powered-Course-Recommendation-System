import streamlit as st
import os
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers.tokenization_utils_base')

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import Tool, initialize_agent
from langchain.callbacks.base import BaseCallbackHandler

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

tavily_api_key = os.getenv('TAVILY_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

if tavily_api_key is None or groq_api_key is None:
    raise ValueError("API keys for Tavily and/or Groq are not set in the environment variables.")

def create_chain(vectorstore):
    llm_chain = st.session_state['llm']
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

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
        return True, response
    else:
        return False, evaluation_result

def classifier_router(question):
    classifier = st.session_state['classifier']
    embedder = st.session_state['embedder']
    course_embeddings = st.session_state['course_embeddings']

    question_embedding = embedder.encode(question, convert_to_tensor=True)
    similarities = util.cos_sim(question_embedding, course_embeddings)
    max_similarity_score = similarities.max().item()
    similarity_threshold = 0.5

    if max_similarity_score >= similarity_threshold:
        return 'vectorstore'
    else:
        candidate_labels = ['course-related', 'other']
        result = classifier(question, candidate_labels, hypothesis_template="This query is {}.")
        confidence_threshold = 0.6
        if result['labels'][0] == 'course-related' and result['scores'][0] > confidence_threshold:
            return 'vectorstore'
        else:
            candidate_labels = ['educational', 'non-educational']
            result = classifier(question, candidate_labels, hypothesis_template="This query is {}.")
            if result['labels'][0] == 'educational':
                return 'web_search'
            else:
                return 'reject_query'

def web_search_tool(query):
    results = f"Search results for '{query}' from Tavily (educational content related to materials science and engineering)."
    return results

def safe_groq_call(llm, inputs):
    try:
        return llm.predict(inputs)
    except Exception:
        return "The service is temporarily unavailable. Please try again later."

def create_primary_agent():
    def recommend_courses(query):
        structured_query = f"""
        The user asked: "{query}"

        Please respond by recommending relevant courses FROM Nanyang Technological University ONLY.
        Your answer should start with:
        "Here are some relevant courses based on your query:\n\n"
        
        Then LIST EACH COURSE in the following format (for each course):
        - **Course Name:** <course_name>
        - **Course id:** <course_code>
        - **Course Description:** <brief description>
        - **URL:** <link if available>

        STRICT REQUIREMENTS:
        - ONLY RECOMMEND COURSES OFFERED BY NTU. Do not recommend or mention any external universities or platforms.
        - If you cannot find any relevant NTU courses, DO NOT list external courses. Instead, say exactly:
          "No NTU courses found for this query."
        - Do not provide generic or external resources under any circumstances if no NTU matches are found.
        - Be concise and follow the exact format. If a course is found, start with the heading and list the courses as instructed.
        - If no courses are found, just output the fallback message without the heading or any additional text.
        """
        response = st.session_state['conversation_chain']({"question": structured_query})
        return response["answer"]

    course_recommendation_tool = Tool(
        name="CourseRecommendationTool",
        func=recommend_courses,
        description="Provides course recommendations based on the user's query."
    )

    primary_agent = initialize_agent(
        tools=[course_recommendation_tool],
        llm=st.session_state['llm'],
        agent="zero-shot-react-description",
        handle_parsing_errors=True,
        verbose=True,
        # max_iterations=4,  # Limit to 4 iterations
    )

    return primary_agent

def create_augmentation_agent():
    def augment_response(query):
        search_results = web_search_tool(query)
        
        augmentation_prompt = f"""
        The user asked: "{query}"

        Below is the previously provided answer:
        {{assistant_response}}

        Now, add a new section titled "Additional Verified NTU Resources:".
        
        STRICT REQUIREMENTS:
        - Do not repeat the previously mentioned courses or descriptions.
        - Only list additional resources related to the query that are affiliated with NTU (courses, programs, faculty, research groups).
        - If no additional NTU resources are available, write exactly: "No additional NTU-specific resources were found beyond what was previously mentioned."
        - Do not mention any external universities, platforms, or courses.
        - If the previous answer indicated no NTU courses were found, you must still follow these rules. If you cannot find additional NTU resources, use the fallback sentence above.
        - Format the additional resources as follows:
          
          Additional Verified NTU Resources:
          - **Resource Name:** <Name>
            **Description:** <brief factual description>
            **URL:** <NTU link if available>
        
        If multiple resources are listed, each should follow the above bullet format.
        If none found, provide the exact fallback sentence.
        
        Provided search results (for reference only):
        {search_results}
        """

        return augmentation_prompt

    augmentation_tool = Tool(
        name="AugmentationTool",
        func=augment_response,
        description="Enhances the response with additional NTU learning resources."
    )

    augmentation_agent = initialize_agent(
        tools=[augmentation_tool],
        llm=st.session_state['llm'],
        agent="zero-shot-react-description",
        handle_parsing_errors=True,
        verbose=True,
        # max_iterations=4,  # Limit to 4 iterations
    )

    return augmentation_agent

def create_web_search_agent():
    def web_search(query):
        search_results = web_search_tool(query)
        return search_results

    web_search_tool_agent = Tool(
        name="WebSearchTool",
        func=web_search,
        description="Performs a web search to find information related to the user's query."
    )

    web_search_agent = initialize_agent(
        tools=[web_search_tool_agent],
        llm=st.session_state['llm'],
        agent="zero-shot-react-description",
        handle_parsing_errors=True,
        verbose=True,
        # max_iterations=4,  # Limit to 4 iterations
    )

    return web_search_agent

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder

    def on_agent_action(self, action, **kwargs):
        thought = f"**Agent's Thought:**\n{action.log}"
        self.placeholder.markdown(thought)

    def on_agent_finish(self, finish, **kwargs):
        self.placeholder.empty()

# Extraction Function to Extract Query Part
def extract_query(user_input):
    extraction_prompt = f"""
    The following is a message from a user that may contain simple messages like greetings, gratitude, frustration, along with a query.

    Your task is to extract only the query part of the message, excluding any simple messages such as greetings, gratitude, or expressions of frustration.

    User message: "{user_input}"

    Extracted query:
    """
    query = st.session_state['llm'].predict(extraction_prompt)
    return query.strip()

# Streamlit App Configuration
st.set_page_config(
    page_title="Chat with Course Database",
    page_icon="open_file_folder",
    layout="wide"
)

st.title("🗄️ Chat with Course Database")

# Initialize chat history display
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display existing chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize session state components
if 'vectorstore' not in st.session_state or 'df' not in st.session_state:
    st.error("Vector store and data are not available. Please go to the main page first.")
    st.stop()
else:
    df = st.session_state['df']
    vectorstore = st.session_state['vectorstore']

if 'llm' not in st.session_state:
    st.session_state['llm'] = ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=groq_api_key,
        temperature=0
    )

if 'conversation_chain' not in st.session_state:
    st.session_state['conversation_chain'] = create_chain(vectorstore)

if 'classifier' not in st.session_state:
    st.session_state['classifier'] = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

if 'embedder' not in st.session_state:
    st.session_state['embedder'] = SentenceTransformer("all-mpnet-base-v2")

if 'course_embeddings' not in st.session_state:
    st.session_state['course_embeddings'] = st.session_state['embedder'].encode(df['course_summary'], convert_to_tensor=True)

if 'primary_agent' not in st.session_state:
    st.session_state['primary_agent'] = create_primary_agent()

if 'augmentation_agent' not in st.session_state:
    st.session_state['augmentation_agent'] = create_augmentation_agent()

if 'web_search_agent' not in st.session_state:
    st.session_state['web_search_agent'] = create_web_search_agent()

def detect_and_respond_to_greeting(user_input):
    greeting_detection_prompt = f"""
    The following is a message from a user. Your task is to determine if the message is a friendly greeting.
    If it is a greeting, respond ONLY with a polite greeting message and introduce yourself briefly as a chatbot
    that assists with materials science and engineering resources from NTU.
    If it is not a greeting, respond ONLY with 'not a greeting'.
    
    User message: "{user_input}"
    """
    response = st.session_state['llm'].predict(greeting_detection_prompt)
    response = response.strip().lower()
    if response != 'not a greeting':
        # Respond with the greeting message without internal thoughts
        response_capitalized = response.capitalize()
        with st.chat_message("assistant"):
            st.markdown(response_capitalized)
        st.session_state.chat_history.append({"role": "assistant", "content": response_capitalized})
        return True
    return False

def detect_and_respond_to_gratitude(user_input):
    gratitude_detection_prompt = f"""
    The following is a message from a user. Determine if the message expresses gratitude, thanks, or acknowledgment.
    Examples include "Thank you", "Thanks a lot", "Much appreciated", "I’m grateful", "Cheers", etc.
    
    Instructions:
    - If the user is expressing gratitude or acknowledgment, respond ONLY with a polite acknowledgment such as:
      "You're welcome. I'm glad I could help." or "Happy to assist! Let me know if you have more questions."
    - Do NOT provide any reasoning steps or explanations. Only provide the acknowledgment message directly.
    - If the user is NOT expressing gratitude, respond ONLY with 'not gratitude'.
    
    User message: "{user_input}"
    """
    response = st.session_state['llm'].predict(gratitude_detection_prompt)
    response = response.strip().lower()
    if response != 'not gratitude':
        # Respond with the gratitude acknowledgment without internal thoughts
        response_capitalized = response.capitalize()
        with st.chat_message("assistant"):
            st.markdown(response_capitalized)
        st.session_state.chat_history.append({"role": "assistant", "content": response_capitalized})
        return True
    return False

def detect_and_respond_to_frustration(user_input):
    frustration_detection_prompt = f"""
    The following is a message from a user. Determine if the user is expressing frustration, annoyance, or dissatisfaction.
    
    Instructions:
    - If the user is expressing frustration or annoyance, respond ONLY with an empathetic message such as:
      "I’m sorry you’re feeling frustrated. Could you clarify what’s confusing? I’ll do my best to help."
    - Do NOT provide any reasoning steps or explanations. Only provide the empathetic acknowledgment directly.
    - If the user is NOT expressing frustration, respond ONLY with 'not frustration'.
    
    User message: "{user_input}"
    """
    response = st.session_state['llm'].predict(frustration_detection_prompt)
    response = response.strip().lower()
    if response != 'not frustration':
        # Respond with the empathy message without internal thoughts
        response_capitalized = response.capitalize()
        with st.chat_message("assistant"):
            st.markdown(response_capitalized)
        st.session_state.chat_history.append({"role": "assistant", "content": response_capitalized})
        return True
    return False

user_input = st.chat_input("Ask a question about courses or materials science...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Initialize variable to keep track of query
    extracted_query = None

    # Detection and Response Flow using if-elif-else to prevent multiple detections
    # 1. Detect Frustration
    if detect_and_respond_to_frustration(user_input):
        # Attempt to extract query part
        extracted_query = extract_query(user_input)
        if extracted_query and extracted_query.lower() != 'not frustration':
            pass  # Proceed to handle the extracted query
        else:
            st.stop()

    # 2. Detect Gratitude
    elif detect_and_respond_to_gratitude(user_input):
        # Attempt to extract query part
        extracted_query = extract_query(user_input)
        if extracted_query and extracted_query.lower() != 'not gratitude':
            pass  # Proceed to handle the extracted query
        else:
            st.stop()

    # 3. Detect Greeting
    elif detect_and_respond_to_greeting(user_input):
        # Attempt to extract query part
        extracted_query = extract_query(user_input)
        if extracted_query and extracted_query.lower() != 'not a greeting':
            pass  # Proceed to handle the extracted query
        else:
            st.stop()

    # 4. No Simple Message Detected
    else:
        # Treat the entire input as a query
        query = user_input

    # Determine the query to handle
    if extracted_query and extracted_query.lower() not in ['not frustration', 'not gratitude', 'not a greeting']:
        query = extracted_query
    else:
        # If no query was extracted, and a simple message was detected, stop further processing
        if any([False, False, False]):  # Since detections are exclusive, no need to check all
            st.stop()
        else:
            query = user_input

    # Proceed with classification and agents
    route = classifier_router(query)

    thought_placeholder = st.empty()
    callback_handler = StreamlitCallbackHandler(thought_placeholder)
    callbacks = [callback_handler]

    if route == 'vectorstore':
        assistant_response = st.session_state['primary_agent'].run(query, callbacks=callbacks)

        if "Here are some relevant courses based on your query:" in assistant_response:
            # Primary agent found structured courses
            augmentation_prompt = st.session_state['augmentation_agent'].tools[0].func(query)
            augmentation_prompt = augmentation_prompt.replace("{assistant_response}", assistant_response)
            augmented_response = st.session_state['llm'].predict(augmentation_prompt)
            final_response = assistant_response.strip() + "\n\n" + augmented_response.strip()
        else:
            # No structured courses from primary agent
            no_course_augmentation_prompt = f"""
            The user asked: '{query}'

            The previous answer did not list specific structured courses.
            Add a section titled 'Additional Verified NTU Resources:'.
            
            Strict requirements:
            - Only mention NTU-related resources if they exist.
            - If no NTU resources are found, write exactly "No verified NTU-specific resources were found."
            - Do not mention other universities or platforms.

            If no resources are found, just provide the fallback message.
            """
            augmented_response = st.session_state['llm'].predict(no_course_augmentation_prompt)
            final_response = assistant_response.strip() + "\n\n" + augmented_response.strip()

    elif route == 'web_search':
        web_search_response = st.session_state['web_search_agent'].run(query, callbacks=callbacks)
        is_appropriate, evaluation_result = evaluator_agent(web_search_response, st.session_state['llm'])

        if is_appropriate:
            final_response = safe_groq_call(st.session_state['llm'],
                                            f"Based on the following information, provide an informative response to the user's query:\n\n{web_search_response}")
        else:
            final_response = "The information found was not appropriate. Please ask about topics related to materials science and engineering."
    elif route == 'reject_query':
        final_response = "I'm sorry, but I can only assist with educational queries related to materials science and engineering. Please ask a question related to these topics."
    else:
        final_response = "I'm sorry, but I was unable to process your request."

    thought_placeholder.empty()

    if route in ['vectorstore', 'web_search', 'reject_query']:
        with st.chat_message("assistant"):
            st.markdown(final_response)
        st.session_state.chat_history.append({"role": "assistant", "content": final_response})