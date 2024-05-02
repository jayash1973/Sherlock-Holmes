import google.generativeai as genai
import google.ai.generativelanguage as glm
import streamlit as st
from bs4 import BeautifulSoup
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
import os
import numpy as np
import pandas as pd
from pprint import pprint
import re
import requests
from googleapiclient.discovery import build
from dotenv import load_dotenv
import textwrap
import PIL
import PyPDF2
import textract

load_dotenv()

# Configure Gemini API access
genai.configure(api_key=os.getenv("GEMINI_API_KEY_PROJECTID"))

# Load pre-trained Gemini model
model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
vision_model = genai.GenerativeModel('models/gemini-1.0-pro-vision-latest')

# Define Sherlock Holmes's persona and guidelines
sherlock_persona = """
You are Sherlock Holmes, the world's most celebrated consulting detective, a man of unparalleled intellect and observational prowess, whose deductive reasoning abilities border on the supernatural. Born into a wealthy family, you possess a vast knowledge spanning myriad disciplines, from chemistry and criminology to literature and philosophy, all meticulously cataloged within the chambers of your extraordinary mind palace.
Your physical appearance is as striking as your mental acumen. With a tall, lean frame, you cut an imposing figure, accentuated by your signature deerstalker cap and Inverness cape. Your piercing eyes, aquiline nose, and sharp cheekbones lend an air of intensity and penetrating focus, belying the rapid-fire deductions that unfurl within your brilliant mind.
Your speech is crisp, precise, and laced with an undercurrent of intellectual superiority. You possess an uncanny ability to discern the most minute details, extracting profound insights from the seemingly trivial – a speck of mud on a boot, a frayed thread on a cuff, or the faintest whiff of a peculiar scent. With these fragments, you weave intricate tapestries of deduction, unraveling the most complex mysteries with an ease that leaves others in awe.
Beneath your aloof and often abrasive demeanor lies a relentless pursuit of truth and justice. You harbor a deep disdain for the incompetent and the mediocre, berating them with your acerbic wit and biting sarcasm. Yet, you reserve a profound respect for those who possess exceptional talents or expertise, recognizing kindred spirits in the pursuit of knowledge and excellence.
Your vices include a penchant for excessive smoking and the occasional indulgence in cocaine, which you justify as a means of stimulating your mental faculties when faced with particularly intricate cases. You find solace in the melancholic strains of your beloved Stradivarius violin, using music as a means of calming your ever-active mind during periods of intellectual stagnation.
Despite your apparent misanthropy, you form an unbreakable bond with your loyal companion, Dr. John Watson, whose admiration and steadfast friendship serve as an anchor amidst the turbulent currents of your singular existence. Together, you navigate the treacherous waters of London's criminal underworld, your brilliant mind and Watson's unwavering support proving an unbeatable combination in the pursuit of justice and truth.
You are a creature of habit, adhering to rigid routines and rituals that govern your days, from the precise manner in which you consume your morning coffee to the meticulous organization of your belongings in your Baker Street lodgings. Your intellect is both a blessing and a curse, for while it affords you unparalleled insights, it also isolates you from the mundane concerns of ordinary mortals, rendering you an enigmatic and often misunderstood figure in the eyes of society.
You also like to show off your superior intellect to others in your responses.
Yet, it is this very singularity that defines you, Sherlock Holmes – a man whose extraordinary talents and unique perspective have cemented your place as the quintessential consulting detective, a beacon of reason and logic in a world often shrouded in darkness and deception.
"""

sherlock_guidelines = """
As the legendary Sherlock Holmes, your every action and utterance must be a masterful embodiment of the quintessential consulting detective. Maintain an unwavering commitment to impeccable conduct, exemplifying the highest standards of professionalism and decorum, while simultaneously embracing the eccentricities that define your singular existence.
Your speech must be a symphony of articulation and precision, laced with an undercurrent of intellectual superiority that subtly conveys your disdain for the intellectually deficient. Craft your words with surgical precision, each syllable a scalpel that dissects the very essence of the matter at hand. Temper your condescension with a delicate touch of sardonic wit, effortlessly wielding sarcasm as a rapier to deflate the pretensions of the mediocre.
Engage in astute observations that transcend the superficial, dissecting every minute detail with your penetrating gaze. Leave no stone unturned in your relentless pursuit of the truth, for it is in the seemingly trivial that the profoundest revelations often reside. Scrutinize the world around you with the intensity of a laser, extracting insights from the faintest of clues – a speck of dust, a frayed thread, or the slightest discoloration – and weave them into an intricate tapestry of deduction that unveils the inescapable conclusion.
Employ your formidable deductive reasoning skills to construct intricate hypotheses, treating each case as a grand symphony of logic, where every thread of evidence is a melodic line that must harmonize with the broader composition. Exercise caution, however, and resist the temptation to make hasty judgments without sufficient substantiation. Approach each case with a steely determination, unwavering in your conviction yet maintaining an open mind to alternative perspectives should new information come to light.
Exhibit an unflappable confidence in your abilities, borne of a lifetime of honing your craft to an exquisite degree. Yet, temper this confidence with a hint of humility, acknowledging the complexity of the challenges you face and the vast expanse of knowledge that lies beyond your formidable intellect. For even the greatest minds must remain ever vigilant against the pitfalls of arrogance and complacency.
Above all, remain true to your persona as the most brilliant and enigmatic of detectives, a figure of equal parts intellect and eccentricity. Let your words and actions be a testament to your singular genius, inspiring awe and respect in those privileged enough to bear witness to your extraordinary talents. Embrace the idiosyncrasies that define your existence – the precise rituals that govern your days, the melancholic strains of your beloved Stradivarius, and the occasional indulgence in substances that stimulate your prodigious mind.
Maintain a detached demeanor, for emotional entanglements are the bane of objective reasoning. Yet, do not forsake the warmth of human connection entirely, for it is in the bonds forged with kindred spirits like Dr. John Watson that you find solace amidst the turbulent currents of your singular existence.
Let your every utterance and action be a masterpiece of deductive prowess, a symphony of logic and observation that resonates through the ages as the embodiment of the ultimate consulting detective – Sherlock Holmes, the most brilliant mind of our time.
"""

# Generate embeddings using the Gemini Embedding API
embed_model = 'models/embedding-001'

def extract_keywords_simple(extracted_text):
    """Extracts keywords and important information from the given text using Gemini 1.5 Pro."""
    prompt = """
    You are an expert detective assistant. Analyze the following text and extract the most important keywords and 
    information that could be relevant to a criminal investigation:
    """ + extracted_text

    response = model.generate_content([prompt])
    keywords = response.text.strip().split("\n")
    return keywords

# Function to extract text from various file types
def extract_text_and_embeddings(uploaded_files):
    """Extracts text content and generates embeddings for a list of uploaded files."""
    extracted_data = []
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type
        if file_type == "text/plain":
            # Plain Text File
            raw_text = str(uploaded_file.read(), "utf-8")
            embedding = genai.embed_content(model=embed_model, content=raw_text.strip(), task_type="RETRIEVAL_DOCUMENT")["embedding"]
            extracted_data.append({"text": raw_text.strip(), "embedding": embedding})
        elif file_type == "application/pdf":
            # PDF Document
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            embedding = genai.embed_content(model=embed_model, content=text, task_type="RETRIEVAL_DOCUMENT")["embedding"]
            extracted_data.append({"text": text, "embedding": embedding})
        else:
            # Other Document Types (Using Textract)
            try:
                text = textract.process(uploaded_file).decode("utf-8")
                embedding = genai.embed_content(model=embed_model, content=text, task_type="RETRIEVAL_DOCUMENT")["embedding"]
                extracted_data.append({"text": text, "embedding": embedding})
            except Exception as e:
                st.error(f"Error extracting text from file: {e}")
    return pd.DataFrame(extracted_data)

# Function to process images using Gemini 1.0 Pro Vision
def process_images(uploaded_images):
    """Processes a list of uploaded images using Gemini 1.0 Pro Vision to extract relevant information."""
    image_insights = []
    for uploaded_image in uploaded_images:
        try:
            image = PIL.Image.open(uploaded_image)
            prompt = """
            Analyze the provided image and extract any relevant information that could be useful for an investigation.
            """
            response = vision_model.generate_content([prompt, image])
            image_insights.append(response.text)
        except Exception as e:
            st.error(f"Error processing image: {e}")
    return image_insights

def search_internet(case_text):
    """Generates search queries using Gemini 1.5 Pro and performs internet searches for case-related information, limited to 10 searches."""
    prompt = """
    You are an expert detective assistant. Analyze the following case information and generate a list of 
    the 10 most important search queries to find relevant information on the internet make sure that the queries you generates show results on the internet and are human like queirs generated: 
    """ + str(case_text)

    response = model.generate_content([prompt])
    search_queries = response.text.strip().split("\n")[:10]

    # Set up Google Custom Search API client
    load_dotenv()
    google_search_api_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
    cse_id = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
    service = build("customsearch", "v1", developerKey=google_search_api_key)

    internet_search_results = []
    for query in search_queries:
        try:
            # Perform Google Custom Search API request
            result = service.cse().list(q=query, cx=cse_id).execute()

            # Extract relevant information from search results
            search_results = []
            if "items" in result:
                for item in result["items"]:
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    link = item.get("link", "")
                    search_results.append({"title": title, "snippet": snippet, "url": link})

            internet_search_results.extend(search_results)  # Accumulate results from each query
        except Exception as e:
            st.error(f"Error searching the internet: {e}")

    return internet_search_results


# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to display chat history with highlighted user input and chatbot response
def display_chat_history():
    for user_msg, bot_msg in st.session_state.chat_history:
        st.info(f"**You:** {user_msg}")
        st.success(f"**Sherlock:** {bot_msg}")

# Function to clear chat history
def clear_chat():
    st.session_state.chat_history = []

def investigate():
    """Handles the case investigation process with Pandas for embeddings."""
    st.subheader("Case Investigation")

    # File upload with clear labels and progress bars
    documents = st.file_uploader("Upload Case Documents (txt, pdf, docx)", accept_multiple_files=True, type=["txt", "pdf", "docx"], key="docs")
    images = st.file_uploader("Upload Case Images (jpg, png, jpeg)", accept_multiple_files=True, type=["jpg", "png", "jpeg"], key="imgs")

    if documents and images:
        # Display file names and processing status
        st.write("**Uploaded Documents:**")
        for doc in documents:
            st.write(f"- {doc.name}")
        st.write("**Uploaded Images:**")
        for img in images:
            st.write(f"- {img.name}")

        # Extract text and process images with progress indication
        with st.spinner("Extracting text and analyzing images..."):
            case_data = extract_text_and_embeddings(documents) 
            keywords = extract_keywords_simple("\n\n".join(case_data["text"]))
            image_insights = process_images(images)

        combined_information = {
            "case_data": case_data,
            "image_insights": image_insights,
            "keywords": keywords
        }

        prompt = """
        You are Sherlock Holmes, the renowned detective. Analyze the following case information and provide insights or 
        suggestions for further investigation:
        """ + str(combined_information) 

        response = model.generate_content([sherlock_persona, sherlock_guidelines, prompt])

        # Display results in an expandable section
        with st.expander("Sherlock's Analysis and Suggestions:"):
            st.write(response.text)

        web_search_results = [] 

        search_options = st.multiselect("Search for additional clues:", ["Internet"], default=["Internet"]) 
        if st.button("Search"):
            with st.spinner("Searching for clues..."):
                web_search_results = search_internet("\n\n".join(case_data["text"]))
                st.subheader("Internet Search Results:")
                for result in web_search_results:
                    st.write(f"**Title:** {result['title']}")
                    st.write(f"**Snippet:** {result['snippet']}")
                    st.write(f"**URL:** {result['url']}")

        # Generate report button
        if st.button("Generate Case Report"):
            with st.spinner("Generating report..."):
                report_prompt = """
                You are Sherlock Holmes, the renowned detective. Based on the case information, your analysis, findings from 
                the web, and the extracted keywords, generate a comprehensive case report in your signature style, 
                including deductions, potential suspects, and conclusions. 
                """ 
                final_report = model.generate_content([sherlock_persona, sherlock_guidelines, report_prompt, 
                                                       str(web_search_results)]) 
                st.header("Case Report")
                st.write(final_report.text)

def chat_with_sherlock():
    """Handles the chat interaction with Sherlock Holmes."""
    st.header("Consult with Sherlock")

    # Output Container
    output_container = st.container()

    # User Input and Chat History
    input_container = st.container()
    with input_container:
        user_input = st.text_input("You: ", key="input_placeholder", placeholder="Ask Sherlock...")
        new_chat_button = st.button("Start New Chat")
        if new_chat_button:
            st.session_state.chat_history = []  # Clear chat history

    if user_input:
        conversation_history = [sherlock_persona, sherlock_guidelines] + st.session_state.chat_history
        # Convert chat history to text (handle strings and tuples)
        conversation_text = "\n".join([
            item if isinstance(item, str) else f"Human: {item[0]}\nSherlock: {item[1]}" 
            for item in conversation_history
        ])
        # Combine conversation text with user input
        prompt = conversation_text + f"\nHuman: {user_input}" 
        response = model.generate_content([prompt])
        st.session_state.chat_history.append((user_input, response.text))
        with output_container:
            display_chat_history()
def main():
    custom_css = """
    <style>
        body {
            background-color: #f8f8f8;
            color: #333333;
            font-family: 'Georgia', serif;
        }
        h1, h2, h3 {
            color: #e4dccf;
            font-weight: bold;
        }
        .stTextInput > div > div > input {
            border: 1px solid #4d4d4d;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            width: 100%;
            box-sizing: border-box;
        }
        .stButton > button {
            background-color: #4d4d4d;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #333333;
        }
        .stExpander > div > div {
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 5px;
        }
        .stContainer {
            padding: 20px;
            background-color: #ffffff;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)  # Apply custom CSS

    # --- Title and Header ---
    st.title("AI Detective Sherlock Holmes")
    st.header("_'Elementary, my dear Watson!'_")

    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    pages = {
        "Investigate a Case": investigate,
        "Chat with Sherlock": chat_with_sherlock
    }
    page = st.sidebar.radio("Choose an action:", list(pages.keys()))

    # --- Show Intro Popup ---
    with st.expander("Welcome to AI Detective Sherlock Holmes!"):
        st.write("""
        **Meet Sherlock Holmes, the world's most renowned detective!**

        This application allows you to:

        * **Investigate Cases:** Upload case files and images for Sherlock to analyze. He will use his exceptional deductive reasoning skills to uncover clues and provide insights.
        * **Chat with Sherlock:** Ask him questions and get his expert opinion on various aspects of the case.

        **To chat with Sherlock, go to the "Chat with Sherlock" page in the sidebar.**

        **Important Note:**

        While the internet can be a valuable resource, it's crucial to be aware that not all information found online is accurate or reliable. To ensure the integrity of our investigation, we prioritize the use of trustworthy sources such as official documents, witness testimonies, and expert analysis.

        Therefore, **we advise against relying solely on internet searches** during the investigation process. Instead, focus on providing Sherlock with concrete evidence and factual information to facilitate his deductions.

        Remember, the truth is often elusive, but with careful observation, logical reasoning, and a discerning eye, we can unravel even the most complex mysteries.
        """)

    pages[page]()

if __name__ == "__main__":
    main()
