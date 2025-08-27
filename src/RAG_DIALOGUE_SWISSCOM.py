import os
import re
import shutil
import datetime
from collections import deque
from typing import Optional
from urllib.parse import urljoin, unquote
import requests
import time
import random

import bs4
import streamlit as st
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from playwright.sync_api import sync_playwright

# --- Initial Setup ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
load_dotenv()


# --- CONFIGURATION CONSTANTS ---
PERSIST_DIRECTORY = "chroma_db_swisscom"
JOB_POST_URLS = [
    "https://swisscom.wd103.myworkdayjobs.com/en-US/SwisscomExternalCareers/job/Business-Analyst-Performance---Reporting_R-0002671-1",
    "https://swisscom.wd103.myworkdayjobs.com/en-US/SwisscomExternalCareers/job/Plattform-Architekt-SMC_R-0002411",
    "https://swisscom.wd103.myworkdayjobs.com/en-US/SwisscomExternalCareers/job/DevOps-Engineer-Virtualisierung_R-0002401",
    "https://swisscom.wd103.myworkdayjobs.com/en-US/SwisscomExternalCareers/job/DevOps-Engineer-CaaS_R-0002404",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Bern/Lehre--Entwickler-in-digitales-Business-EFZ_R-0001774",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Bern/Lehre--Informatiker-in-EFZ-Plattformentwicklung_R-0001766",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Bern/Lehre--Mediamatiker-in-EFZ-im-Modell--Way-up-_R-0001752",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Bern/ICT-Supporter-I_R-0002039",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Zurich/Senior-Data-Scientist_R-0002414",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Bern/Leiter-in--Central-Controlling-B2B_R-0002444-1",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Bern/Lehre--Informatiker-in-EFZ-mit-Fachrichtung-Applikationsentwicklung-im-Modell--Way-up-_R-0001782",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/PiBs--Bachelor-of-Science--BSc--in-Informatik_R-0001781",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Zurich/ICT-Security-Engineer_R-0002381",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Lehre--Kauffrau--mann-EFZ_R-0001754",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Application-Manager-Avanti_R-0001559-1",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Full-Stack-Developper-and-client-partner_R-0002246",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Financial-Controller-NETworks_R-0001934",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Specialised-Sales_R-0001889-1",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Purchasing-Manager_R-0002777-1",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/ICT-System-Engineer-Citrix-Workplace_R-0001415-1",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/ICT-Application-Operation-Manager_R-0001785-1",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Cloud-Specialized-Sales-D-CH_R-0002109",
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Specialised-Sales-Cyper-Security-Services_R-0002471-1",
]
GENERAL_INFO_URL = "https://www.swisscom.ch/de/about.html"
KNOWN_LOCATIONS = {
    "Bern", "Zurich", "Geneva", "Lausanne", "Sion", "Basel", "Luzern",
    "St. Gallen", "Olten", "Chur", "Ostermundigen", "Liebefeld", "Rothenburg"
}

# --- DATABASE CLEANUP ---
if os.path.exists(PERSIST_DIRECTORY):
    print(f"Clearing old database directory: {PERSIST_DIRECTORY}")
    shutil.rmtree(PERSIST_DIRECTORY)


# --- Pydantic Model for Data Extraction ---
class JobWorkload(BaseModel):
    workload: Optional[str] = Field(description="The workload or 'Pensum', e.g., '80-100%'. If not found, this should be null.")


# --- HELPER FUNCTIONS ---
def save_unanswered_question(question: str):
    timestamp = datetime.datetime.now().isoformat()
    with open("Unanswered_questions_Swisscom.txt", "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {question}\n")

def extract_location_and_title_from_url(url: str) -> tuple[str, str]:
    try:
        slug_path = url.split('/job/')[1]
        path_parts = slug_path.split('/')
        title_part = path_parts[-1]
        location_parts = path_parts[:-1]
        location = " ".join(location_parts) if location_parts else "N/A"
        title_slug = re.split(r'_(R[-_]\d+)', title_part)[0]
        raw_title = unquote(title_slug).replace('--', ' ').replace('-', ' ')
        cleaned_title = " ".join(raw_title.split())
        cleaned_location = " ".join(location.split())
        return cleaned_location, cleaned_title
    except (IndexError, AttributeError):
        return "N/A", "Title Not Found In URL"


# --- DATA LOADING FUNCTIONS ---
def load_jobs_with_playwright(urls: list[str], retries: int = 3) -> list[Document]:
    print(f"Loading {len(urls)} job posts with Playwright...")
    job_docs = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for url in urls:
            is_loaded = False
            for attempt in range(retries):
                try:
                    delay = random.uniform(1.5, 3.5)
                    time.sleep(delay)
                    page = browser.new_page()
                    page.goto(url, wait_until="load", timeout=30000)
                    html_content = page.locator("body").inner_html()
                    soup = bs4.BeautifulSoup(html_content, "lxml")
                    text_content = soup.get_text(separator=" ", strip=True)
                    job_docs.append(Document(page_content=text_content, metadata={"source": url}))
                    page.close()
                    is_loaded = True
                    break
                except Exception as e:
                    print(f"  - Attempt {attempt + 1}/{retries} failed for {url}.")
                    if attempt < retries - 1:
                        time.sleep(2)
                    page.close()
            if not is_loaded:
                 print(f"  - FAILED to load {url} after {retries} attempts.")
        browser.close()
    return job_docs

def hybrid_crawler(start_url: str, max_depth: int = 1, retries: int = 3) -> list[Document]:
    print(f"Crawling {start_url} for general info...")
    all_docs = []
    visited_urls = set()
    queue = deque([(start_url, 0)])
    start_domain = re.search(r"https?://([^/]+)", start_url).group(1)
    while queue:
        current_url, depth = queue.popleft()
        if current_url in visited_urls or depth > max_depth:
            continue
        visited_urls.add(current_url)
        for attempt in range(retries):
            try:
                response = requests.get(current_url, timeout=10)
                response.raise_for_status()
                soup = bs4.BeautifulSoup(response.content, "lxml")
                all_docs.append(Document(page_content=soup.get_text(strip=True, separator=" "), metadata={"source": current_url}))
                if depth < max_depth:
                    for link in soup.find_all("a", href=True):
                        absolute_url = urljoin(current_url, link['href']).split('#')[0]
                        if start_domain in absolute_url and not absolute_url.startswith(("mailto:", "tel:")) and absolute_url not in visited_urls:
                            queue.append((absolute_url, depth + 1))
                break 
            except requests.exceptions.RequestException as e:
                print(f"  - Crawler failed on {current_url} (Attempt {attempt + 1}/{retries}). Error: {e}")
                if attempt < retries - 1:
                    time.sleep(2)
    return all_docs


# --- RAG DATA PIPELINE ---
@st.cache_resource(show_spinner="Loading company info and job posts...")
def load_and_process_data() -> tuple[Chroma, list[Document]]:
    """Loads, processes, and stores data, returning the vectorstore and a clean list of jobs for the UI."""
    job_docs = load_jobs_with_playwright(JOB_POST_URLS)
    general_docs = hybrid_crawler(GENERAL_INFO_URL)
    all_docs = job_docs + general_docs

    if not all_docs:
        raise ValueError("Fatal Error: Failed to load any documents. Check internet connection.")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    parser = PydanticOutputParser(pydantic_object=JobWorkload)
    prompt = PromptTemplate(template="Extract the workload.\n{format_instructions}\nTEXT: ```{text}```", input_variables=["text"], partial_variables={"format_instructions": parser.get_format_instructions()})
    extraction_chain = prompt | llm | parser

    processed_docs_for_db = []
    unique_jobs_for_ui = {}

    for doc in all_docs:
        doc.metadata = doc.metadata.copy()
        source_url = doc.metadata.get('source', '')
        if source_url in JOB_POST_URLS:
            doc.metadata['type'] = 'job'
            raw_location, raw_title = extract_location_and_title_from_url(source_url)
            
            if raw_location.title() in KNOWN_LOCATIONS:
                location = raw_location.title()
                title = raw_title.upper()
            else:
                location = "N/A"
                title = " ".join([p for p in [raw_location, raw_title] if p and p != "N/A"]).upper()

            doc.metadata['location'] = location
            doc.metadata['title'] = title
            job_id = f"{title} ({location})"
            doc.metadata['job_id'] = job_id
            
            try:
                workload = extraction_chain.invoke({"text": doc.page_content[:4000]}).workload or "N/A"
            except Exception:
                workload = "N/A"
            doc.metadata['workload'] = workload
            
            if job_id not in unique_jobs_for_ui:
                unique_jobs_for_ui[job_id] = doc
        else:
            doc.metadata['type'] = 'info'
        processed_docs_for_db.append(doc)
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    splits = text_splitter.split_documents(processed_docs_for_db)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)
    
    print("Data processing and database creation complete.")
    return vectorstore, list(unique_jobs_for_ui.values())

# --- STREAMLIT APPLICATION ---
st.title("Swisscom Karriere-Assistent")

# Load data and get both the vectorstore and the clean list of jobs for the UI
vectorstore, ui_jobs = load_and_process_data()

# --- INITIALIZE STREAMLIT SESSION STATE (Moved to correct location) ---
# This must be done after data loading but BEFORE the RAG chain is defined.
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "all_jobs" not in st.session_state:
    st.session_state.all_jobs = ui_jobs
    print(f"Loaded {len(st.session_state.all_jobs)} unique jobs into the UI menu.")

# --- RAG Chain Logic ---
def get_combined_context(question: str, selected_job_id: str) -> str:
    parts = []
    general_retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"type": "info"}})
    general_docs = general_retriever.get_relevant_documents(question)
    if general_docs:
        parts.append("Allgemeine Unternehmensinformationen:\n" + "\n\n".join(d.page_content for d in general_docs))
    if selected_job_id:
        job_data = vectorstore.get(where={"job_id": selected_job_id})
        if job_data and job_data.get('documents'):
            full_job_text = "\n\n".join(job_data['documents'])
            parts.append(f"Informationen zur ausgewählten Stelle ({selected_job_id.split(' (')[0]}):\n" + full_job_text)
    return "\n\n---\n\n".join(parts) if parts else "Kein spezifischer Kontext gefunden."

prompt_template = PromptTemplate(input_variables=["context", "question"], template="""Sie sind ein hilfreicher Assistent für Bewerber bei Swisscom. Beantworten Sie die Frage klar und präzise basierend auf dem Kontext. Wenn die Antwort nicht im Kontext steht, sagen Sie das. Antworten Sie auf Deutsch. Stellen Sie am Ende eine relevante Follow-up-Frage.\n\nKontext: {context}\nFrage: {question}\n\nIhre Antwort:""")
rag_chain = ({"context": lambda x: get_combined_context(x["question"], st.session_state.selected_job), "question": lambda x: x["question"]} | prompt_template | ChatOpenAI(model="gpt-4o", temperature=0) | StrOutputParser())

# --- Chat Interface Logic ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Fragen Sie etwas über Swisscom oder eine Stelle..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    if user_input.strip().lower() == "exit":
        st.session_state.selected_job = None
        st.session_state.messages.append({"role": "assistant", "content": "🔄 Job-Auswahl zurückgesetzt."})
        st.rerun()
    else:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"question": user_input})
            if "ich weiß nicht" in response.lower() or "kann ich ihnen nicht beantworten" in response.lower():
                save_unanswered_question(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# --- Job Menu Display Logic ---
if st.session_state.selected_job is None:
    st.markdown("### Aktuell offene Stellen")
    if not st.session_state.all_jobs:
        st.info("Jobs werden geladen oder sind nicht verfügbar.")
    else:
        for doc in sorted(st.session_state.all_jobs, key=lambda x: x.metadata.get('title', '')):
            meta = doc.metadata
            label = f"**{meta.get('title', 'N/A')}** – Ort: {meta.get('location', 'N/A')} – Pensum: {meta.get('workload', '?')}"
            if st.button(label, key=meta.get("job_id")):
                st.session_state.selected_job = meta.get("job_id")
                st.session_state.messages.append({"role": "assistant", "content": f"✅ **{meta.get('title')}** ausgewählt. Wozu möchten Sie mehr erfahren?\n\n*Um die Auswahl zurückzusetzen, schreiben Sie **exit**.*"})
                st.rerun()