import os
import streamlit as st
import bs4
import datetime
import re
import requests
from urllib.parse import urljoin, unquote
import shutil
from collections import deque
from langchain.output_parsers import PydanticOutputParser

# LangChain and vector store imports
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.docstore.document import Document
from pydantic import BaseModel, Field
from typing import Optional
from playwright.sync_api import sync_playwright

load_dotenv()

# --- Helper functions to extract data from URL ---
def extract_info_from_url(url: str) -> dict:
    """Parses a Swisscom Workday URL to extract a clean title and the unique Requisition ID."""
    try:
        # Isolate the part of the URL with the job info
        job_part = url.split('/job/')[-1]
        
        # --- IMPROVEMENT: Use Requisition ID for uniqueness ---
        req_id_match = re.search(r'(R-\d+(-\d+)?)', job_part)
        req_id = req_id_match.group(0) if req_id_match else f"UNKNOWN_ID_{hash(url)}"

        # Extract title slug, which may now contain a location prefix
        # Example: Bern/Leiter-in--Central-Controlling-B2B_R-0002444-1
        title_slug = job_part.split(f'_{req_id}')[0]
        
        # Remove potential location prefix like "Bern/" for a cleaner title
        if '/' in title_slug:
            title_slug = title_slug.split('/')[-1]

        cleaned_title = unquote(title_slug).replace('---', ' ').replace('--', '-').replace('-', ' ')
        return {
            "title": cleaned_title.upper(),
            "id": req_id
        }
    except IndexError:
        return {
            "title": "TITLE NOT FOUND",
            "id": f"UNKNOWN_ID_{hash(url)}"
        }

# --- Function to save unanswered questions ---
def save_unanswered_question(question: str):
    timestamp = datetime.datetime.now().isoformat()
    with open("Unanswered_questions_Swisscom.txt", "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {question}\n")

# --- Initialize st.session_state variables ---
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
if "all_jobs" not in st.session_state:
    st.session_state.all_jobs = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "jobs_loaded" not in st.session_state:
    st.session_state.jobs_loaded = False

# --- Pydantic class for LLM extraction ---
class JobDetails(BaseModel):
    workload: Optional[str] = Field(description="The workload or 'Pensum', e.g., '80-100%'. If not found, this should be null.")
    location: Optional[str] = Field(description="The primary job location, e.g., 'Bern', 'Zürich', 'Remote'. If not found, this should be null.")

# --- Manual, reliable data loaders ---
def load_jobs_with_playwright(urls: list[str]) -> list[Document]:
    """Manually controls Playwright to load JS-heavy pages and create clean Document objects."""
    print(f"Loading {len(urls)} job posts with Playwright...")
    job_docs = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for url in urls:
            try:
                page = browser.new_page()
                # --- IMPROVEMENT: Increased timeout for slow pages ---
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                
                # --- IMPROVEMENT: Use a more specific selector for job content ---
                content_selector = '[data-automation-id="jobPostingDescription"]'
                page.wait_for_selector(content_selector, timeout=30000) # Wait for the specific element
                
                html_content = page.locator(content_selector).inner_html()
                soup = bs4.BeautifulSoup(html_content, "lxml")
                text_content = soup.get_text(separator=" ", strip=True)
                
                doc = Document(page_content=text_content, metadata={"source": url})
                job_docs.append(doc)
                page.close()
                print(f"  - Successfully loaded {url}")
            except Exception as e:
                print(f"  - FAILED to load {url}. Error: {e}")
        browser.close()
    print(f"Successfully loaded {len(job_docs)} of {len(urls)} job posts.")
    return job_docs

def hybrid_crawler(start_url: str, max_depth: int = 1) -> list[Document]:
    """Crawls a website for general info."""
    all_docs = []
    visited_urls = set()
    queue = deque([(start_url, 0)])
    start_domain = re.search(r"https?://([^/]+)", start_url).group(1)
    while queue:
        current_url, depth = queue.popleft()
        if current_url in visited_urls or depth > max_depth:
            continue
        visited_urls.add(current_url)
        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()
            soup = bs4.BeautifulSoup(response.content, "lxml")
            text_content = soup.get_text(strip=True, separator=" ")
            doc = Document(page_content=text_content, metadata={"source": current_url})
            all_docs.append(doc)
            if depth < max_depth:
                for link in soup.find_all("a", href=True):
                    href = link['href']
                    absolute_url = urljoin(current_url, href).split('#')[0]
                    if start_domain in absolute_url and not absolute_url.startswith(("mailto:", "tel:")):
                         if absolute_url not in visited_urls:
                            queue.append((absolute_url, depth + 1))
        except Exception as e:
            print(f"  - Crawler failed on {current_url}. Error: {e}")
    return all_docs


# --- RAG DATA LOADING AND PROCESSING ---
@st.cache_resource(show_spinner="Loading company info and job posts...")
def load_rag_components():
    persist_directory = "chroma_db_swisscom"
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    # 1. Load Data
    job_post_urls = [
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
    "https://swisscom.wd103.myworkdayjobs.com/de-DE/SwisscomExternalCareers/job/Specialised-Sales-Cyper-Security-Services_R-0002471-1",
    ]
    
    job_docs = load_jobs_with_playwright(job_post_urls)
    general_docs = hybrid_crawler("https://www.swisscom.ch/de/about.html")
    all_docs = job_docs + general_docs

    # 2. Process and Add Metadata
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    parser = PydanticOutputParser(pydantic_object=JobDetails)
    prompt_for_extraction = PromptTemplate(template="From the following text, extract the workload (Pensum) and the primary job location.\n{format_instructions}\nTEXT: ```{text}```", input_variables=["text"], partial_variables={"format_instructions": parser.get_format_instructions()})
    extraction_chain = prompt_for_extraction | llm | parser

    for doc in all_docs:
        source_url = doc.metadata.get('source', '')
        if source_url in job_post_urls:
            doc.metadata['type'] = 'job'
            # --- IMPROVEMENT: Use new function to get title and ID ---
            job_info = extract_info_from_url(source_url)
            doc.metadata['title'] = job_info['title']
            doc.metadata['job_id'] = job_info['id'] 
            try:
                details = extraction_chain.invoke({"text": doc.page_content[:4000]})
                workload = details.workload or "N/A"
                location = details.location or "N/A"
            except Exception:
                workload = "N/A"
                location = "N/A"
            doc.metadata['workload'] = workload
            doc.metadata['location'] = location
            print(f"  > Processed Job: {job_info['title']} ({job_info['id']}) (Location: {location})")
        else:
            doc.metadata['type'] = 'info'

    # 3. Split and Store in VectorDB
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    splits = text_splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
    return vectorstore

# --- Load Static RAG Resources ---
vectorstore = load_rag_components()

# --- ROBUST UI LIST LOADING ---
if vectorstore and not st.session_state.jobs_loaded:
    print("Retrieving unique jobs for UI menu...")
    retrieved_data = vectorstore.get(where={"type": "job"})
    
    all_job_docs = []
    if retrieved_data and retrieved_data.get('documents'):
        for i, text in enumerate(retrieved_data['documents']):
            all_job_docs.append(Document(page_content=text, metadata=retrieved_data['metadatas'][i]))
    
    unique_jobs = {}
    for doc in all_job_docs:
        # --- IMPROVEMENT: Use the reliable job_id for uniqueness ---
        job_id = doc.metadata.get("job_id")
        if job_id and job_id not in unique_jobs:
            unique_jobs[job_id] = doc # Store the first chunk for each job
    
    st.session_state.all_jobs = list(unique_jobs.values())
    st.session_state.jobs_loaded = True
    print(f"Loaded {len(st.session_state.all_jobs)} unique jobs into the UI menu.")


# --- ROBUST DYNAMIC CONTEXT BUILDING ---
def get_combined_context(question: str, selected_job_id: str, chroma_vectorstore_instance: Chroma) -> str:
    parts = []
    general_retriever = chroma_vectorstore_instance.as_retriever(search_kwargs={"k": 3, "filter": {"type": "info"}})
    general_docs = general_retriever.get_relevant_documents(question)
    if general_docs:
        parts.append("Allgemeine Unternehmensinformationen:\n" + "\n\n".join(d.page_content for d in general_docs))

    if selected_job_id:
        # --- IMPROVEMENT: Filter context by the reliable job_id ---
        job_data = chroma_vectorstore_instance.get(where={"job_id": selected_job_id})
        if job_data and job_data.get('documents'):
            job_title = job_data['metadatas'][0].get('title', selected_job_id)
            full_job_text = "\n\n".join(job_data['documents'])
            parts.append(f"Informationen zur ausgewählten Stelle ({job_title}):\n" + full_job_text)

    return "\n\n---\n\n".join(parts) if parts else "Kein spezifischer Kontext gefunden."

# --- RAG Chain Definition ---
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Sie sind ein hilfreicher Assistent für Bewerber bei Swisscom. Beantworten Sie die Frage klar und präzise basierend auf dem Kontext. Wenn die Antwort nicht im Kontext steht, sagen Sie das. Antworten Sie auf Deutsch. Stellen Sie am Ende eine relevante Follow-up-Frage, um die Motivation oder die Soft Skills des Bewerbers zu bewerten.\n\nKontext: {context}\nFrage: {question}\n\nIhre Antwort:"""
)

def get_rag_chain(selected_job_id):
    chain = (
        {"context": lambda x: get_combined_context(x["question"], selected_job_id, vectorstore), "question": lambda x: x["question"]}
        | prompt_template
        | ChatOpenAI(model="gpt-4o", temperature=0)
        | StrOutputParser()
    )
    return chain

# --- Main Streamlit App UI ---
st.title("Swisscom Karriere-Assistent")

# --- Chat UI Logic ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Fragen Sie etwas über Swisscom oder eine Stelle..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if user_input.strip().lower() == "exit":
        st.session_state.selected_job = None
        response_msg = "🔄 Job-Auswahl zurückgesetzt. Bitte wählen Sie eine andere Stelle oder stellen Sie eine allgemeine Frage."
        st.session_state.messages.append({"role": "assistant", "content": response_msg})
        st.rerun()
    else:
        with st.spinner("Thinking..."):
            rag_chain = get_rag_chain(st.session_state.selected_job)
            response = rag_chain.invoke({"question": user_input})
            if "ich weiß nicht" in response.lower() or "kann ich ihnen nicht beantworten" in response.lower():
                save_unanswered_question(user_input)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# --- Job Menu UI ---
if st.session_state.selected_job is None:
    st.markdown("### Aktuell offene Stellen")
    if not st.session_state.all_jobs:
        st.info("Jobs werden geladen oder sind nicht verfügbar.")
    else:
        # Sort jobs alphabetically by title for a consistent order
        sorted_jobs = sorted(st.session_state.all_jobs, key=lambda doc: doc.metadata.get("title", ""))
        for i, doc in enumerate(sorted_jobs):
            meta = doc.metadata
            job_title = meta.get("title", "UNBEKANNTE STELLE")
            job_id = meta.get("job_id", f"unknown_{i}") # Use job_id for the button key
            workload = meta.get("workload", "?")
            location = meta.get("location", "?")
            label = f"**{job_title}** – Ort: {location} | Pensum: {workload}"
            
            if st.button(label, key=f"job_btn_{job_id}"):
                # --- IMPROVEMENT: Store the reliable job_id in the session state ---
                st.session_state.selected_job = job_id
                welcome_msg = f"✅ **{job_title}** ausgewählt. Wozu möchten Sie mehr erfahren?\n\n*Um die Auswahl zurückzusetzen, schreiben Sie **exit**.*"
                st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
                st.rerun()