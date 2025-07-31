import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
import fitz  # PyMuPDF

from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.nodes import EpisodeType

# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------
load_dotenv()

# Neo4j connection
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')

# Google Gemini API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError('GEMINI_API_KEY must be set in .env')

# Directory of PDFs to ingest
PDF_DIR = os.getenv('PDF_DIR', './pdfs')

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------
# INITIALIZE Graphiti
# ------------------------------------------------
graphiti = Graphiti(
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    llm_client=GeminiClient(config=LLMConfig(api_key=GEMINI_API_KEY, model='gemini-2.0-flash')),
    embedder=GeminiEmbedder(config=GeminiEmbedderConfig(api_key=GEMINI_API_KEY, embedding_model='embedding-001')),
    cross_encoder=GeminiRerankerClient(config=LLMConfig(api_key=GEMINI_API_KEY, model='gemini-2.5-flash-lite-preview-06-17'))
)

async def extract_sections(pdf_path):
    """
    Parse a PDF into sections based on its Table of Contents.
    Returns a list of dicts: {"heading", "subheading", "content"}.
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc(simple=True)
    # If no TOC, treat entire doc as one section
    if not toc:
        text = "".join(page.get_text() for page in doc)
        return [{
            'heading': os.path.basename(pdf_path),
            'subheading': None,
            'content': text
        }]

    # Build sections by TOC entries
    sections = []
    for idx, (level, title, page_no) in enumerate(toc):
        start = page_no - 1
        end = toc[idx+1][2] - 1 if idx+1 < len(toc) else doc.page_count
        content = ''
        for p in range(start, end):
            content += doc[p].get_text()
        if level == 1:
            # Top-level heading
            sections.append({'heading': title, 'subheading': None, 'content': content})
        else:
            # Subheading under last heading
            if not sections:
                sections.append({'heading': os.path.basename(pdf_path), 'subheading': title, 'content': content})
            else:
                sections.append({'heading': sections[-1]['heading'], 'subheading': title, 'content': content})
    doc.close()
    return sections

async def ingest_pdf(pdf_path):
    logger.info(f'Processing {pdf_path}')
    sections = await extract_sections(pdf_path)
    # Add each section as an episode
    for i, sec in enumerate(sections):
        body = json.dumps({
            'heading': sec['heading'],
            'subheading': sec['subheading'],
            'text': sec['content']
        })
        name = f"{os.path.basename(pdf_path)} - section {i+1}"
        await graphiti.add_episode(
            name=name,
            episode_body=body,
            source=EpisodeType.json,
            source_description=f"Section import from {os.path.basename(pdf_path)}",
            reference_time=datetime.now(timezone.utc)
        )
        logger.info(f'Added episode: {name}')

async def main():
    try:
        # Build indices and constraints (run once)
        await graphiti.build_indices_and_constraints()

        # Iterate PDF files
        pdf_files = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
        # Parallel ingestion
        tasks = [ingest_pdf(path) for path in pdf_files]
        await asyncio.gather(*tasks)

        logger.info('All PDFs ingested successfully')
    finally:
        await graphiti.close()
        logger.info('Graphiti connection closed')

if __name__ == '__main__':
    asyncio.run(main())
