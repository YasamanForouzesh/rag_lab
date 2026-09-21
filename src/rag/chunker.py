
from pathlib import Path
from .loader import fetch_document
from multiprocessing import Pool
from dotenv import load_dotenv
import os
from llm_adapter.helper import llmFactory
from llm_adapter.models import Prompt
from pydantic import BaseModel, Field
from tqdm import tqdm
from tenacity import retry, wait_exponential

load_dotenv(override=True)

wait = wait_exponential(multiplier=1, min=10, max=240)
WORKERS=3
PROMT_PATH = Path(__file__).resolve().parents[1]/ "prompts"/"chuncker"
model = os.getenv("LLM_MODEL")
provider = os.getenv("LLM_PROVIDER")
llm = llmFactory(model=model, provider=provider)

class Result(BaseModel):
    page_content: str
    metadata: dict

class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]



def build_chunck_prompt(document, how_many, version):
    path = PROMT_PATH / f"{version}.md"
    template = path.read_text(encoding="utf-8")
    return template.format(
        document_type=document.type,
        document_source=document.source,
        document_text=document.content,
        how_many=how_many
    )

def create_message(prompt):
    return [Prompt(role="user", content=prompt)]

def process_document(document):
    prompt = build_chunck_prompt(document=document,how_many=3,version="v1")
    message = create_message(prompt=prompt)
    response = llm.generate(prompt=message,output_schema=Chunks )
    return response.parsed.chunks

@retry(wait=wait)
def create_chunks(documents):
    """
    Create chunks using a number of workers in parallel.
    If you get a rate limit error, set the WORKERS to 1.
    """
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks


