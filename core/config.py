from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_llm_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "multimodal-rag"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Cohere
    cohere_api_key: str

    # App
    app_env: str = "dev"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Storage
    storage_backend: str = "local"
    local_storage_path: str = "./data"

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/rag.db"

    # Tenancy
    default_tenant_id: str = "tenant_dev"
    default_user_id: str = "user_dev"

    # Chunking
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50
    table_row_threshold: int = 20

    # Retrieval
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    chat_history_limit: int = 10

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()