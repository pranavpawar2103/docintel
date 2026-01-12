# DocIntel Architecture

## Table of Contents

- [Overview](#overview)
- [System Components](#system-components)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Design Decisions](#design-decisions)
- [Scalability Considerations](#scalability-considerations)

---

## Overview

DocIntel implements a **Retrieval-Augmented Generation (RAG)** architecture that combines:
- **Information Retrieval**: Finding relevant document chunks using semantic search
- **Generation**: Using LLMs to generate natural language answers
- **Augmentation**: Enhancing responses with source citations and confidence scores

### High-Level Architecture
```
┌──────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                   │
├──────────────────────┬───────────────────────────────────┤
│  Streamlit Web UI    │      REST API (FastAPI)          │
│  - Document Upload   │      - /api/documents/upload     │
│  - Query Interface   │      - /api/query                │
│  - Results Display   │      - /api/stats                │
└──────────────┬───────┴───────────────────────┬───────────┘
               │                               │
               └───────────────┬───────────────┘
                               ▼
┌──────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                     │
├────────────────────┬──────────────────┬──────────────────┤
│ Ingestion Pipeline │  RAG Pipeline    │  LLM Generator   │
│ - Document Parser  │  - Query Handler │  - Prompt Eng.   │
│ - Text Chunker     │  - Retrieval     │  - Response Gen. │
│ - Embedding Gen.   │  - Context Build │  - Citation Ext. │
└────────────┬───────┴────────┬─────────┴────────┬─────────┘
             │                │                  │
             ▼                ▼                  ▼
┌──────────────────────────────────────────────────────────┐
│                        DATA LAYER                        │
├────────────────────┬──────────────────┬──────────────────┤
│   Vector Database  │   OpenAI API     │  File Storage    │
│   (ChromaDB)       │   - Embeddings   │  - Uploads       │
│   - Embeddings     │   - LLM          │  - Documents     │
│   - Metadata       │   - Completions  │                  │
└────────────────────┴──────────────────┴──────────────────┘
```

---

## System Components

### 1. Document Ingestion Pipeline

**Purpose**: Transform raw documents into searchable vector representations.

**Components**:

#### Document Parser (`src/ingestion/parser.py`)
- Supports: PDF, DOCX, TXT, MD
- Extracts: Text content, metadata (pages, author, title)
- Strategy Pattern: Different parser for each format

#### Text Chunker (`src/ingestion/chunker.py`)
- **Hierarchical Chunking Strategy**:
  1. Split by paragraphs (semantic boundaries)
  2. If chunks too large → split by sentences
  3. If still too large → split by tokens
- **Overlap**: 50 tokens between consecutive chunks
- **Size**: 512 tokens per chunk (optimal for balance)

#### Embedding Generator (`src/retrieval/embeddings.py`)
- Model: `text-embedding-3-small` (1536 dimensions)
- Batching: 100 texts per API call for efficiency
- Caching: Store embeddings to avoid recomputation

### 2. Retrieval System

**Purpose**: Find most relevant document chunks for a given query.

#### Vector Store (`src/retrieval/vectorstore.py`)
- Database: ChromaDB (local, persistent)
- Index: HNSW (Hierarchical Navigable Small World)
- Search: Cosine similarity
- Metadata: Document name, page, chunk index

**Search Process**:
```
1. Query → Embedding (1536D vector)
2. Similarity Search (cosine distance)
3. Top-K Retrieval (default K=5)
4. Rank by relevance score
5. Return chunks + metadata
```

### 3. Generation System

**Purpose**: Generate natural language answers using LLM.

#### LLM Generator (`src/retrieval/llm.py`)
- Model: GPT-4o-mini (128k context, fast, cheap)
- Temperature: 0.1 (deterministic)
- Max Tokens: 1000 (configurable)

**Prompt Engineering**:
```
System: You are a helpful assistant. Answer ONLY from context.
        Include citations. Say "I don't know" if insufficient info.

Context:
[Source 1: doc.pdf, Page 3]
<retrieved chunk 1>

[Source 2: doc.pdf, Page 5]
<retrieved chunk 2>

Question: What is RAG?

Answer:
```

#### Response Post-Processing
- Citation Extraction: Parse source references
- Confidence Calculation: Based on retrieval scores
- Source Deduplication: Unique documents only

### 4. API Layer

**Purpose**: Expose functionality via REST endpoints.

#### FastAPI Backend (`src/api/main.py`)

**Endpoints**:
- `POST /api/documents/upload`: Upload and process documents
- `GET /api/documents`: List indexed documents
- `DELETE /api/documents/{id}`: Remove documents
- `POST /api/query`: Ask questions
- `POST /api/query/stream`: Streaming responses
- `GET /api/stats`: System statistics
- `GET /health`: Health check

**Features**:
- Automatic OpenAPI documentation (Swagger UI)
- Request validation (Pydantic models)
- Error handling and logging
- CORS support for frontend

### 5. Web Interface

**Purpose**: User-friendly interface for document Q&A.

#### Streamlit App (`streamlit_app/app.py`)

**Features**:
- Drag-and-drop document upload
- Real-time query interface
- Source citation display
- Confidence visualization
- System statistics dashboard

---

## Data Flow

### Ingestion Flow
```
User Uploads Document
        │
        ▼
┌───────────────────┐
│  Document Parser  │
│  - Extract text   │
│  - Get metadata   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Text Chunker    │
│  - Semantic split │
│  - Add overlap    │
│  - Create chunks  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Embedding Gen.    │
│  - Batch process  │
│  - Call OpenAI    │
│  - Get vectors    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Vector Store    │
│  - Store vectors  │
│  - Store metadata │
│  - Build index    │
└───────────────────┘
```

### Query Flow
```
User Asks Question
        │
        ▼
┌───────────────────┐
│ Generate Embedding│
│  - Convert query  │
│  - to vector      │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Vector Search    │
│  - Find similar   │
│  - Rank by score  │
│  - Get top-K      │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Build Prompt     │
│  - Add context    │
│  - Format query   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  LLM Generation   │
│  - Call GPT-4o    │
│  - Generate text  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Post-Process      │
│  - Extract cites  │
│  - Calc confidence│
│  - Format result  │
└────────┬──────────┘
         │
         ▼
    Return Answer
```

---

## Technology Stack

### Core Technologies

| Component | Technology | Why? |
|-----------|-----------|------|
| **Backend Framework** | FastAPI | Modern, fast, automatic docs |
| **Frontend Framework** | Streamlit | Rapid prototyping, beautiful UI |
| **LLM** | GPT-4o-mini | Best cost/performance ratio |
| **Embeddings** | text-embedding-3-small | High quality, low cost |
| **Vector Database** | ChromaDB | Easy setup, good for dev/small scale |
| **Orchestration** | LangChain | RAG utilities, connectors |
| **Testing** | pytest | Industry standard, powerful |
| **Code Quality** | black, flake8 | Consistent formatting |

### Alternative Options Considered

**Vector Databases**:
- ❌ Pinecone: Overkill for MVP, requires hosting
- ❌ Weaviate: Complex setup
- ✅ ChromaDB: Simple, local, perfect for demo

**LLM Providers**:
- ❌ Anthropic Claude: More expensive
- ❌ Open-source models: Require GPU hosting
- ✅ OpenAI GPT-4o-mini: Best balance

---

## Design Decisions

### 1. Chunk Size: 512 Tokens

**Why not larger?**
- Too large → poor retrieval precision
- Irrelevant info included in context
- Higher embedding costs

**Why not smaller?**
- Too small → context loss
- Incomplete information
- More chunks needed (slower)

**Optimal**: 512 tokens with 50 token overlap

### 2. Embedding Model: text-embedding-3-small

**Comparison**:
| Model | Dimensions | Cost | Quality |
|-------|-----------|------|---------|
| ada-002 | 1536 | $0.10/1M | Good |
| text-3-small | 1536 | $0.02/1M | Better |
| text-3-large | 3072 | $0.13/1M | Best |

**Choice**: text-3-small offers best cost/quality ratio.

### 3. Top-K: 5 Chunks

**Why 5?**
- Covers ~2500 tokens of context
- Fits comfortably in GPT-4o context window
- Balances precision vs. recall
- Fast retrieval

### 4. Local ChromaDB vs. Cloud Vector DB

**For MVP/Demo**:
- ✅ No external dependencies
- ✅ No hosting costs
- ✅ Easy setup
- ✅ Fast for <100K chunks

**For Production**:
- Migrate to Pinecone or Weaviate
- Horizontal scaling
- Better performance at scale

---

## Scalability Considerations

### Current Limitations

**ChromaDB**:
- Single machine
- Limited to ~1M chunks
- No built-in replication

**Solution**: Migrate to Pinecone/Weaviate for production

### Scaling Strategy

**Phase 1: Current (MVP)**
- ChromaDB local
- Single API instance
- Suitable for: Demo, small teams (<50 users)

**Phase 2: Small Scale (100-1K users)**
- ChromaDB with persistent volume
- Load balancer (2-3 API instances)
- Redis cache for repeated queries
- Estimated: <10K documents, 1K queries/day

**Phase 3: Medium Scale (1K-10K users)**
- Migrate to Pinecone/Weaviate
- Horizontal API scaling (5-10 instances)
- Redis cluster
- CDN for static assets
- Estimated: <100K documents, 10K queries/day

**Phase 4: Large Scale (10K+ users)**
- Fully managed vector DB
- Auto-scaling API (Kubernetes)
- Distributed caching
- Rate limiting
- Monitoring and alerting
- Estimated: 1M+ documents, 100K+ queries/day

### Performance Optimization

**Query Optimization**:
- Cache embeddings (avoid re-computing)
- Batch API calls (100 texts per call)
- Async processing for uploads
- Connection pooling

**Cost Optimization**:
- Use cheaper models where possible
- Implement caching (reduce API calls by 50%)
- Batch similar queries
- Monitor and optimize token usage

---

## Security Considerations

**Current Implementation**:
- API keys in environment variables
- No authentication (demo only)

**Production Requirements**:
- User authentication (OAuth2, JWT)
- API rate limiting
- Input validation and sanitization
- Audit logging
- Encrypted storage
- HTTPS only

---

## Monitoring and Observability

**Metrics to Track**:
- Query latency (P50, P95, P99)
- API availability
- Error rates
- Token usage
- Cost per query
- User activity

**Tools**:
- Prometheus for metrics
- Grafana for dashboards
- Sentry for error tracking
- OpenTelemetry for distributed tracing

---

## Future Enhancements

1. **Multi-tenancy**: Support multiple users with data isolation
2. **Advanced Search**: Filter by date, author, document type
3. **Document Comparison**: Compare multiple documents
4. **Export Features**: Download Q&A history, reports
5. **Integrations**: Slack, Teams, Google Drive
6. **Mobile App**: iOS/Android applications

---

**Last Updated**: January 2026
**Version**: 1.0.0