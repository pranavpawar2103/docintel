# DocIntel Configuration Guide

Detailed guide for configuring DocIntel for different use cases.

## Table of Contents

- [Environment Variables](#environment-variables)
- [Model Selection](#model-selection)
- [Chunking Strategy](#chunking-strategy)
- [Retrieval Tuning](#retrieval-tuning)
- [Performance Optimization](#performance-optimization)
- [Production Setup](#production-setup)

---

## Environment Variables

### Required Variables

#### OPENAI_API_KEY

**Description**: Your OpenAI API key for embeddings and LLM.

**How to get:**
1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Copy and paste into `.env`

**Security:**
- Never commit to git
- Rotate regularly
- Monitor usage for anomalies

---

### Model Configuration

#### EMBEDDING_MODEL

**Options:**

| Model | Dimensions | Cost | Performance | Use Case |
|-------|-----------|------|-------------|----------|
| text-embedding-3-small | 1536 | $0.02/1M | Excellent | **Recommended** |
| text-embedding-3-large | 3072 | $0.13/1M | Best | High precision needed |
| text-embedding-ada-002 | 1536 | $0.10/1M | Good | Legacy projects |

**Recommendation**: `text-embedding-3-small` offers best value.

**When to use large:**
- Very technical documents
- Need highest possible accuracy
- Cost is not a concern

#### LLM_MODEL

**Options:**

| Model | Context | Cost | Speed | Use Case |
|-------|---------|------|-------|----------|
| gpt-4o-mini | 128k | $0.15/1M in | Fast | **Recommended** |
| gpt-4o | 128k | $2.50/1M in | Fast | High quality needed |
| gpt-4-turbo | 128k | $10/1M in | Medium | Legacy |
| gpt-3.5-turbo | 16k | $0.50/1M in | Fastest | Simple queries |

**Recommendation**: `gpt-4o-mini` for best cost/performance.

**Cost comparison** (1000 queries):
- gpt-4o-mini: ~$0.12
- gpt-4o: ~$2.00
- gpt-4-turbo: ~$8.00

---

## Chunking Strategy

### CHUNK_SIZE

**Impact on system:**

| Size | Pros | Cons | Use Case |
|------|------|------|----------|
| 256 | Precise retrieval | Less context | Short Q&A |
| 512 | **Balanced** | - | **Recommended** |
| 1024 | More context | Less precise | Long-form |
| 2048 | Maximum context | Slow, expensive | Rare |

**How to choose:**
```python
# Short, factual documents
CHUNK_SIZE=256

# Technical docs, papers (recommended)
CHUNK_SIZE=512

# Books, long-form content
CHUNK_SIZE=1024
```

### CHUNK_OVERLAP

**Purpose**: Prevent information loss at chunk boundaries.

**Rules of thumb:**
- 10% of chunk size minimum
- 50 tokens = ~10-15 words
- More overlap = safer but more storage

**Examples:**
```bash
CHUNK_SIZE=512
CHUNK_OVERLAP=50  # ~10%, recommended

CHUNK_SIZE=1024
CHUNK_OVERLAP=100  # ~10%
```

---

## Retrieval Tuning

### TOP_K_RESULTS

**Impact:**

| K | Context Size | Speed | Cost | Accuracy |
|---|-------------|-------|------|----------|
| 3 | ~1500 tokens | Fast | Low | Good |
| 5 | ~2500 tokens | Medium | Medium | **Best** |
| 10 | ~5000 tokens | Slow | High | Diminishing |

**When to adjust:**
```bash
# Simple factual queries
TOP_K_RESULTS=3

# Complex questions requiring synthesis (recommended)
TOP_K_RESULTS=5

# Very broad questions
TOP_K_RESULTS=8
```

### SIMILARITY_THRESHOLD

**Purpose**: Filter out irrelevant chunks.

**Scale:**
- 0.9+: Very strict (only near-exact matches)
- 0.7-0.9: Moderate (recommended)
- 0.5-0.7: Permissive
- <0.5: Too loose (noise)

**Tuning guide:**
```bash
# High precision required
SIMILARITY_THRESHOLD=0.8

# Balanced (recommended)
SIMILARITY_THRESHOLD=0.7

# Recall more important
SIMILARITY_THRESHOLD=0.6
```

---

## Performance Optimization

### For Cost Optimization
```bash
# Use cheapest models
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini

# Reduce retrieval
TOP_K_RESULTS=3
CHUNK_SIZE=512

# Implement caching (code changes needed)
# Cache repeated queries
# Cache embeddings for uploaded docs
```

**Expected savings**: 50-70% cost reduction

### For Speed Optimization
```bash
# Fast models
LLM_MODEL=gpt-4o-mini

# Smaller retrieval
TOP_K_RESULTS=3

# Larger chunks (less processing)
CHUNK_SIZE=1024

# Batch uploads asynchronously
```

**Expected improvement**: 30-40% faster

### For Accuracy Optimization
```bash
# Best models
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-4o

# More context
TOP_K_RESULTS=8
CHUNK_SIZE=512
CHUNK_OVERLAP=75

# Higher threshold
SIMILARITY_THRESHOLD=0.75
```

**Expected improvement**: 10-15% accuracy increase

---

## Production Setup

### Environment-Specific Configs

**Development** (`.env.dev`):
```bash
OPENAI_API_KEY=sk-dev-key
LLM_MODEL=gpt-4o-mini
LOG_LEVEL=DEBUG
API_DEBUG=true
```

**Staging** (`.env.staging`):
```bash
OPENAI_API_KEY=sk-staging-key
LLM_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
API_DEBUG=false
```

**Production** (`.env.prod`):
```bash
OPENAI_API_KEY=sk-prod-key
LLM_MODEL=gpt-4o-mini
LOG_LEVEL=WARNING
API_DEBUG=false
# Additional: Rate limiting, monitoring, backups
```

### Security Best Practices

**API Keys:**
```bash
# ❌ Never do this
git add .env

# ✅ Do this
echo ".env" >> .gitignore

# ✅ Use secret management
# AWS: AWS Secrets Manager
# Azure: Azure Key Vault
# GCP: Secret Manager
```

**Access Control:**
```python
# Add to production:
# - JWT authentication
# - API rate limiting
# - IP whitelisting
# - Audit logging
```

### Monitoring

**Metrics to track:**
```python
# Add to production:
# - Query latency (P50, P95, P99)
# - Error rates
# - Token usage
# - Cost per query
# - User activity
# - System resources
```

**Tools:**
- Prometheus + Grafana
- DataDog
- New Relic
- OpenTelemetry

---

## Troubleshooting

### High Costs

**Symptoms**: API bills higher than expected

**Solutions:**
1. Check `TOP_K_RESULTS` - reduce if too high
2. Implement caching for repeated queries
3. Use `gpt-4o-mini` instead of `gpt-4o`
4. Monitor token usage per query

### Slow Queries

**Symptoms**: Queries taking >5 seconds

**Solutions:**
1. Reduce `TOP_K_RESULTS`
2. Use faster model (`gpt-4o-mini`)
3. Optimize chunk size
4. Check network latency to OpenAI
5. Implement Redis caching

### Low Accuracy

**Symptoms**: Answers missing key information

**Solutions:**
1. Increase `TOP_K_RESULTS` to 8-10
2. Reduce `SIMILARITY_THRESHOLD` to 0.6
3. Reduce `CHUNK_SIZE` for better precision
4. Increase `CHUNK_OVERLAP`
5. Try `text-embedding-3-large`

### Out of Memory

**Symptoms**: Process killed, memory errors

**Solutions:**
1. Process documents in batches
2. Reduce `EMBEDDING_BATCH_SIZE`
3. Clear ChromaDB cache periodically
4. Use streaming for large responses

---

## Advanced Configuration

### Custom Prompts

Edit `src/retrieval/llm.py`:
```python
# Customize system prompt for domain
SYSTEM_PROMPT = """
You are a medical document assistant.
Only answer medical questions from the context.
Include source citations and confidence.
"""
```

### Custom Chunking

Edit `src/ingestion/chunker.py`:
```python
# Custom chunking for code
def chunk_code(text: str) -> List[str]:
    # Split by functions/classes
    return custom_split(text)
```

### Database Configuration

For production vector databases:

**Pinecone:**
```bash
VECTOR_DB=pinecone
PINECONE_API_KEY=your-key
PINECONE_ENVIRONMENT=us-west1-gcp
```

**Weaviate:**
```bash
VECTOR_DB=weaviate
WEAVIATE_URL=http://localhost:8080
```

---

## Configuration Examples

### Academic Research
```bash
# High accuracy for research papers
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-4o
CHUNK_SIZE=512
CHUNK_OVERLAP=75
TOP_K_RESULTS=8
SIMILARITY_THRESHOLD=0.75
```

### Customer Support
```bash
# Fast, cheap for high volume
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
CHUNK_SIZE=256
CHUNK_OVERLAP=25
TOP_K_RESULTS=3
SIMILARITY_THRESHOLD=0.65
```

### Legal Documents
```bash
# Precise, comprehensive
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-4o
CHUNK_SIZE=1024
CHUNK_OVERLAP=100
TOP_K_RESULTS=10
SIMILARITY_THRESHOLD=0.8
```

---

**Last Updated**: January 2026
```
