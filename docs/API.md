# DocIntel API Documentation

Complete reference for the DocIntel REST API.

## Base URL
```
http://localhost:8000
```

## Authentication

Currently no authentication required (demo version).

For production deployment, implement OAuth2 or JWT authentication.

---

## Endpoints

### Health Check

#### `GET /health`

Check if the API is running.

**Response**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-13T10:30:00.000Z"
}
```

**Status Codes**
- `200`: API is healthy
- `500`: API is experiencing issues

---

### Document Management

#### `POST /api/documents/upload`

Upload and process a document.

**Request**

- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file` (required): Document file (PDF, DOCX, TXT, MD)

**Example**
```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@document.pdf"
```

**Response**
```json
{
  "success": true,
  "document_id": "doc_abc123_20260113",
  "document_name": "document.pdf",
  "num_chunks": 45,
  "message": "Successfully processed 45 chunks"
}
```

**Status Codes**
- `200`: Document successfully processed
- `400`: Invalid file type or format
- `500`: Processing error

**Supported Formats**
- PDF (`.pdf`)
- Word Documents (`.docx`)
- Plain Text (`.txt`)
- Markdown (`.md`)

---

#### `GET /api/documents`

List all indexed documents.

**Response**
```json
{
  "documents": [
    "doc_abc123_20260113",
    "doc_def456_20260113"
  ],
  "total_documents": 2,
  "total_chunks": 87
}
```

**Status Codes**
- `200`: Success
- `500`: Server error

---

#### `DELETE /api/documents/{document_id}`

Delete a document and all its chunks.

**Parameters**
- `document_id` (path, required): ID of document to delete

**Example**
```bash
curl -X DELETE "http://localhost:8000/api/documents/doc_abc123_20260113"
```

**Response**
```json
{
  "success": true,
  "message": "Deleted 45 chunks from document doc_abc123_20260113"
}
```

**Status Codes**
- `200`: Document deleted
- `404`: Document not found
- `500`: Deletion error

---

### Query

#### `POST /api/query`

Ask a question and get an answer with sources.

**Request Body**
```json
{
  "question": "What is machine learning?",
  "n_results": 5,
  "filters": null,
  "include_context": false
}
```

**Parameters**
- `question` (required, string): Question to ask (1-1000 chars)
- `n_results` (optional, integer): Number of chunks to retrieve (1-20, default: 5)
- `filters` (optional, object): Metadata filters for retrieval
- `include_context` (optional, boolean): Include retrieved chunks in response

**Example**
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "n_results": 5
  }'
```

**Response**
```json
{
  "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. [Source: ml_guide.pdf, Page: 3]",
  "sources": [
    {
      "document_name": "ml_guide.pdf",
      "page_number": 3,
      "relevance_score": 0.89
    },
    {
      "document_name": "ai_basics.pdf",
      "page_number": 12,
      "relevance_score": 0.85
    }
  ],
  "confidence": 0.87,
  "model": "gpt-4o-mini",
  "tokens_used": 456,
  "processing_time_ms": 2340,
  "timestamp": "2026-01-13T10:35:22.123Z"
}
```

**Response Fields**
- `answer`: Generated answer with citations
- `sources`: Array of source citations
  - `document_name`: Name of source document
  - `page_number`: Page number (or "N/A")
  - `relevance_score`: Similarity score (0-1)
- `confidence`: Overall confidence score (0-1)
- `model`: LLM model used
- `tokens_used`: Total tokens consumed
- `processing_time_ms`: Time taken in milliseconds
- `timestamp`: ISO 8601 timestamp

**Status Codes**
- `200`: Query successful
- `400`: Invalid request (e.g., empty question)
- `500`: Query processing error

---

#### `POST /api/query/stream`

Ask a question with streaming response (Server-Sent Events).

**Request Body**

Same as `/api/query`

**Example**
```bash
curl -X POST "http://localhost:8000/api/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI?"}' \
  --no-buffer
```

**Response**

Server-Sent Events stream:
```
data: {"chunk": "Artificial "}

data: {"chunk": "intelligence "}

data: {"chunk": "is..."}

data: {"done": true}
```

**Status Codes**
- `200`: Streaming started
- `400`: Invalid request
- `500`: Streaming error

---

### System Statistics

#### `GET /api/stats`

Get system statistics.

**Response**
```json
{
  "total_documents": 15,
  "total_chunks": 678,
  "total_queries": 234,
  "avg_confidence": 0.82,
  "documents": [
    "doc_abc123_20260113",
    "doc_def456_20260113"
  ],
  "uptime_seconds": 3600.5
}
```

**Status Codes**
- `200`: Success
- `500`: Error retrieving stats

---

## Error Responses

All errors follow this format:
```json
{
  "error": "Brief error message",
  "detail": "Detailed error information",
  "timestamp": "2026-01-13T10:30:00.000Z"
}
```

### Common Error Codes

| Status | Meaning | Common Causes |
|--------|---------|---------------|
| 400 | Bad Request | Invalid input, missing required fields |
| 404 | Not Found | Document ID doesn't exist |
| 422 | Validation Error | Pydantic validation failed |
| 500 | Internal Server Error | Unexpected server error |
| 503 | Service Unavailable | OpenAI API down, rate limited |

---

## Rate Limiting

Currently no rate limiting (demo version).

For production, implement:
- 100 requests per minute per IP
- 1000 requests per hour per IP
- Burst allowance: 10 requests

---

## Interactive Documentation

FastAPI provides interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to:
- View all endpoints
- See request/response schemas
- Try API calls directly
- Download OpenAPI specification

---

## Code Examples

### Python
```python
import requests

API_URL = "http://localhost:8000"

# Upload document
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{API_URL}/api/documents/upload", files=files)
    doc_id = response.json()["document_id"]

# Query
query = {
    "question": "What is the main topic?",
    "n_results": 5
}
response = requests.post(f"{API_URL}/api/query", json=query)
answer = response.json()["answer"]
print(answer)
```

### JavaScript
```javascript
const API_URL = "http://localhost:8000";

// Upload document
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const uploadResponse = await fetch(`${API_URL}/api/documents/upload`, {
  method: "POST",
  body: formData
});
const uploadData = await uploadResponse.json();

// Query
const queryResponse = await fetch(`${API_URL}/api/query`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question: "What is the main topic?",
    n_results: 5
  })
});
const queryData = await queryResponse.json();
console.log(queryData.answer);
```

### cURL
```bash
# Upload
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@document.pdf"

# Query
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'

# Delete
curl -X DELETE "http://localhost:8000/api/documents/doc_abc123"

# Stats
curl http://localhost:8000/api/stats
```

---

## Webhooks (Future Feature)

Coming in v1.1:
- Document processing completion notifications
- Query result callbacks
- System alerts

---

## Changelog

### v1.0.0 (January 2026)
- Initial API release
- Document upload and management
- Query endpoints
- Statistics endpoint
- Health check

---

## Support

For API issues or questions:
- GitHub Issues: [github.com/pranavpawar2103/docintel/issues](https://github.com/pranavpawar2103/docintel/issues)
- Email: pranavpawar2126@gmail.com

---

**Last Updated**: January 2026  
**API Version**: 1.0.0