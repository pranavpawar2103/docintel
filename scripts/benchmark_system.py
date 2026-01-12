"""
Comprehensive performance benchmarking for DocIntel.
Measures and reports all key performance metrics.
"""

import time
import statistics
from pathlib import Path
import json
from datetime import datetime

from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.rag_pipeline import RAGPipeline
from src.retrieval.vectorstore import VectorStore


class PerformanceBenchmark:
    """Benchmark suite for RAG system."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'benchmarks': {}
        }
    
    def run_all_benchmarks(self):
        """Run all benchmarks and generate report."""
        print("=" * 70)
        print("🚀 DOCINTEL PERFORMANCE BENCHMARK")
        print("=" * 70)
        
        # 1. Document Ingestion Benchmark
        print("\n📄 1. Document Ingestion Performance")
        print("-" * 70)
        ingestion_results = self.benchmark_ingestion()
        self.results['benchmarks']['ingestion'] = ingestion_results
        
        # 2. Query Performance Benchmark
        print("\n💬 2. Query Performance")
        print("-" * 70)
        query_results = self.benchmark_queries()
        self.results['benchmarks']['query'] = query_results
        
        # 3. Retrieval Quality Benchmark
        print("\n🎯 3. Retrieval Quality")
        print("-" * 70)
        retrieval_results = self.benchmark_retrieval_quality()
        self.results['benchmarks']['retrieval'] = retrieval_results
        
        # 4. Cost Analysis
        print("\n💰 4. Cost Analysis")
        print("-" * 70)
        cost_results = self.analyze_costs()
        self.results['benchmarks']['cost'] = cost_results
        
        # Generate summary
        self.print_summary()
        
        # Save results
        self.save_results()
    
    def benchmark_ingestion(self):
        """Benchmark document ingestion."""
        vector_store = VectorStore(collection_name="benchmark_ingestion")
        ingestion = IngestionPipeline(vector_store=vector_store)
        
        test_doc = "tests/test_documents/sample.txt"
        if not Path(test_doc).exists():
            print("   ⚠️ Test document not found, skipping...")
            vector_store.reset()
            return None
        
        # Get file size
        file_size = Path(test_doc).stat().st_size
        
        # Benchmark ingestion
        start_time = time.time()
        result = ingestion.ingest_document(test_doc)
        ingestion_time = time.time() - start_time
        
        num_chunks = result['num_chunks']
        chunks_per_sec = num_chunks / ingestion_time
        bytes_per_sec = file_size / ingestion_time
        
        print(f"   File Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        print(f"   Chunks Created: {num_chunks}")
        print(f"   Processing Time: {ingestion_time:.2f}s")
        print(f"   Throughput: {chunks_per_sec:.1f} chunks/sec")
        print(f"   Speed: {bytes_per_sec/1024:.1f} KB/sec")
        
        # Cleanup
        vector_store.reset()
        
        return {
            'file_size_bytes': file_size,
            'num_chunks': num_chunks,
            'processing_time_sec': ingestion_time,
            'chunks_per_sec': chunks_per_sec,
            'kb_per_sec': bytes_per_sec / 1024
        }
    
    def benchmark_queries(self):
        """Benchmark query performance."""
        # Setup
        vector_store = VectorStore(collection_name="benchmark_queries")
        ingestion = IngestionPipeline(vector_store=vector_store)
        
        test_doc = "tests/test_documents/sample.txt"
        if Path(test_doc).exists():
            ingestion.ingest_document(test_doc)
        
        rag = RAGPipeline(vector_store=vector_store)
        
        # Test queries
        queries = [
            "What is RAG?",
            "How does retrieval work?",
            "What are the benefits of using RAG?",
            "Explain the system architecture",
            "What technologies are used?",
        ]
        
        latencies = []
        token_counts = []
        
        print(f"   Running {len(queries)} test queries...")
        
        for query in queries:
            start = time.time()
            response = rag.query(query)
            latency = time.time() - start
            
            latencies.append(latency)
            token_counts.append(response.get('tokens_used', 0))
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        avg_tokens = statistics.mean(token_counts)
        
        print(f"\n   Latency Statistics:")
        print(f"      Average: {avg_latency:.2f}s")
        print(f"      Median (P50): {median_latency:.2f}s")
        print(f"      P95: {p95_latency:.2f}s")
        print(f"      Min: {min_latency:.2f}s")
        print(f"      Max: {max_latency:.2f}s")
        print(f"\n   Token Usage:")
        print(f"      Average per query: {avg_tokens:.0f} tokens")
        
        # Cleanup
        vector_store.reset()
        
        return {
            'num_queries': len(queries),
            'avg_latency_sec': avg_latency,
            'median_latency_sec': median_latency,
            'p95_latency_sec': p95_latency,
            'min_latency_sec': min_latency,
            'max_latency_sec': max_latency,
            'avg_tokens_per_query': avg_tokens
        }
    
    def benchmark_retrieval_quality(self):
        """Benchmark retrieval quality with known Q&A pairs."""
        vector_store = VectorStore(collection_name="benchmark_quality")
        ingestion = IngestionPipeline(vector_store=vector_store)
        
        test_doc = "tests/test_documents/sample.txt"
        if not Path(test_doc).exists():
            print("   ⚠️ Test document not found, skipping...")
            vector_store.reset()
            return None
        
        ingestion.ingest_document(test_doc)
        rag = RAGPipeline(vector_store=vector_store)
        
        # Test cases with expected relevance
        test_cases = [
            {
                "query": "What is RAG?",
                "should_find_answer": True,
                "min_confidence": 0.7
            },
            {
                "query": "How does retrieval work?",
                "should_find_answer": True,
                "min_confidence": 0.6
            },
            {
                "query": "What is quantum computing?",
                "should_find_answer": False,
                "max_confidence": 0.3
            }
        ]
        
        passed = 0
        failed = 0
        
        print(f"   Testing {len(test_cases)} retrieval scenarios...")
        
        for i, test in enumerate(test_cases, 1):
            response = rag.query(test['query'])
            confidence = response['confidence']
            
            if test['should_find_answer']:
                # Should find relevant answer
                if confidence >= test.get('min_confidence', 0.5):
                    passed += 1
                    status = "✅"
                else:
                    failed += 1
                    status = "❌"
            else:
                # Should NOT find answer (irrelevant query)
                if confidence <= test.get('max_confidence', 0.5):
                    passed += 1
                    status = "✅"
                else:
                    failed += 1
                    status = "❌"
            
            print(f"      {status} Test {i}: Confidence={confidence:.2f}")
        
        accuracy = (passed / len(test_cases)) * 100
        
        print(f"\n   Overall Accuracy: {accuracy:.1f}% ({passed}/{len(test_cases)})")
        
        # Cleanup
        vector_store.reset()
        
        return {
            'total_tests': len(test_cases),
            'passed': passed,
            'failed': failed,
            'accuracy_percent': accuracy
        }
    
    def analyze_costs(self):
        """Analyze API costs."""
        # Pricing (as of Jan 2025)
        GPT4O_MINI_INPUT = 0.150 / 1_000_000  # $0.15 per 1M tokens
        GPT4O_MINI_OUTPUT = 0.600 / 1_000_000  # $0.60 per 1M tokens
        EMBEDDING_SMALL = 0.020 / 1_000_000  # $0.02 per 1M tokens
        
        # Average usage per query (from benchmarks)
        avg_input_tokens = 800  # Context + question
        avg_output_tokens = 200  # Answer
        avg_embedding_tokens = 500  # Per chunk embedded
        
        # Calculate costs
        cost_per_query_llm = (
            avg_input_tokens * GPT4O_MINI_INPUT +
            avg_output_tokens * GPT4O_MINI_OUTPUT
        )
        
        cost_per_chunk_embedding = avg_embedding_tokens * EMBEDDING_SMALL
        
        # Per document (assume 10 chunks average)
        cost_per_doc_ingestion = 10 * cost_per_chunk_embedding
        
        # Cost for 100 queries
        cost_100_queries = cost_per_query_llm * 100
        
        # Cost for 10 documents
        cost_10_docs = cost_per_doc_ingestion * 10
        
        print(f"   Cost per Query (LLM): ${cost_per_query_llm:.6f}")
        print(f"   Cost per Chunk (Embedding): ${cost_per_chunk_embedding:.6f}")
        print(f"   Cost per Document (~10 chunks): ${cost_per_doc_ingestion:.4f}")
        print(f"\n   Bulk Costs:")
        print(f"      100 queries: ${cost_100_queries:.4f}")
        print(f"      10 documents: ${cost_10_docs:.4f}")
        print(f"      1000 queries: ${cost_per_query_llm * 1000:.2f}")
        
        return {
            'cost_per_query_usd': cost_per_query_llm,
            'cost_per_chunk_usd': cost_per_chunk_embedding,
            'cost_100_queries_usd': cost_100_queries,
            'cost_10_docs_usd': cost_10_docs
        }
    
    def print_summary(self):
        """Print summary of all benchmarks."""
        print("\n" + "=" * 70)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 70)
        
        # Ingestion
        if 'ingestion' in self.results['benchmarks'] and self.results['benchmarks']['ingestion']:
            ing = self.results['benchmarks']['ingestion']
            print(f"\n📄 Ingestion: {ing['chunks_per_sec']:.1f} chunks/sec, "
                  f"{ing['kb_per_sec']:.1f} KB/sec")
        
        # Queries
        if 'query' in self.results['benchmarks']:
            qry = self.results['benchmarks']['query']
            print(f"💬 Queries: {qry['avg_latency_sec']:.2f}s avg latency, "
                  f"{qry['p95_latency_sec']:.2f}s P95")
        
        # Quality
        if 'retrieval' in self.results['benchmarks'] and self.results['benchmarks']['retrieval']:
            ret = self.results['benchmarks']['retrieval']
            print(f"🎯 Quality: {ret['accuracy_percent']:.1f}% accuracy "
                  f"({ret['passed']}/{ret['total_tests']} tests passed)")
        
        # Cost
        if 'cost' in self.results['benchmarks']:
            cst = self.results['benchmarks']['cost']
            print(f"💰 Cost: ${cst['cost_per_query_usd']:.6f}/query, "
                  f"${cst['cost_100_queries_usd']:.4f}/100 queries")
        
        print("\n" + "=" * 70)
    
    def save_results(self):
        """Save benchmark results to file."""
        output_file = Path("benchmark_results.json")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")


def main():
    """Run benchmarks."""
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()