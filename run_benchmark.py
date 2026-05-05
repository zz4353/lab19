"""
Script chay benchmark don gian - So sanh Flat RAG vs GraphRAG
"""
import json
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

from src.config import Config, validate_config
from src.data_loader import DataLoader
from src.graph_builder import GraphBuilder
from src.query_engine import GraphRAGQueryEngine, FlatRAGQueryEngine

load_dotenv()


def load_test_questions(file_path="test_questions.json"):
    """Load cau hoi test tu file JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Flatten tat ca cau hoi tu cac nhom
    all_questions = []
    for group in data['question_groups']:
        all_questions.extend(group['questions'])
    
    return all_questions


def initialize_systems():
    """Khoi tao GraphRAG va Flat RAG"""
    print("Dang khoi tao he thong...")
    
    # Load config
    try:
        config = Config.from_yaml('config.yaml')
    except FileNotFoundError:
        config = Config.from_env()
    
    validate_config(config)
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=config.openai_api_key)
    
    # Initialize GraphRAG
    print("  - Khoi tao GraphRAG...")
    graph_builder = GraphBuilder(
        neo4j_uri=config.neo4j_uri,
        neo4j_user=config.neo4j_user,
        neo4j_password=config.neo4j_password
    )
    
    graphrag_engine = GraphRAGQueryEngine(
        graph_builder=graph_builder,
        llm_client=openai_client,
        model=config.openai_model
    )
    
    # Initialize Flat RAG
    print("  - Khoi tao Flat RAG...")
    flatrag_engine = FlatRAGQueryEngine(
        llm_client=openai_client,
        model=config.openai_model,
        embedding_model=config.embedding_model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        top_k=config.top_k
    )
    
    # Index documents for Flat RAG
    print("  - Dang index documents cho Flat RAG...")
    data_loader = DataLoader(data_dir=config.data_dir)
    documents = data_loader.load_all_markdown_files()
    flatrag_engine.index_documents(documents)
    
    print(f"Hoan tat khoi tao! Da index {len(documents)} documents\n")
    
    return graphrag_engine, flatrag_engine, graph_builder


def run_single_question(question_data, engine, system_name):
    """Chay mot cau hoi tren mot engine"""
    question = question_data['question']
    
    print(f"\n{'='*70}")
    print(f"Q{question_data['id']}: {question}")
    print(f"System: {system_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        result = engine.query(question)
        response_time = time.time() - start_time
        
        print(f"\nCau tra loi:")
        print(result.answer)
        print(f"\nDap an chuan:")
        print(question_data['answer'])
        print(f"\nThoi gian: {response_time:.2f}s")
        print(f"Token: {result.token_usage}")
        
        return {
            "question_id": question_data['id'],
            "question": question,
            "expected_answer": question_data['answer'],
            "actual_answer": result.answer,
            "response_time": response_time,
            "token_usage": result.token_usage,
            "success": True
        }
        
    except Exception as e:
        response_time = time.time() - start_time
        print(f"\nLoi: {str(e)}")
        
        return {
            "question_id": question_data['id'],
            "question": question,
            "expected_answer": question_data['answer'],
            "actual_answer": f"ERROR: {str(e)}",
            "response_time": response_time,
            "token_usage": 0,
            "success": False,
            "error": str(e)
        }


def run_benchmark(questions, graphrag_engine, flatrag_engine):
    """Chay benchmark cho tat ca cau hoi"""
    results = {
        "metadata": {
            "test_date": datetime.now().isoformat(),
            "total_questions": len(questions)
        },
        "graphrag_results": [],
        "flatrag_results": []
    }
    
    # Chay GraphRAG
    print("\n" + "="*70)
    print("PHAN 1: CHAY GRAPHRAG")
    print("="*70)
    
    for q in questions:
        result = run_single_question(q, graphrag_engine, "GraphRAG")
        results["graphrag_results"].append(result)
    
    # Chay Flat RAG
    print("\n" + "="*70)
    print("PHAN 2: CHAY FLAT RAG")
    print("="*70)
    
    for q in questions:
        result = run_single_question(q, flatrag_engine, "Flat RAG")
        results["flatrag_results"].append(result)
    
    return results


def print_summary(results):
    """In tom tat ket qua"""
    print("\n" + "="*70)
    print("TOM TAT KET QUA")
    print("="*70)
    
    for system_name, system_results in [("GraphRAG", results["graphrag_results"]), 
                                         ("Flat RAG", results["flatrag_results"])]:
        print(f"\n{system_name}:")
        print("-" * 70)
        
        total = len(system_results)
        success = sum(1 for r in system_results if r.get('success', False))
        avg_time = sum(r['response_time'] for r in system_results) / total
        total_tokens = sum(r['token_usage'] for r in system_results)
        avg_tokens = total_tokens / total
        
        print(f"  Tong cau hoi: {total}")
        print(f"  Thanh cong: {success}/{total}")
        print(f"  Thoi gian TB: {avg_time:.2f}s")
        print(f"  Token TB: {avg_tokens:.0f}")
        print(f"  Tong token: {total_tokens}")


def save_results(results, output_file="benchmark_results.json"):
    """Luu ket qua vao file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nDa luu ket qua vao: {output_file}")


def main():
    """Main function"""
    print("""
==================================================================
        GRAPHRAG BENCHMARK - TECH COMPANY CORPUS                
        So sanh Flat RAG vs GraphRAG - 20 cau hoi               
==================================================================
    """)
    
    # Load test questions
    print("Dang load cau hoi test...")
    questions = load_test_questions()
    print(f"Da load {len(questions)} cau hoi\n")
    
    # Initialize systems
    graphrag_engine, flatrag_engine, graph_builder = initialize_systems()
    
    # Run benchmark
    results = run_benchmark(questions, graphrag_engine, flatrag_engine)
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(results, output_file)
    
    # Cleanup
    graph_builder.close()
    
    print("\nHoan thanh benchmark!")


if __name__ == "__main__":
    main()
