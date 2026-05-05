# LAB DAY 19: XÂY DỰNG HỆ THỐNG GRAPHRAG VỚI TECH COMPANY CORPUS

## 1. Data
- [data/](./data/) - Corpus gồm 5 file markdown về các công ty công nghệ

## 2. Code
- [src/](./src/) - Source code chính
- [build_graph.py](./build_graph.py) - Script xây dựng đồ thị tri thức
- [query.py](./query.py) - Script query GraphRAG và Flat RAG
- [run_benchmark.py](./run_benchmark.py) - Script chạy benchmark 20 câu hỏi

## 3. Ảnh Graph
- [GRAPH.png](./GRAPH.png) - Ảnh đồ thị tri thức Knowledge Graph

## 4. Report
- [REPORT.md](./REPORT.md) - Báo cáo phân tích chi tiết kết quả benchmark

## Chạy hệ thống

```bash
# 1. Build graph
python build_graph.py

# 2. Chạy benchmark
python run_benchmark.py

# 3. Xem kết quả
python compare_results.py
```
