"""
Package src.evaluation (Đánh giá Thực nghiệm Định lượng trên Bộ 50 câu hỏi Benchmark,
Ablation Study 4 cấu hình, Confusion Matrix định tuyến và tích hợp RAGAS).
"""

from src.evaluation.benchmark_runner import (
    compute_routing_confusion_matrix,
    evaluate_self_revision_rate,
    evaluate_with_ragas_optional,
    load_benchmark_questions,
    run_ablation_study,
    run_full_benchmark_evaluation,
)

__all__ = [
    "load_benchmark_questions",
    "compute_routing_confusion_matrix",
    "evaluate_self_revision_rate",
    "evaluate_with_ragas_optional",
    "run_ablation_study",
    "run_full_benchmark_evaluation",
]
