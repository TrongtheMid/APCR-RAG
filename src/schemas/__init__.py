"""
Package src.schemas: Tập hợp đầy đủ các Pydantic Models của Hệ thống APCR-RAG.
"""

from src.schemas.benchmark_schema import BenchmarkQuestion
from src.schemas.irac_schema import CitationMetadata, IRACOutput
from src.schemas.judge_schema import JudgeVerificationResult
from src.schemas.procedure_schema import (
    CanCuPhapLy,
    GiayToBatBuoc,
    LePhi,
    MaVaTenThuTuc,
    ProcedureRecord,
    ThanhPhanHoSo,
    ThoiHanGiaiQuyet,
)

__all__ = [
    "MaVaTenThuTuc",
    "GiayToBatBuoc",
    "ThanhPhanHoSo",
    "ThoiHanGiaiQuyet",
    "LePhi",
    "CanCuPhapLy",
    "ProcedureRecord",
    "CitationMetadata",
    "IRACOutput",
    "JudgeVerificationResult",
    "BenchmarkQuestion",
]
