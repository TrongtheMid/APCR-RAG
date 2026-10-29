"""
Facade tương thích ngược cho src.schemas (đảm bảo mọi lệnh `from src.schema import ...` đều hoạt động).
"""

from src.schemas import (
    BenchmarkQuestion,
    CanCuPhapLy,
    CitationMetadata,
    GiayToBatBuoc,
    IRACOutput,
    JudgeVerificationResult,
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
