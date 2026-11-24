"""
Quản lý hàng đợi kiểm duyệt (Staging Queue) và gộp an toàn vào `data/data_procedures.json`:
- Sao lưu tự động có dấu thời gian trước mọi lần cập nhật kho chính.
- Kiểm định 100% qua Pydantic `ProcedureRecord`.
- Làm mới nóng RetrieverAgent (BM25 + ChromaDB) ngay sau khi duyệt.
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import DEFAULT_DATA_PATH, PROJECT_ROOT
from src.ingestion.models import ExtractedProcedureCandidate, StagingQueue
from src.retriever.agent import reset_retriever_agent
from src.schemas.procedure_schema import ProcedureRecord

DEFAULT_STAGING_PATH = PROJECT_ROOT / "data" / "staging_procedures.json"
DEFAULT_BACKUPS_DIR = PROJECT_ROOT / "data" / "backups"


def load_staging_queue(staging_path: Path | str = DEFAULT_STAGING_PATH) -> StagingQueue:
    """Đọc hàng đợi kiểm duyệt từ đĩa (trả về hàng đợi rỗng nếu chưa tồn tại hoặc hỏng)."""
    path = Path(staging_path)
    if not path.exists():
        return StagingQueue()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return StagingQueue.model_validate(data)
    except Exception:
        return StagingQueue()


def save_staging_queue(queue: StagingQueue, staging_path: Path | str = DEFAULT_STAGING_PATH) -> None:
    """Lưu hàng đợi kiểm duyệt ra tệp JSON an toàn."""
    path = Path(staging_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(queue.model_dump(), f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def add_candidates_to_staging(
    candidates: List[ExtractedProcedureCandidate],
    source_summary: str = "",
    staging_path: Path | str = DEFAULT_STAGING_PATH,
) -> Dict[str, int]:
    """Thêm ứng viên vào hàng đợi Staging (khử trùng lặp ứng viên PENDING cùng mã & số hiệu)."""
    queue = load_staging_queue(staging_path)
    existing_keys = {
        (
            c.action,
            c.record.ma_va_ten_thu_tuc.ma_thu_tuc,
            c.source_so_hieu,
        )
        for c in queue.candidates
        if c.status == "PENDING"
    }

    added = 0
    skipped = 0
    for cand in candidates:
        key = (cand.action, cand.record.ma_va_ten_thu_tuc.ma_thu_tuc, cand.source_so_hieu)
        if key in existing_keys:
            skipped += 1
            continue
        existing_keys.add(key)
        queue.candidates.insert(0, cand)
        added += 1

    queue.last_sync_at = datetime.datetime.now().isoformat(timespec="seconds")
    if source_summary:
        queue.last_source_summary = source_summary
    save_staging_queue(queue, staging_path=staging_path)
    return {"added": added, "skipped": skipped, "total_pending": sum(1 for c in queue.candidates if c.status == "PENDING")}


def _create_backup(data_path: Path, backups_dir: Path) -> Optional[Path]:
    if not data_path.exists():
        return None
    backups_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backups_dir / f"data_procedures_{ts}.json"
    shutil.copy2(data_path, backup_path)
    return backup_path


def approve_and_merge_candidate(
    candidate_id: str,
    edited_record_dict: Optional[Dict[str, Any]] = None,
    data_path: Path | str = DEFAULT_DATA_PATH,
    staging_path: Path | str = DEFAULT_STAGING_PATH,
    backups_dir: Path | str = DEFAULT_BACKUPS_DIR,
    reload_retriever: bool = True,
) -> Dict[str, Any]:
    """
    Duyệt 1 ứng viên từ Staging và ghi vào kho chính (`data_procedures.json`):
    - Kiểm định qua `ProcedureRecord.model_validate`
    - Tạo bản sao lưu trước khi ghi
    - Hỗ trợ ADD, UPDATE_EXISTING và UPDATE_EXPIRY
    - Làm mới nóng RetrieverAgent (nếu `reload_retriever=True`)
    """
    queue = load_staging_queue(staging_path)
    target_cand: Optional[ExtractedProcedureCandidate] = None
    for c in queue.candidates:
        if c.candidate_id == candidate_id:
            target_cand = c
            break

    if target_cand is None:
        raise ValueError(f"Không tìm thấy mục Staging có mã: {candidate_id}")

    final_record = (
        ProcedureRecord.model_validate(edited_record_dict)
        if edited_record_dict is not None
        else ProcedureRecord.model_validate(target_cand.record.model_dump())
    )
    final_dict = final_record.model_dump()
    ma_tt = final_dict["ma_va_ten_thu_tuc"]["ma_thu_tuc"]

    d_path = Path(data_path)
    if not d_path.exists():
        raise FileNotFoundError(f"Không tìm thấy kho dữ liệu tại: {d_path}")

    with open(d_path, "r", encoding="utf-8") as f:
        current_records: List[Dict[str, Any]] = json.load(f)

    backup_path = _create_backup(d_path, Path(backups_dir))

    lookup_code = target_cand.target_existing_code or ma_tt
    replaced = False
    for idx, proc in enumerate(current_records):
        if proc["ma_va_ten_thu_tuc"]["ma_thu_tuc"] == lookup_code:
            current_records[idx] = final_dict
            replaced = True
            break

    if not replaced:
        current_records.append(final_dict)

    tmp_data = d_path.with_suffix(".tmp")
    with open(tmp_data, "w", encoding="utf-8") as f:
        json.dump(current_records, f, ensure_ascii=False, indent=2)
    tmp_data.replace(d_path)

    target_cand.record = final_record
    target_cand.status = "APPROVED"
    save_staging_queue(queue, staging_path=staging_path)

    if reload_retriever:
        reset_retriever_agent()

    return {
        "merged_code": ma_tt,
        "action": target_cand.action,
        "replaced_existing": replaced,
        "total_procedures": len(current_records),
        "backup_path": str(backup_path) if backup_path else "",
    }


def reject_candidate(
    candidate_id: str,
    staging_path: Path | str = DEFAULT_STAGING_PATH,
) -> bool:
    """Đánh dấu loại bỏ 1 mục trong hàng đợi Staging."""
    queue = load_staging_queue(staging_path)
    for c in queue.candidates:
        if c.candidate_id == candidate_id:
            c.status = "REJECTED"
            save_staging_queue(queue, staging_path=staging_path)
            return True
    return False


def clear_processed_candidates(staging_path: Path | str = DEFAULT_STAGING_PATH) -> int:
    """Dọn các mục đã APPROVED / REJECTED khỏi danh sách hiển thị."""
    queue = load_staging_queue(staging_path)
    before = len(queue.candidates)
    queue.candidates = [c for c in queue.candidates if c.status == "PENDING"]
    save_staging_queue(queue, staging_path=staging_path)
    return before - len(queue.candidates)
