"""
Package `src.ingestion` — Tự động hóa thu thập, bóc tách 6 trường chuẩn hóa
và kiểm duyệt (Staging) dữ liệu pháp luật chính thống cho APCR-RAG.
"""

from src.ingestion.chinhphu_client import (
    fetch_chinhphu_detail,
    fetch_congbao_latest,
    get_official_sources_status,
    parse_congbao_html,
    parse_detail_page_html,
    parse_search_results_html,
    parse_vn_date,
    search_chinhphu_documents,
)
from src.ingestion.models import (
    ExtractedProcedureCandidate,
    OfficialSourceDocument,
    ProcedureExtractionBatch,
    StagingQueue,
)
from src.ingestion.pdf_extractor import download_and_extract_pdf, extract_text_from_pdf_bytes
from src.ingestion.procedure_extractor import (
    detect_expiry_impacts,
    extract_procedures_deterministic,
    extract_procedures_from_document,
)
from src.ingestion.staging_store import (
    DEFAULT_BACKUPS_DIR,
    DEFAULT_STAGING_PATH,
    add_candidates_to_staging,
    approve_and_merge_candidate,
    clear_processed_candidates,
    load_staging_queue,
    reject_candidate,
    save_staging_queue,
)

__all__ = [
    "OfficialSourceDocument",
    "ProcedureExtractionBatch",
    "ExtractedProcedureCandidate",
    "StagingQueue",
    "get_official_sources_status",
    "parse_vn_date",
    "parse_search_results_html",
    "parse_detail_page_html",
    "parse_congbao_html",
    "search_chinhphu_documents",
    "fetch_chinhphu_detail",
    "fetch_congbao_latest",
    "extract_text_from_pdf_bytes",
    "download_and_extract_pdf",
    "detect_expiry_impacts",
    "extract_procedures_deterministic",
    "extract_procedures_from_document",
    "DEFAULT_STAGING_PATH",
    "DEFAULT_BACKUPS_DIR",
    "load_staging_queue",
    "save_staging_queue",
    "add_candidates_to_staging",
    "approve_and_merge_candidate",
    "reject_candidate",
    "clear_processed_candidates",
]
