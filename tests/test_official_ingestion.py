"""
Bộ kiểm thử cho hệ thống tự động hóa nguồn dữ liệu chính thống (`src/ingestion/`):
- Phân tích HTML tìm kiếm, trang chi tiết `vanban.chinhphu.vn` và `congbao.chinhphu.vn`.
- Trích xuất văn bản PDF (`extract_text_from_pdf_bytes`).
- Chuẩn hóa sang lược đồ 6 trường `ProcedureRecord` và phát hiện văn bản hết hiệu lực.
- Hàng đợi Staging, sao lưu tự động, gộp vào JSON và hot-reload `RetrieverAgent`.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ingestion import (
    OfficialSourceDocument,
    add_candidates_to_staging,
    approve_and_merge_candidate,
    detect_expiry_impacts,
    extract_procedures_from_document,
    extract_text_from_pdf_bytes,
    get_official_sources_status,
    load_staging_queue,
    parse_congbao_html,
    parse_detail_page_html,
    parse_search_results_html,
    parse_vn_date,
    reject_candidate,
)
from src.retriever import RetrieverAgent
from src.schemas import ProcedureRecord

_SAMPLE_SEARCH_HTML = """
<html><body><table>
<tr>
  <td><a href="/?pageid=27160&amp;docid=219796"><span class="code">14/2026/TT-BTP</span><span class="issue-v2">30/09/2026</span></a></td>
  <td><span class="issued-date">30/09/2026</span></td>
  <td>
    <a href="/?pageid=27160&amp;docid=219796"><span class="substract">Hướng dẫn thực hiện thủ tục hành chính trên môi trường điện tử</span></a>
    <div class="bl-doc-files"><div class="bl-doc-file"><a href="https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/10/14-btp.signed.pdf">Tải</a></div></div>
  </td>
</tr>
</table></body></html>
"""

_SAMPLE_DETAIL_HTML = """
<html><head><title>Thông tư số 14/2026/TT-BTP</title></head><body>
<table>
  <tr><td>Số ký hiệu</td><td>14/2026/TT-BTP</td></tr>
  <tr><td>Ngày ban hành</td><td>30-09-2026</td></tr>
  <tr><td>Ngày có hiệu lực</td><td>10-10-2026</td></tr>
  <tr><td>Loại văn bản</td><td>Thông tư</td></tr>
  <tr><td>Cơ quan ban hành</td><td>Bộ Tư pháp</td></tr>
  <tr><td>Trích yếu</td><td>Hướng dẫn về thực hiện thủ tục hành chính trên môi trường điện tử</td></tr>
</table>
<a href="https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/10/14-btp.signed.pdf">14-btp.signed.pdf</a>
</body></html>
"""

_SAMPLE_CONGBAO_HTML = """
<html><body>
  <a href="/van-ban/nghi-dinh-so-404-2026-nd-cp-470959.htm">Nghị định số 404/2026/NĐ-CP quy định thủ tục cấp phép</a>
</body></html>
"""


class TestOfficialIngestion(unittest.TestCase):
    def test_parse_vn_date_and_portal_html(self) -> None:
        self.assertEqual(parse_vn_date("30-09-2026"), "2026-09-30")
        self.assertEqual(parse_vn_date("5/1/2025"), "2025-01-05")
        self.assertEqual(parse_vn_date("2026-10-10"), "2026-10-10")

        hits = parse_search_results_html(_SAMPLE_SEARCH_HTML)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["doc_id"], "219796")
        self.assertEqual(hits[0]["so_hieu"], "14/2026/TT-BTP")
        self.assertEqual(hits[0]["ngay_ban_hanh"], "2026-09-30")
        self.assertIn("14-btp.signed.pdf", hits[0]["pdf_urls"][0])

        detail = parse_detail_page_html(_SAMPLE_DETAIL_HTML, source_url="https://vanban.chinhphu.vn/?pageid=27160&docid=219796")
        self.assertEqual(detail.so_hieu, "14/2026/TT-BTP")
        self.assertEqual(detail.ngay_hieu_luc, "2026-10-10")
        self.assertEqual(detail.co_quan_ban_hanh, "Bộ Tư pháp")
        self.assertEqual(detail.trang_thai_hieu_luc, "ĐANG HIỆU LỰC")

        cb = parse_congbao_html(_SAMPLE_CONGBAO_HTML)
        self.assertEqual(len(cb), 1)
        self.assertEqual(cb[0]["so_hieu"], "404/2026/NĐ-CP")

    def test_pdf_extractor_handles_empty_invalid_and_mocked_reader(self) -> None:
        self.assertEqual(extract_text_from_pdf_bytes(b""), "")
        self.assertEqual(extract_text_from_pdf_bytes(b"not a pdf"), "")

        fake_page = MagicMock()
        fake_page.extract_text.return_value = "Điều 1. Thủ tục cấp bản sao hộ tịch\nLệ phí 50.000 đồng."
        fake_reader = MagicMock()
        fake_reader.is_encrypted = False
        fake_reader.pages = [fake_page]

        with patch("pypdf.PdfReader", return_value=fake_reader):
            txt = extract_text_from_pdf_bytes(b"%PDF-1.4 dummy binary bytes")
        self.assertIn("Điều 1", txt)
        self.assertIn("50.000 đồng", txt)

    def test_deterministic_extraction_and_expiry_impact_detection(self) -> None:
        doc = OfficialSourceDocument(
            source_name="vanban.chinhphu.vn",
            doc_id="999001",
            source_url="https://vanban.chinhphu.vn/?pageid=27160&docid=999001",
            so_hieu="99/2026/NĐ-CP",
            trich_yeu="Quy định thủ tục đăng ký giấy xác nhận tình trạng hôn nhân điện tử",
            loai_van_ban="Nghị định",
            co_quan_ban_hanh="Chính phủ",
            ngay_ban_hanh="2026-11-01",
            ngay_hieu_luc="2026-11-15",
            raw_text=(
                "Điều 5. Trình tự thực hiện thủ tục hộ tịch điện tử:\n"
                "Bước 1: Nộp Tờ khai cấp giấy xác nhận trực tuyến trên Cổng dịch vụ công.\n"
                "Bước 2: UBND cấp xã kiểm tra dữ liệu hộ tịch điện tử.\n"
                "Thời hạn giải quyết: tối đa 03 ngày làm việc kể từ ngày nhận đủ hồ sơ.\n"
                "Lệ phí: mức thu 15.000 đồng cho mỗi bản cấp.\n"
                "Điều 20. Hiệu lực thi hành: Nghị định này thay thế quy định cũ tại 101/2015/QH13."
            ),
        )
        fake_existing = [
            {
                "ma_va_ten_thu_tuc": {
                    "ma_thu_tuc": "TTHS-BL-121",
                    "ten_thu_tuc": "Thủ tục nhận bảo lĩnh",
                    "phan_nhom": "Tố tụng hình sự",
                    "linh_vuc": "Biện pháp ngăn chặn",
                    "cap_tham_quyen": ["Cơ quan CSĐT"],
                    "giai_doan_to_tung": ["Điều tra"],
                },
                "trinh_tu_thuc_hien": ["Bước 1: Nộp đơn"],
                "thanh_phan_ho_so": {
                    "giay_to_bat_buoc": [{"form_id": "DON-01", "ten_giay_to": "Đơn xin bảo lĩnh"}],
                    "dieu_kien_bat_buoc": "Có ít nhất 02 người thân thích bảo lĩnh.",
                },
                "thoi_han_giai_quyet": {"so_ngay_toi_da": 3, "don_vi_tinh": "ngày làm việc", "chi_tiet": "3 ngày"},
                "le_phi": {"muc_phi_vnd": 0, "mo_ta": "Miễn phí"},
                "can_cu_phap_ly": [
                    {
                        "law_id": "bltths2015-d121",
                        "so_hieu_van_ban": "Bộ luật Tố tụng Hình sự số 101/2015/QH13",
                        "dieu_khoan_diem": "Điều 121",
                        "ngay_hieu_luc": "2018-01-01",
                        "trang_thai_hieu_luc": "ĐANG HIỆU LỰC",
                        "van_ban_thay_the": None,
                    }
                ],
            }
        ]

        cands = extract_procedures_from_document(doc, existing_procedures=fake_existing, use_llm=False)
        actions = [c.action for c in cands]
        self.assertIn("ADD", actions)
        self.assertIn("UPDATE_EXPIRY", actions)

        add_cand = next(c for c in cands if c.action == "ADD")
        self.assertEqual(add_cand.record.le_phi.muc_phi_vnd, 15000)
        self.assertEqual(add_cand.record.thoi_han_giai_quyet.so_ngay_toi_da, 3)
        self.assertEqual(add_cand.record.can_cu_phap_ly[0].ngay_hieu_luc, "2026-11-15")

        exp_cand = next(c for c in cands if c.action == "UPDATE_EXPIRY")
        self.assertEqual(exp_cand.target_existing_code, "TTHS-BL-121")
        self.assertEqual(exp_cand.record.can_cu_phap_ly[0].trang_thai_hieu_luc, "HẾT HIỆU LỰC")
        self.assertEqual(exp_cand.record.can_cu_phap_ly[0].van_ban_thay_the, "99/2026/NĐ-CP")

        # Kiểm chứng detect_expiry_impacts không tự gắn chính văn bản mới là văn bản cũ
        self.assertEqual(detect_expiry_impacts(doc, [], llm_replaced_codes=["99/2026/NĐ-CP"]), [])

    def test_staging_approve_creates_backup_and_updates_retriever(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_path = tmp / "data_procedures.json"
            staging_path = tmp / "staging.json"
            backups_dir = tmp / "backups"

            shutil.copy2(Path("data/data_procedures.json"), data_path)
            initial_count = len(json.loads(data_path.read_text(encoding="utf-8")))

            doc = OfficialSourceDocument(
                source_name="vanban.chinhphu.vn",
                doc_id="888888",
                source_url="https://vanban.chinhphu.vn/?pageid=27160&docid=888888",
                so_hieu="77/2026/NĐ-CP",
                trich_yeu="Thủ tục cấp phép bay thiết bị không người lái thử nghiệm",
                loai_van_ban="Nghị định",
                co_quan_ban_hanh="Bộ Quốc phòng",
                ngay_ban_hanh="2026-11-10",
                ngay_hieu_luc="2026-11-20",
                raw_text="Bước 1: Nộp Đơn đề nghị cấp phép bay. Thời hạn giải quyết 07 ngày làm việc. Miễn lệ phí.",
            )
            cands = extract_procedures_from_document(doc, use_llm=False)
            stats = add_candidates_to_staging(cands, source_summary="unit-test", staging_path=staging_path)
            self.assertEqual(stats["added"], 1)

            # Kiểm tra chống thêm trùng
            dup_stats = add_candidates_to_staging(cands, source_summary="unit-test-dup", staging_path=staging_path)
            self.assertEqual(dup_stats["skipped"], 1)

            queue = load_staging_queue(staging_path)
            cand = queue.candidates[0]
            edited = cand.record.model_dump()
            edited["ma_va_ten_thu_tuc"]["ma_thu_tuc"] = "TTHC-TEST-BAY-77"
            ProcedureRecord.model_validate(edited)

            merge_res = approve_and_merge_candidate(
                cand.candidate_id,
                edited_record_dict=edited,
                data_path=data_path,
                staging_path=staging_path,
                backups_dir=backups_dir,
                reload_retriever=False,
            )
            self.assertEqual(merge_res["merged_code"], "TTHC-TEST-BAY-77")
            self.assertEqual(merge_res["total_procedures"], initial_count + 1)
            self.assertTrue(Path(merge_res["backup_path"]).exists())

            # RetrieverAgent nạp từ file đã cập nhật phải tìm thấy ngay thủ tục mới
            retriever = RetrieverAgent(data_path=data_path)
            search_res = retriever.retrieve_hybrid("cấp phép bay thiết bị không người lái", top_k=3)
            found_codes = [item["ma_thu_tuc"] for item in search_res["results"]]
            self.assertIn("TTHC-TEST-BAY-77", found_codes)

            # Kiểm tra reject_candidate
            cands2 = extract_procedures_from_document(doc, use_llm=False)
            cands2[0].record.ma_va_ten_thu_tuc.ma_thu_tuc = "TTHC-REJECT-ME"
            add_candidates_to_staging([cands2[0]], staging_path=staging_path)
            self.assertTrue(reject_candidate(cands2[0].candidate_id, staging_path=staging_path))

    def test_official_sources_status_reports_transparently(self) -> None:
        statuses = get_official_sources_status()
        ids = {s["id"] for s in statuses}
        self.assertEqual(ids, {"vanban_chinhphu", "congbao_chinhphu", "dichvucong_ndxp"})


if __name__ == "__main__":
    unittest.main()
