"""
NHÁNH NGUYỄN HỮU HẢI LONG — Module Query Rewriting & Domain Routing:
- Chuyển đổi câu hỏi đời thường của người dân thành truy vấn pháp lý chuẩn (kèm mã thủ tục/điều luật) cho 40 thủ tục.
- Tự động phân định câu hỏi thuộc nhóm 'Thủ tục hành chính' hay 'Tố tụng hình sự'.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


COLLOQUIAL_TO_LEGAL_MAP: List[Tuple[List[str], str, Optional[str]]] = [
    (
        ["đặt tiền", "đặt để bảo đảm", "nộp tiền bảo lãnh", "tiền bảo đảm", "ký quỹ tại ngoại", "điều 122", "ttlt 06/2018", "kho bạc nhà nước"],
        "Thủ tục đặt tiền để bảo đảm thay thế biện pháp tạm giam Điều 122 BLTTHS TTHS-DT-122 Kho bạc Nhà nước",
        "Tố tụng hình sự",
    ),
    (
        ["bảo lãnh", "bảo lĩnh", "xin tại ngoại", "nhận bảo lĩnh", "don-bl-01", "điều 121"],
        "Thủ tục nhận bảo lĩnh cho bị can bị cáo tại ngoại Điều 121 BLTTHS TTHS-BL-121 ít nhất 02 người thân thích",
        "Tố tụng hình sự",
    ),
    (
        ["lừa đảo", "trình báo", "tố giác", "tin báo về tội phạm", "báo công an", "mất trộm", "đột nhập", "cướp giật", "chiếm đoạt", "điều 145", "điều 146", "điều 147"],
        "Thủ tục trình báo tố giác tin báo về tội phạm Điều 145 Điều 147 BLTTHS TTHS-TG-145 miễn phí",
        "Tố tụng hình sự",
    ),
    (
        ["giám định thương tích", "trưng cầu giám định", "giám định pháp y", "tỷ lệ thương tật", "tổn thương cơ thể", "định giá tài sản", "hội đồng định giá", "bị đánh", "điều 205", "điều 207", "điều 208", "điều 215"],
        "Thủ tục yêu cầu trưng cầu giám định tỷ lệ tổn thương cơ thể định giá tài sản Điều 205 BLTTHS TTHS-GD-205",
        "Tố tụng hình sự",
    ),
    (
        ["thăm gặp", "thăm nuôi", "gặp thân nhân", "gặp người bị tạm giam", "gặp người đang bị tạm giữ", "trại tạm giam", "nhà tạm giữ", "cơ sở giam giữ"],
        "Thủ tục xin thăm gặp người đang bị tạm giữ tạm giam Điều 22 Luật Thi hành tạm giữ tạm giam TTHS-TG-022",
        "Tố tụng hình sự",
    ),
    (
        ["kháng cáo", "chống án", "đơn kháng cáo", "quá hạn kháng cáo", "điều 331", "điều 332", "điều 333"],
        "Thủ tục nộp đơn kháng cáo bản án sơ thẩm hình sự Điều 331 Điều 333 BLTTHS TTHS-KC-331 thời hạn 15 ngày",
        "Tố tụng hình sự",
    ),
    (
        [
            "xóa án tích theo quyết định",
            "tòa án xóa án tích",
            "tòa án quyết định xóa án tích",
            "đơn xin xóa án tích",
            "bác đơn xin xóa án tích",
            "quyết định xóa án tích",
            "an ninh quốc gia",
            "xin tòa xóa án tích",
            "phá hoại hòa bình",
            "chương xiii",
            "chương xxvi",
            "điều 71",
            "khoản 2 điều 369",
            "khoản 3 điều 369",
        ],
        "Thủ tục nộp hồ sơ xin Tòa án xóa án tích theo quyết định Điều 71 BLHS TTHS-XAT-071 TTHS-XAT-071",
        "Tố tụng hình sự",
    ),
    (
        ["đương nhiên xóa án tích", "đương nhiên được xóa án tích", "chứng nhận xóa án tích", "điều 70 blhs", "điều 70 bộ luật hình sự"],
        "Thủ tục xác nhận đương nhiên được xóa án tích Điều 70 BLHS Điều 369 BLTTHS TTHS-XAT-070 Sở Tư pháp",
        "Tố tụng hình sự",
    ),
    (
        ["lý lịch tư pháp", "phiếu số 1", "phiếu số 2", "lltp", "thông tư 244/2016"],
        "Thủ tục yêu cầu cấp Phiếu lý lịch tư pháp số 1 và số 2 Luật Lý lịch tư pháp TTHS-LLTP-045 Sở Tư pháp",
        "Tố tụng hình sự",
    ),
    (
        ["hoãn thi hành án", "hoãn chấp hành án", "hoãn chấp hành hình phạt tù", "hoãn đi tù", "nuôi con dưới 36 tháng", "lao động duy nhất", "điều 67 blhs", "điều 67 bộ luật hình sự", "điều 24 luật thi hành án"],
        "Thủ tục xin hoãn chấp hành án phạt tù Điều 67 BLHS Điều 24 Luật Thi hành án hình sự TTHS-THA-067",
        "Tố tụng hình sự",
    ),
    (
        ["khiếu nại tố tụng", "khiếu nại quyết định", "khiếu nại hành vi tố tụng", "khiếu nại lệnh bắt", "khiếu nại điều tra viên", "khiếu nại thẩm phán", "thời hiệu khiếu nại", "không khởi tố", "điều 471", "điều 475", "điều 476", "điều 477"],
        "Thủ tục khiếu nại quyết định hành vi tố tụng trong tố tụng hình sự Điều 475 BLTTHS TTHS-KN-475",
        "Tố tụng hình sự",
    ),
    (
        ["trợ giúp pháp lý", "trợ giúp viên pháp lý", "luật sư miễn phí", "bào chữa miễn phí", "cử luật sư", "cử người bào chữa", "hộ nghèo bị buộc tội", "luật trợ giúp pháp lý", "luật tgpl"],
        "Thủ tục yêu cầu Trợ giúp pháp lý miễn phí cử Luật sư bào chữa Điều 7 Luật Trợ giúp pháp lý TTHS-TGPL-007",
        "Tố tụng hình sự",
    ),
    (
        ["liên thông khai sinh", "liên thông điện tử", "khai sinh liên thông", "3 trong 1", "khai sinh thường trú bhyt", "nghị định 63/2024"],
        "Thủ tục liên thông điện tử đăng ký khai sinh thường trú cấp thẻ BHYT cho trẻ dưới 6 tuổi Nghị định 63/2024/NĐ-CP TTHC-HT-06 TTHC-HT-06",
        "Thủ tục hành chính",
    ),
    (
        ["khai sinh", "đăng ký khai sinh", "giấy chứng sinh", "mới sinh con"],
        "Thủ tục đăng ký khai sinh thông thường Điều 15 Điều 16 Luật Hộ tịch TTHC-HT-01 UBND cấp xã",
        "Thủ tục hành chính",
    ),
    (
        [
            "kết hôn với người nước ngoài",
            "kết hôn có yếu tố nước ngoài",
            "yếu tố nước ngoài",
            "chồng nước ngoài",
            "vợ nước ngoài",
            "quốc tịch hàn",
            "quốc tịch nhật",
            "quốc tịch pháp",
            "người đức",
            "người nước ngoài",
            "hợp pháp hóa lãnh sự",
            "điều 37 luật hộ tịch",
        ],
        "Thủ tục đăng ký kết hôn có yếu tố nước ngoài Điều 37 Luật Hộ tịch TTHC-HT-05 TTHC-HT-05 UBND cấp huyện",
        "Thủ tục hành chính",
    ),
    (
        ["tình trạng hôn nhân", "xác nhận độc thân", "giấy độc thân", "nghị định 123/2015", "tk-tthn-01"],
        "Thủ tục cấp Giấy xác nhận tình trạng hôn nhân Nghị định 123/2015/NĐ-CP TTHC-HT-03 TTHC-HT-03 UBND cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["kết hôn trong nước", "đăng ký kết hôn"],
        "Thủ tục đăng ký kết hôn trong nước Điều 17 Điều 18 Luật Hộ tịch TTHC-HT-02 UBND cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["khai tử", "giấy báo tử", "trích lục khai tử", "người thân mất", "điều 32 luật hộ tịch", "điều 34 luật hộ tịch"],
        "Thủ tục đăng ký khai tử Điều 32 Điều 34 Luật Hộ tịch TTHC-HT-04 UBND cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["sao y", "chứng thực bản sao", "công chứng bản sao", "sao y bản chính", "bản sao điện tử", "nghị định 23/2015"],
        "Thủ tục chứng thực bản sao từ bản chính Nghị định 23/2015/NĐ-CP TTHC-HT-07 UBND cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["thường trú", "nhập khẩu", "sổ hộ khẩu", "hk07", "ct01"],
        "Thủ tục đăng ký thường trú Luật Cư trú 2020 Điều 20 TTHC-CT-01 Công an cấp xã bãi bỏ Sổ hộ khẩu giấy",
        "Thủ tục hành chính",
    ),
    (
        ["tạm trú", "gia hạn tạm trú", "sổ tạm trú", "ở trọ", "điều 27 luật cư trú"],
        "Thủ tục đăng ký tạm trú Luật Cư trú 2020 Điều 27 TTHC-CT-02 Công an cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["định danh điện tử", "vneid mức 2", "vneid mức độ 2", "tài khoản định danh", "kích hoạt vneid", "nghị định 69/2024"],
        "Thủ tục cấp tài khoản định danh điện tử mức độ 2 VNeID Nghị định 69/2024/NĐ-CP TTHC-CT-03 TTHC-CT-03",
        "Thủ tục hành chính",
    ),
    (
        ["tạm vắng", "khai báo tạm vắng", "đi khỏi nơi cư trú", "mẫu ct03", "điều 31 luật cư trú"],
        "Thủ tục khai báo tạm vắng Luật Cư trú 2020 Điều 31 TTHC-CT-04 Công an cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["căn cước", "cccd", "cmnd 9 số", "cmnd 12 số", "chứng minh nhân dân", "đổi thẻ căn cước", "luật căn cước 2023"],
        "Thủ tục cấp đổi cấp lại thẻ Căn cước Luật Căn cước 2023 Điều 24 TTHC-CC-01 CMND hết hiệu lực 01/01/2025",
        "Thủ tục hành chính",
    ),
    (
        ["trợ cấp thất nghiệp", "bảo hiểm thất nghiệp", "thất nghiệp"],
        "Thủ tục nộp hồ sơ đăng ký và hưởng trợ cấp thất nghiệp Luật Việc làm Điều 46 TTHC-LD-01 Trung tâm dịch vụ việc làm",
        "Thủ tục hành chính",
    ),
    (
        ["giấy phép lao động", "lao động nước ngoài", "chuyên gia nước ngoài"],
        "Thủ tục cấp Giấy phép lao động cho người nước ngoài Nghị định 152/2020/NĐ-CP Nghị định 70/2023/NĐ-CP TTHC-LD-02",
        "Thủ tục hành chính",
    ),
    (
        ["bhxh một lần", "bảo hiểm xã hội 1 lần", "rút bhxh", "rút bảo hiểm xã hội"],
        "Thủ tục giải quyết hưởng chế độ Bảo hiểm xã hội một lần Điều 60 Luật BHXH TTHC-BH-01",
        "Thủ tục hành chính",
    ),
    (
        ["thẻ bhyt", "bảo hiểm y tế", "đổi thẻ bhyt", "cấp lại thẻ bhyt", "bhyt hộ gia đình"],
        "Thủ tục cấp lại đổi thẻ Bảo hiểm y tế BHYT Luật Bảo hiểm y tế TTHC-BH-02",
        "Thủ tục hành chính",
    ),
    (
        ["thai sản", "chế độ thai sản", "sinh con hưởng bhxh", "tiền thai sản"],
        "Thủ tục giải quyết hưởng chế độ thai sản Điều 31 Luật BHXH TTHC-BH-03",
        "Thủ tục hành chính",
    ),
    (
        ["giấy phép lái xe", "bằng lái xe", "gplx", "đổi bằng lái", "mất bằng lái", "lái xe"],
        "Thủ tục cấp đổi cấp lại Giấy phép lái xe Thông tư 12/2017/TT-BGTVT TTHC-GT-01 TTHC-GT-01 Sở Giao thông vận tải",
        "Thủ tục hành chính",
    ),
    (
        ["đăng ký xe", "biển số xe", "biển số định danh", "sang tên xe", "cà vẹt xe"],
        "Thủ tục đăng ký cấp biển số xe mô tô ô tô biển số định danh Thông tư 24/2023/TT-BCA TTHC-GT-02",
        "Thủ tục hành chính",
    ),
    (
        ["hộ chiếu", "passport", "hộ chiếu gắn chíp", "làm hộ chiếu", "cấp hộ chiếu"],
        "Thủ tục cấp hộ chiếu phổ thông ở trong nước Luật Xuất cảnh nhập cảnh TTHC-HC-01 TTHC-HC-01",
        "Thủ tục hành chính",
    ),
    (
        ["sang tên sổ đỏ", "biến động đất đai", "chuyển nhượng đất", "tặng cho đất", "sang tên nhà đất"],
        "Thủ tục đăng ký biến động đất đai sang tên Sổ đỏ Luật Đất đai 2024 Nghị định 101/2024/NĐ-CP TTHC-DD-01",
        "Thủ tục hành chính",
    ),
    (
        ["cấp sổ đỏ lần đầu", "làm sổ đỏ lần đầu", "cấp giấy chứng nhận quyền sử dụng đất lần đầu", "làm sổ đỏ"],
        "Thủ tục đăng ký đất đai cấp Giấy chứng nhận quyền sử dụng đất Sổ đỏ lần đầu Luật Đất đai 2024 TTHC-DD-02",
        "Thủ tục hành chính",
    ),
    (
        ["giấy phép xây dựng", "xin phép xây nhà", "xây dựng nhà ở"],
        "Thủ tục cấp Giấy phép xây dựng mới đối với nhà ở riêng lẻ Luật Xây dựng TTHC-DD-03 UBND cấp huyện",
        "Thủ tục hành chính",
    ),
    (
        ["hộ kinh doanh", "đăng ký hộ kinh doanh", "kinh doanh cá thể", "mở cửa hàng"],
        "Thủ tục đăng ký thành lập Hộ kinh doanh cá thể Nghị định 01/2021/NĐ-CP TTHC-KD-01",
        "Thủ tục hành chính",
    ),
    (
        ["thành lập doanh nghiệp", "thành lập công ty", "mở công ty", "đăng ký doanh nghiệp"],
        "Thủ tục đăng ký thành lập mới Doanh nghiệp Công ty TNHH Cổ phần Luật Doanh nghiệp 2020 TTHC-KD-02",
        "Thủ tục hành chính",
    ),
    (
        ["quyết toán thuế", "hoàn thuế", "thuế thu nhập cá nhân", "thuế tncn", "etax mobile"],
        "Thủ tục quyết toán và hoàn thuế thu nhập cá nhân TNCN Luật Quản lý thuế TTHC-KD-03",
        "Thủ tục hành chính",
    ),
]


def rewrite_query_and_route(
    query: str,
    preferred_domain: Optional[str] = None,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Chuyển đổi câu hỏi đời thường thành truy vấn pháp lý chuẩn (Query Rewriting)
    và phân loại miền dữ liệu (Domain Routing: Thủ tục hành chính vs Tố tụng hình sự).

    Hai tầng xử lý:
    1. COLLOQUIAL_TO_LEGAL_MAP (tốc độ cao, zero-token-cost, không cần LLM)
    2. LLM Query Decomposition (tùy chọn, dùng khi câu hỏi đa ý / liên thông)
    """
    q_lower = query.lower()
    expansions: List[str] = []
    detected_domain: Optional[str] = preferred_domain

    # Tầng 1: Khớp COLLOQUIAL_TO_LEGAL_MAP (có khử nhiễu khi thủ tục con đặc thù đã khớp)
    matched_domains: List[str] = []
    has_interconnected_birth = (
        ("liên thông" in q_lower and ("khai sinh" in q_lower or "bhyt" in q_lower))
        or "3 trong 1" in q_lower
        or "nghị định 63/2024" in q_lower
    )
    has_foreign_marriage = "kết hôn" in q_lower and any(
        m in q_lower
        for m in (
            "nước ngoài", "quốc tịch", "hợp pháp hóa lãnh sự", "nhật bản", "hàn quốc",
            "người đức", "pháp", "mỹ", "anh", "úc", "đài loan", "trung quốc", "điều 37",
        )
    )
    has_marital_status_cert = any(
        m in q_lower for m in ("tình trạng hôn nhân", "xác nhận độc thân", "tk-tthn-01", "nghị định 123/2015")
    )
    has_vneid_proc = any(
        m in q_lower for m in ("vneid mức 2", "vneid mức độ 2", "tài khoản định danh", "định danh điện tử", "nghị định 69/2024")
    )

    for keywords, legal_expansion, domain in COLLOQUIAL_TO_LEGAL_MAP:
        if any(kw in q_lower for kw in keywords):
            # Tránh để thủ tục cha chung lấn át thủ tục con đặc thù đã xác định rõ
            if has_interconnected_birth and "TTHC-HT-01" in legal_expansion:
                continue
            if has_foreign_marriage and ("TTHC-HT-02" in legal_expansion or "TTHC-HT-03" in legal_expansion):
                continue
            if has_marital_status_cert and not has_foreign_marriage and "TTHC-HT-02" in legal_expansion:
                continue
            if "khai sinh" in q_lower and "TTHC-HT-02" in legal_expansion:
                continue
            if has_vneid_proc and "kích hoạt" in q_lower and "TTHC-CC-01" in legal_expansion:
                continue
            if ("giấy phép lao động" in q_lower or "11/pli" in q_lower) and "TTHS-LLTP-045" in legal_expansion:
                continue
            if ("tạm vắng" in q_lower or "ct03" in q_lower) and "TTHS-BL-121" in legal_expansion:
                continue

            expansions.append(legal_expansion)
            if domain:
                matched_domains.append(domain)

    if not detected_domain and matched_domains:
        # Ưu tiên domain của thủ tục hành chính chuyên ngành nếu câu hỏi nêu đích danh tên TTHC
        admin_override_phrases = (
            "khai báo tạm vắng",
            "phiếu khai báo tạm vắng",
            "giấy phép lao động",
            "11/pli",
            "đăng ký thường trú",
            "đăng ký tạm trú",
        )
        if any(p in q_lower for p in admin_override_phrases):
            detected_domain = "Thủ tục hành chính"
        else:
            hc_votes = matched_domains.count("Thủ tục hành chính")
            hs_votes = matched_domains.count("Tố tụng hình sự")
            detected_domain = "Tố tụng hình sự" if hs_votes >= hc_votes and hs_votes > 0 else matched_domains[0]

    # Phát hiện domain qua marker từ khóa hình sự khi chưa khớp bảng ánh xạ
    if not detected_domain:
        criminal_markers = [
            "bị can", "bị cáo", "tạm giam", "tạm giữ", "tội phạm", "khởi tố",
            "điều tra", "viện kiểm sát", "xét xử", "án tích", "kháng cáo", "án phạt tù",
            "hình phạt tù", "chấp hành án", "bảo lĩnh", "tố giác", "giám định", "định giá tài sản",
            "lý lịch tư pháp", "bào chữa", "trợ giúp pháp lý", "trợ giúp viên", "bị hại",
            "quyết định tố tụng", "hành vi tố tụng", "bltths", "tố tụng hình sự", "bộ luật hình sự",
            "blhs", "csđt", "vụ án",
        ]
        if any(m in q_lower for m in criminal_markers):
            detected_domain = "Tố tụng hình sự"
        else:
            detected_domain = "Thủ tục hành chính"

    rewritten_query = query
    if expansions:
        rewritten_query = f"{query} | Truy vấn pháp lý mở rộng: {' ; '.join(expansions)}"

    # Tầng 2: LLM Query Decomposition (tùy chọn — chỉ chạy khi có LLM và câu hỏi dài/phức tạp)
    llm_decomposed: List[str] = []
    if llm is not None and len(query.split()) >= 8 and len(expansions) == 0:
        try:
            decompose_prompt = (
                "Bạn là chuyên gia pháp lý Việt Nam. Hãy phân tách câu hỏi sau thành "
                "tối đa 3 truy vấn pháp lý con (mỗi dòng một truy vấn), chỉ liệt kê "
                "thuật ngữ pháp lý, không giải thích:\n"
                f"Câu hỏi: {query}\nTruy vấn con:"
            )
            response = llm.invoke(decompose_prompt)
            raw_text = getattr(response, "content", str(response))
            llm_decomposed = [
                line.strip().lstrip("-•●123456789. ")
                for line in raw_text.strip().split("\n")
                if len(line.strip()) >= 5
            ][:3]
            if llm_decomposed:
                rewritten_query = (
                    f"{rewritten_query} | LLM phân tách: {' ; '.join(llm_decomposed)}"
                )
        except Exception:
            pass  # Fallback im lặng về Tầng 1

    return {
        "original_query": query,
        "rewritten_query": rewritten_query,
        "detected_domain": detected_domain,
        "matched_expansions": expansions,
        "llm_decomposed": llm_decomposed,
    }
