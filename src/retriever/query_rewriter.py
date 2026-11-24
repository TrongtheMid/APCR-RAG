"""
NHÁNH NGUYỄN HỮU HẢI LONG — Module Query Rewriting & Domain Routing:
- Chuyển đổi câu hỏi đời thường của người dân thành truy vấn pháp lý chuẩn (kèm mã thủ tục/điều luật) cho 40 thủ tục.
- Tự động phân định câu hỏi thuộc nhóm 'Thủ tục hành chính' hay 'Tố tụng hình sự'.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


COLLOQUIAL_TO_LEGAL_MAP: List[Tuple[List[str], str, Optional[str]]] = [
    (
        ["bảo lãnh", "bảo lĩnh", "tại ngoại", "bắt tạm giam", "xin về"],
        "Thủ tục nhận bảo lĩnh cho bị can bị cáo tại ngoại Điều 121 BLTTHS TTHS-BL-121 ít nhất 02 người thân thích",
        "Tố tụng hình sự",
    ),
    (
        ["đặt tiền", "nộp tiền bảo lãnh", "tiền bảo đảm", "ký quỹ tại ngoại"],
        "Thủ tục đặt tiền để bảo đảm thay thế biện pháp tạm giam Điều 122 BLTTHS TTHS-DT-122 Kho bạc Nhà nước",
        "Tố tụng hình sự",
    ),
    (
        ["lừa đảo", "trình báo", "tố giác", "báo công an", "mất trộm", "trộm cắp", "chiếm đoạt"],
        "Thủ tục trình báo tố giác tin báo về tội phạm Điều 145 Điều 147 BLTTHS TTHS-TG-145 miễn phí",
        "Tố tụng hình sự",
    ),
    (
        ["giám định thương tích", "tỷ lệ thương tật", "tổn thương cơ thể", "định giá tài sản", "bị đánh"],
        "Thủ tục yêu cầu trưng cầu giám định tỷ lệ tổn thương cơ thể định giá tài sản Điều 205 BLTTHS TTHS-GD-205",
        "Tố tụng hình sự",
    ),
    (
        ["thăm gặp", "thăm nuôi", "gặp người bị tạm giam", "trại tạm giam", "nhà tạm giữ"],
        "Thủ tục xin thăm gặp người đang bị tạm giữ tạm giam Điều 22 Luật Thi hành tạm giữ tạm giam TTHS-TG-022",
        "Tố tụng hình sự",
    ),
    (
        ["kháng cáo", "chống án", "phúc thẩm", "quá hạn kháng cáo"],
        "Thủ tục nộp đơn kháng cáo bản án sơ thẩm hình sự Điều 331 Điều 333 BLTTHS TTHS-KC-331 thời hạn 15 ngày",
        "Tố tụng hình sự",
    ),
    (
        ["xóa án tích theo quyết định", "an ninh quốc gia", "xin tòa xóa án tích", "phá hoại hòa bình"],
        "Thủ tục nộp hồ sơ xin Tòa án xóa án tích theo quyết định Điều 71 BLHS TTHS-XAT-071 TTHS-XAT-071",
        "Tố tụng hình sự",
    ),
    (
        ["đương nhiên xóa án tích", "chứng nhận xóa án tích", "tội trộm cắp", "tội đánh bạc"],
        "Thủ tục xác nhận đương nhiên được xóa án tích Điều 70 BLHS Điều 369 BLTTHS TTHS-XAT-070 Sở Tư pháp",
        "Tố tụng hình sự",
    ),
    (
        ["lý lịch tư pháp", "phiếu số 1", "phiếu số 2", "lltp"],
        "Thủ tục yêu cầu cấp Phiếu lý lịch tư pháp số 1 và số 2 Luật Lý lịch tư pháp TTHS-LLTP-045 Sở Tư pháp",
        "Tố tụng hình sự",
    ),
    (
        ["hoãn thi hành án", "hoãn chấp hành án", "hoãn đi tù", "nuôi con dưới 36 tháng", "lao động duy nhất"],
        "Thủ tục xin hoãn chấp hành án phạt tù Điều 67 BLHS Điều 24 Luật Thi hành án hình sự TTHS-THA-067",
        "Tố tụng hình sự",
    ),
    (
        ["khiếu nại tố tụng", "khiếu nại quyết định", "khiếu nại lệnh bắt", "khiếu nại điều tra viên", "không khởi tố"],
        "Thủ tục khiếu nại quyết định hành vi tố tụng trong tố tụng hình sự Điều 475 BLTTHS TTHS-KN-475",
        "Tố tụng hình sự",
    ),
    (
        ["trợ giúp pháp lý", "luật sư miễn phí", "bào chữa miễn phí", "hộ nghèo bị buộc tội"],
        "Thủ tục yêu cầu Trợ giúp pháp lý miễn phí cử Luật sư bào chữa Điều 7 Luật Trợ giúp pháp lý TTHS-TGPL-007",
        "Tố tụng hình sự",
    ),
    (
        ["liên thông khai sinh", "khai sinh liên thông", "3 trong 1", "khai sinh thường trú bhyt"],
        "Thủ tục liên thông điện tử đăng ký khai sinh thường trú cấp thẻ BHYT cho trẻ dưới 6 tuổi Nghị định 63/2024/NĐ-CP TTHC-HT-06",
        "Thủ tục hành chính",
    ),
    (
        ["khai sinh", "đăng ký khai sinh", "giấy chứng sinh", "mới sinh con"],
        "Thủ tục đăng ký khai sinh thông thường Điều 15 Điều 16 Luật Hộ tịch TTHC-HT-01 UBND cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["kết hôn với người nước ngoài", "chồng nước ngoài", "vợ nước ngoài", "quốc tịch hàn", "quốc tịch pháp", "yếu tố nước ngoài"],
        "Thủ tục đăng ký kết hôn có yếu tố nước ngoài Điều 37 Luật Hộ tịch TTHC-HT-05 TTHC-HT-05 UBND cấp huyện",
        "Thủ tục hành chính",
    ),
    (
        ["kết hôn trong nước", "giấy chứng nhận kết hôn", "đăng ký kết hôn"],
        "Thủ tục đăng ký kết hôn trong nước Điều 17 Điều 18 Luật Hộ tịch TTHC-HT-02 UBND cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["độc thân", "tình trạng hôn nhân", "xác nhận độc thân"],
        "Thủ tục cấp Giấy xác nhận tình trạng hôn nhân Nghị định 123/2015/NĐ-CP TTHC-HT-03 UBND cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["khai tử", "giấy báo tử", "chứng tử", "người thân mất"],
        "Thủ tục đăng ký khai tử Điều 32 Điều 34 Luật Hộ tịch TTHC-HT-04 UBND cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["sao y", "chứng thực bản sao", "công chứng bản sao", "sao y bản chính"],
        "Thủ tục chứng thực bản sao từ bản chính Nghị định 23/2015/NĐ-CP TTHC-HT-07 UBND cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["thường trú", "nhập khẩu", "sổ hộ khẩu", "hk07", "ct01"],
        "Thủ tục đăng ký thường trú Luật Cư trú 2020 Điều 20 TTHC-CT-01 Công an cấp xã bãi bỏ Sổ hộ khẩu giấy",
        "Thủ tục hành chính",
    ),
    (
        ["tạm trú", "gia hạn tạm trú", "ở trọ"],
        "Thủ tục đăng ký tạm trú Luật Cư trú 2020 Điều 27 TTHC-CT-02 Công an cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["định danh điện tử", "vneid mức 2", "tài khoản định danh", "kích hoạt vneid"],
        "Thủ tục cấp tài khoản định danh điện tử mức độ 2 VNeID Nghị định 69/2024/NĐ-CP TTHC-CT-03",
        "Thủ tục hành chính",
    ),
    (
        ["tạm vắng", "khai báo tạm vắng", "đi khỏi nơi cư trú"],
        "Thủ tục khai báo tạm vắng Luật Cư trú 2020 Điều 31 TTHC-CT-04 Công an cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["căn cước", "cccd", "cmnd 9 số", "chứng minh nhân dân", "đổi thẻ căn cước"],
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

    # Tầng 1: Khớp COLLOQUIAL_TO_LEGAL_MAP
    for keywords, legal_expansion, domain in COLLOQUIAL_TO_LEGAL_MAP:
        if any(kw in q_lower for kw in keywords):
            expansions.append(legal_expansion)
            if not detected_domain and domain:
                detected_domain = domain

    # Phát hiện domain qua marker từ khóa hình sự
    if not detected_domain:
        criminal_markers = [
            "bị can", "bị cáo", "tạm giam", "tạm giữ", "tội phạm", "khởi tố",
            "điều tra", "viện kiểm sát", "xét xử", "án tích", "kháng cáo", "án phạt tù",
            "bảo lĩnh", "tố giác", "giám định", "lý lịch tư pháp", "bào chữa",
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
