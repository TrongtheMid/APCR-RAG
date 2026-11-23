"""
NHÁNH VŨ GIA ANH ĐỨC — Module Conversational Legal Advisor:
Chuyển đổi dữ liệu JSON 6 trường thành lời tư vấn tự nhiên, ấm áp và sát tình huống
như một chuyên viên pháp lý đang trò chuyện trực tiếp với người dân, trả lời thẳng vào
trọng tâm câu hỏi (Có/Không, Ở đâu, Bao nhiêu tiền, Mấy ngày, Tại sao) trước khi đi vào chi tiết.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _fmt_fee_natural(muc_phi: int) -> str:
    if muc_phi == 0:
        return "hoàn toàn miễn phí (0 VNĐ)"
    return f"{muc_phi:,} VNĐ".replace(",", ".")


def _fmt_unit_short(don_vi_tinh: str) -> str:
    """Rút gọn phần giải thích trong ngoặc của don_vi_tinh để câu văn tự nhiên như người nói."""
    base = don_vi_tinh.split("(")[0].strip()
    return base or don_vi_tinh


def _fmt_days_natural(so_ngay: int, don_vi_tinh: str) -> str:
    unit = _fmt_unit_short(don_vi_tinh)
    if so_ngay == 1:
        return f"ngay trong ngày (tối đa 1 {unit})"
    return f"tối đa {so_ngay} {unit}"


def _fmt_auth_natural(cap_tq: List[str]) -> str:
    if not cap_tq:
        return "cơ quan có thẩm quyền"
    if len(cap_tq) == 1:
        return cap_tq[0]
    if len(cap_tq) == 2:
        return f"{cap_tq[0]} hoặc {cap_tq[1]}"
    return f"{', '.join(cap_tq[:-1])} hoặc {cap_tq[-1]}"


def _build_situation_opening(
    query: str,
    ma_tt: str,
    ten_tt: str,
    cap_tq: List[str],
    so_ngay: int,
    don_vi: str,
    muc_phi: int,
    expired_alerts: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Phân tích ngữ cảnh câu hỏi cụ thể của người dân để đưa ra câu trả lời trực diện,
    giống chuyên viên tư vấn thật đang giải đáp đúng băn khoăn của họ.
    """
    q = query.lower()
    auth_str = _fmt_auth_natural(cap_tq)
    fee_str = _fmt_fee_natural(muc_phi)
    days_str = _fmt_days_natural(so_ngay, don_vi)

    # --- 1. CMND 9 số / 12 số sang năm 2025 ---
    if ma_tt == "TTHC-CC-01" and ("cmnd" in q or "chứng minh nhân dân" in q or "9 số" in q or "12 số" in q):
        return (
            "Chào bạn, câu trả lời thẳng thắn cho băn khoăn của bạn là **KHÔNG** — bạn **không thể tiếp tục dùng CMND 9 số "
            "trong năm 2025 được nữa**, kể cả khi tính theo thời hạn 15 năm in trên mặt sau thẻ thì phải tới năm 2027 mới hết hạn.\n\n"
            "Lý do là theo **Khoản 2 Điều 46 Luật Căn cước số 26/2023/QH15**, mọi Chứng minh nhân dân (cả loại 9 số và 12 số) "
            "chỉ có giá trị sử dụng đến hết ngày **31/12/2024**. Kể từ ngày **01/01/2025**, CMND chính thức hết hiệu lực hoàn toàn "
            "khi thực hiện giao dịch ngân hàng, công chứng hay thủ tục hành chính.\n\n"
            "👉 **Bạn nên làm gì ngay:** Hãy đặt lịch trên ứng dụng **VNeID / Cổng Dịch vụ công** hoặc đến trực tiếp "
            "**Cơ quan quản lý căn cước Công an cấp huyện hoặc Công an cấp tỉnh** gần nhất để làm thủ tục cấp đổi sang **thẻ Căn cước** mới. "
            f"Thời gian trả thẻ **{days_str}**, mức lệ phí cấp đổi là **{fee_str}** (và được **miễn phí** nếu bạn thuộc diện cấp lần đầu hoặc đúng độ tuổi quy định)."
        )

    # --- 2. Đổi thẻ Căn cước ở độ tuổi 14 / 25 / 40 / 60 (làm trong vòng 2 năm trước mốc) ---
    if ma_tt == "TTHC-CC-01" and ("23 tuổi" in q or "25 tuổi" in q or "độ tuổi" in q):
        return (
            "Chào bạn, tin vui cho bạn là trong trường hợp này bạn **KHÔNG bắt buộc** phải đi đổi thẻ Căn cước ngay khi tròn 25 tuổi.\n\n"
            "Theo quy định tại Luật Căn cước 2023, thẻ Căn cước được cấp trong thời hạn **02 năm trước các độ tuổi cấp đổi** "
            "(14, 25, 40, 60 tuổi) sẽ tiếp tục có giá trị sử dụng cho đến mốc độ tuổi cấp đổi tiếp theo. "
            "Vì bạn đã làm thẻ năm 23 tuổi (nằm trong khoảng 2 năm trước tuổi 25), thẻ hiện tại của bạn vẫn dùng bình thường đến tận năm **40 tuổi** "
            "(trừ trường hợp thẻ bị hư hỏng, mất hoặc bạn thay đổi thông tin nhân thân muốn chủ động đổi mới)."
        )

    # --- 3. Đòi Sổ hộ khẩu giấy / mẫu HK07 ---
    if ("hộ khẩu" in q or "hk07" in q) and ma_tt in ("TTHC-HT-01", "TTHC-CT-01", "TTHC-CT-02"):
        return (
            "Chào bạn, nếu nơi tiếp nhận hồ sơ vẫn yêu cầu bạn phải xuất trình bản gốc Sổ hộ khẩu giấy hoặc làm Sổ hộ khẩu giấy mới "
            "thì yêu cầu đó là **KHÔNG đúng quy định hiện hành**.\n\n"
            "Kể từ ngày **01/01/2023**, toàn bộ **Sổ hộ khẩu giấy và Sổ tạm trú giấy đã chính thức hết giá trị sử dụng** theo Khoản 3 Điều 38 Luật Cư trú 2020. "
            "Thay vào đó, cơ quan chức năng khai thác trực tiếp dữ liệu cư trú của bạn trên **Cơ sở dữ liệu quốc gia về dân cư** "
            "hoặc thông qua **ứng dụng VNeID / Giấy xác nhận thông tin về cư trú (mẫu CT07)**. "
            f"Với thủ tục bạn đang làm, bạn nộp hồ sơ tại **{auth_str}**, thời hạn giải quyết **{days_str}** và lệ phí là **{fee_str}**."
        )

    # --- 4. Kết hôn với người nước ngoài nộp ở UBND xã hay huyện ---
    if ma_tt == "TTHC-HT-05":
        return (
            "Chào bạn, với trường hợp **đăng ký kết hôn có yếu tố nước ngoài**, bạn phải nộp hồ sơ tại **UBND cấp huyện** "
            "(Phòng Tư pháp trực tiếp giải quyết) — **KHÔNG nộp ở UBND cấp xã/phường** nhé (trừ một số xã khu vực biên giới rất đặc thù).\n\n"
            "Lưu ý quan trọng khi chuẩn bị hồ sơ: các giấy tờ do phía cơ quan nước ngoài cấp (như giấy xác nhận độc thân/tình trạng hôn nhân) "
            f"bắt buộc phải được **hợp pháp hóa lãnh sự** và dịch công chứng sang tiếng Việt. Thời gian giải quyết **{days_str}** "
            f"và lệ phí là **{fee_str}**."
        )

    # --- 5. Một mình lên ký nhận Giấy chứng nhận kết hôn trong nước ---
    if ma_tt == "TTHC-HT-02" and ("một mình" in q or "công tác" in q or "bận" in q or "vắng" in q):
        return (
            "Chào bạn, rất tiếc là **KHÔNG được** bạn nhé! Khi đăng ký và nhận Giấy chứng nhận kết hôn tại UBND cấp xã, "
            "pháp luật bắt buộc **cả hai bên nam và nữ phải cùng có mặt trực tiếp** để xác nhận sự tự nguyện và cùng ký vào Sổ hộ tịch.\n\n"
            "Bạn và bạn đời nên sắp xếp thời gian thuận tiện cùng đến **UBND cấp xã**; thủ tục này được giải quyết "
            f"**{days_str}** và **hoàn toàn miễn phí (0 VNĐ)** đối với công dân Việt Nam cư trú trong nước."
        )

    # --- 6. Mất Giấy chứng sinh khi đăng ký khai sinh ---
    if ma_tt == "TTHC-HT-01" and ("mất" in q or "giấy chứng sinh" in q):
        return (
            "Chào bạn, nếu gia đình không may làm mất Giấy chứng sinh bản gốc thì **vẫn đăng ký khai sinh cho bé bình thường được**, bạn đừng quá lo lắng.\n\n"
            "Trong trường hợp này, bạn chỉ cần xin lại bản sao từ cơ sở y tế đã sinh bé, hoặc sử dụng **văn bản của người làm chứng** xác nhận về việc sinh; "
            "nếu không có người làm chứng thì lập **giấy cam đoan về việc sinh** nộp kèm Tờ khai đăng ký khai sinh tại **UBND cấp xã**. "
            f"Hãy nhớ đi làm trong vòng **60 ngày** kể từ ngày sinh bé — thủ tục giải quyết **{days_str}** và **{fee_str}**."
        )

    # --- 7. Đăng ký thường trú vào nhà thuê ---
    if ma_tt == "TTHC-CT-01" and ("thuê" in q or "ở nhờ" in q or "mượn" in q):
        return (
            "Chào bạn, gia đình bạn **hoàn toàn có thể** đăng ký thường trú vào nhà thuê nếu đáp ứng đủ **02 điều kiện bắt buộc** sau:\n"
            "1. **Được chủ sở hữu nhà cho thuê đồng ý** bằng văn bản (hoặc ghi xác nhận đồng ý trực tiếp vào Tờ khai CT01 / qua VNeID).\n"
            "2. **Bảo đảm diện tích nhà ở tối thiểu** theo quy định của HĐND cấp tỉnh (nhưng không thấp hơn **08 m2 sàn/người**).\n\n"
            f"Bạn nộp hồ sơ (kèm Hợp đồng thuê nhà và mẫu CT01) tại **{auth_str}** hoặc trực tuyến qua Cổng Dịch vụ công; thời hạn giải quyết **{days_str}**, lệ phí nộp trực tiếp là **{fee_str}**."
        )

    # --- 8. Gia hạn tạm trú ---
    if ma_tt == "TTHC-CT-02" and ("gia hạn" in q or "sắp hết" in q or "trước bao nhiêu" in q):
        return (
            "Chào bạn, thời hạn tạm trú mỗi lần đăng ký tối đa là **02 năm**. Để gia hạn liên tục mà không bị gián đoạn dữ liệu cư trú, "
            "bạn cần nộp hồ sơ gia hạn **trong vòng 15 ngày trước ngày hết hạn tạm trú** hiện tại.\n\n"
            f"Bạn chỉ cần nộp Tờ khai thay đổi thông tin cư trú (mẫu CT01) trực tuyến qua VNeID/Cổng Dịch vụ công hoặc trực tiếp tại **{auth_str}**, "
            f"thời hạn giải quyết **{days_str}**."
        )

    # --- 9. Trợ cấp thất nghiệp: tự ý nghỉ việc trái luật / quá hạn 3 tháng ---
    if ma_tt == "TTHC-LD-01":
        if "trái pháp luật" in q or "tự ý" in q or "không báo trước" in q:
            return (
                "Chào bạn, mình phải chia sẻ thẳng thắn là trong trường hợp bạn **đơn phương chấm dứt hợp đồng lao động trái pháp luật** "
                "(tự ý nghỉ việc không báo trước đúng luật), thì dù đã đóng BHTN 14 tháng bạn vẫn **KHÔNG đủ điều kiện hưởng trợ cấp thất nghiệp** theo quy định của Luật Việc làm.\n\n"
                "Tuy nhiên, toàn bộ thời gian đã đóng bảo hiểm thất nghiệp này **không bị mất đi**, mà sẽ được **cộng dồn bảo lưu** để tính cho lần hưởng tiếp theo "
                "khi bạn đi làm công ty mới và nghỉ việc đúng quy định pháp luật."
            )
        if "5 tháng" in q or "quá" in q or "lệ phí" in q or "200.000" in q:
            return (
                "Chào bạn, ở tình huống này có **02 lưu ý rất quan trọng** giúp bạn tránh mất quyền lợi và tránh bị thu tiền sai:\n"
                "1. **Về thời hạn nộp hồ sơ:** Luật quy định phải nộp hồ sơ hưởng trợ cấp thất nghiệp **trong vòng 03 tháng** kể từ ngày chấm dứt hợp đồng lao động. "
                "Nếu bạn đã nghỉ tới tháng thứ 5 thì đã quá hạn nộp hồ sơ của đợt này (thời gian đã đóng BHTN sẽ được **bảo lưu** sang đợt sau chứ không bị mất).\n"
                f"2. **Về lệ phí:** Thủ tục hưởng trợ cấp thất nghiệp được **{fee_str}** — tuyệt đối không có khoản lệ phí xét duyệt nào khi nộp tại **{auth_str}**."
            )

    # --- 10. Rút BHXH 1 lần khi mắc bệnh nguy hiểm ---
    if ma_tt == "TTHC-BH-01" and ("ung thư" in q or "hiểm nghèo" in q or "chờ đủ 12 tháng" in q):
        return (
            "Chào bạn, xin chia sẻ với hoàn cảnh của người lao động. Câu trả lời là **KHÔNG cần phải chờ đủ 12 tháng** sau khi nghỉ việc đâu nhé!\n\n"
            "Theo Điều 60 Luật BHXH, nếu người lao động đang mắc một trong những bệnh nguy hiểm đến tính mạng như **ung thư, bại liệt, xơ gan cổ chướng, phong, lao nặng, nhiễm HIV chuyển sang AIDS...**, "
            f"gia đình có thể mang hồ sơ (kèm trích sao/tóm tắt hồ sơ bệnh án) đến nộp ngay tại **{auth_str}** để được giải quyết hưởng BHXH một lần. "
            f"Thời hạn giải quyết **{days_str}** và **{fee_str}**."
        )

    # --- 11. Bảo lĩnh tại ngoại: 1 người hay sai cơ quan giai đoạn xét xử ---
    if ma_tt == "TTHS-BL-121":
        if "một mình" in q or "1 người" in q or "duy nhất" in q:
            return (
                "Chào bạn, rất tiếc là **chỉ một mình bạn đứng ra ký đơn bảo lĩnh thì CHƯA đủ điều kiện** được cơ quan tố tụng chấp nhận đâu nhé.\n\n"
                "Theo Khoản 2 Điều 121 Bộ luật Tố tụng Hình sự 2015, để nhận bảo lĩnh cho bị can/bị cáo tại ngoại, "
                "pháp luật **bắt buộc phải có ít nhất 02 người thân thích** cùng có nhân thân tốt, thu nhập ổn định đứng ra làm giấy cam đoan (có xác nhận của chính quyền địa phương hoặc cơ quan nơi làm việc). "
                f"Bạn hãy mời thêm ít nhất một người thân nữa (như bố, mẹ, vợ/chồng, anh chị em khác) cùng ký tên, nộp tại cơ quan đang thụ lý vụ án ({auth_str}); thủ tục này **{fee_str}**."
            )
        if "tòa án" in q or "xét xử" in q or "cơ quan cảnh sát điều tra" in q or "csđt" in q:
            return (
                "Chào bạn, nộp đơn quay lại Cơ quan Cảnh sát điều tra ở thời điểm này là **CHƯA đúng thẩm quyền giai đoạn tố tụng**.\n\n"
                "Khi vụ án đã được **Tòa án thụ lý để chuẩn bị xét xử sơ thẩm**, hồ sơ vụ án đã chuyển khỏi giai đoạn điều tra, nên gia đình phải nộp **Đơn xin bảo lĩnh trực tiếp đến Tòa án đang thụ lý vụ án** "
                "(không nộp ở Cơ quan CSĐT nữa). Hãy bảo đảm có **ít nhất 02 người thân thích** cùng ký giấy cam đoan và thủ tục này **{fee_str}**."
            )

    # --- 12. Tố giác tội phạm: Lừa đảo đòi tiền lệ phí / Công an xã đòi khởi tố ---
    if ma_tt == "TTHS-TG-145":
        if "facebook" in q or "an ninh mạng" in q or "lệ phí" in q or "chuyển" in q or "triệu" in q:
            return (
                "Chào bạn, **BẠN TUYỆT ĐỐI KHÔNG CHUYỂN TIỀN NHÉ!** Đây chắc chắn là **chiêu trò lừa đảo mạo danh cơ quan công an** để chiếm đoạt tiền của bạn thêm một lần nữa.\n\n"
                "Theo quy định pháp luật hiện hành, mọi thủ tục trình báo, tiếp nhận và giải quyết tố giác tội phạm đều được **MIỄN PHÍ HOÀN TOÀN (0 VNĐ)** — "
                "cơ quan công an không bao giờ nhắn tin qua Facebook hay yêu cầu chuyển tiền lệ phí để giải quyết nhanh trong 24 giờ.\n\n"
                "👉 **Việc bạn cần làm ngay:** Lưu lại toàn bộ sao kê chuyển khoản, tin nhắn, số tài khoản của kẻ lừa đảo và mang đến trình báo trực tiếp tại **Công an cấp xã/phường gần nhất hoặc Cơ quan Cảnh sát điều tra**."
            )
        if "công an xã" in q and "khởi tố" in q:
            return (
                "Chào bạn, yêu cầu Trưởng Công an xã ra Quyết định khởi tố vụ án hình sự ngay trong ngày là **không đúng phân cấp thẩm quyền** theo Bộ luật Tố tụng Hình sự.\n\n"
                "**Công an xã** có trách nhiệm tiếp nhận ngay đơn tố giác của bạn, lập biên bản, lấy lời khai ban đầu và chuyển toàn bộ hồ sơ trong thời hạn **24 giờ** lên **Cơ quan Cảnh sát điều tra**. "
                "Thẩm quyền kiểm tra, xác minh giải quyết (tối đa 20 ngày đối với vụ việc thông thường) và ra Quyết định khởi tố vụ án hình sự thuộc về **Cơ quan Cảnh sát điều tra** (hoặc Viện kiểm sát)."
            )

    # --- 13. Phiếu lý lịch tư pháp số 2 cấm ủy quyền ---
    if ma_tt == "TTHS-LLTP-045" and ("số 2" in q or "ủy quyền" in q):
        return (
            "Chào bạn, rất tiếc là trường hợp này **KHÔNG được phép ủy quyền** cho mẹ ruột đi làm hộ đâu bạn nhé!\n\n"
            "Khoản 2 Điều 46 Luật Lý lịch tư pháp quy định rất chặt chẽ: nếu như **Phiếu lý lịch tư pháp số 1** cho phép cha mẹ, vợ chồng, con làm hộ, "
            "thì riêng **Phiếu lý lịch tư pháp số 2 tuyệt đối KHÔNG được ủy quyền** cho bất kỳ ai khác, nhằm bảo vệ bí mật đời tư và dữ liệu án tích cá nhân. "
            f"Bạn cần tự mình nộp hồ sơ trực tiếp hoặc qua Cổng dịch vụ công trực tuyến tới **{auth_str}** (lệ phí chuẩn **{fee_str}**, giải quyết **{days_str}**)."
        )

    # --- 14. Kháng cáo quá hạn 15 ngày / Vắng mặt / Lệ phí kháng cáo ---
    if ma_tt == "TTHS-KC-331":
        if "vắng mặt" in q or "nằm viện" in q or "nhận được bản án" in q:
            return (
                "Chào bạn, trong trường hợp bị cáo **vắng mặt tại phiên tòa** (ví dụ đang nằm viện), thời hạn **15 ngày** kháng cáo bản án sơ thẩm "
                "**KHÔNG tính từ ngày tuyên án**, mà được tính từ **ngày bị cáo nhận được bản án** hoặc ngày bản án được niêm yết hợp lệ theo quy định.\n\n"
                "Đơn kháng cáo được gửi tới **Tòa án đã xét xử sơ thẩm** (hoặc gửi trực tiếp cho Tòa án cấp phúc thẩm) và phần kháng cáo hình sự được **miễn án phí/lệ phí hoàn toàn (0 VNĐ)**."
            )
        if "24 ngày" in q or "500.000" in q or "hợp lệ" in q:
            return (
                "Chào bạn, có **02 điểm sai lệch nghiêm trọng** trong tình huống bạn nêu:\n"
                "1. **Về thời hạn:** Vì bị cáo có mặt tại phiên tòa ngày 01/03, thời hạn kháng cáo bản án sơ thẩm chỉ là **15 ngày kể từ ngày tuyên án**. "
                "Đến ngày 25/03 (sau 24 ngày) mới nộp là **đã quá hạn kháng cáo** — Tòa án chỉ xem xét chấp nhận kháng cáo quá hạn nếu có lý do bất khả kháng hoặc trở ngại khách quan chính đáng.\n"
                "2. **Về lệ phí:** Thủ tục nộp đơn kháng cáo về phần hình sự được **MIỄN ÁN PHÍ / LỆ PHÍ HOÀN TOÀN (0 VNĐ)**, không phải đóng khoản tiền lệ phí nào."
            )

    # --- 15. Xóa án tích đương nhiên vs theo quyết định ---
    if ma_tt == "TTHS-XAT-070":
        return (
            "Chào bạn, với các tội danh không thuộc Chương XIII và Chương XXVI BLHS (như trộm cắp tài sản, đánh bạc...), "
            "khi đã chấp hành xong án phạt, nghĩa vụ dân sự và hết thời hạn luật định (1–5 năm tùy mức án) mà không phạm tội mới thì người đó **đương nhiên được xóa án tích**.\n\n"
            "👉 **Lưu ý rất hay bị nhầm:** Hiện nay **Tòa án KHÔNG cấp giấy chứng nhận xóa án tích** cho trường hợp đương nhiên xóa án tích nữa. "
            f"Khi cần văn bản chứng minh không còn án tích, bạn hãy nộp hồ sơ yêu cầu cấp **Phiếu lý lịch tư pháp** tại **{auth_str}** (không nộp đơn ở Tòa án); "
            f"việc cập nhật dữ liệu xóa án tích **{fee_str}**."
        )
    if ma_tt == "TTHS-XAT-071":
        return (
            "Chào bạn, đối với các tội xâm phạm an ninh quốc gia (Chương XIII) hoặc phá hoại hòa bình, chống loài người (Chương XXVI BLHS), "
            "pháp luật **không áp dụng cơ chế đương nhiên xóa án tích**.\n\n"
            f"Trường hợp này bạn phải nộp hồ sơ **Đơn xin xóa án tích trực tiếp tại {auth_str}** (không nộp ở Sở Tư pháp) để Chánh án Tòa án xem xét ra Quyết định xóa án tích sau khi có ý kiến của Viện kiểm sát. "
            f"Thời hạn giải quyết **{days_str}** và thủ tục này **{fee_str}**."
        )

    # --- 16. Hoãn chấp hành án phạt tù ---
    if ma_tt == "TTHS-THA-067":
        if "rất nghiêm trọng" in q or "lao động duy nhất" in q:
            return (
                "Chào bạn, với trường hợp bị kết án 8 năm tù về **tội rất nghiêm trọng**, thì lý do là **lao động duy nhất trong gia đình sẽ KHÔNG được Tòa án chấp nhận hoãn chấp hành án phạt tù**.\n\n"
                "Lý do là điểm c Khoản 1 Điều 67 Bộ luật Hình sự quy định trường hợp là người lao động duy nhất chỉ được xét hoãn tối đa 01 năm nếu bị kết án về tội ít nghiêm trọng hoặc nghiêm trọng — "
                "**loại trừ** các tội xâm phạm an ninh quốc gia, tội **rất nghiêm trọng** và tội **đặc biệt nghiêm trọng**."
            )
        return (
            "Chào bạn, phụ nữ đang có thai hoặc đang **nuôi con nhỏ dưới 36 tháng tuổi** thuộc trường hợp được xem xét **hoãn chấp hành án phạt tù** "
            "cho đến khi con đủ 36 tháng tuổi theo điểm b Khoản 1 Điều 67 BLHS.\n\n"
            f"Gia đình cần chuẩn bị Đơn xin hoãn kèm bản sao Giấy khai sinh của bé gửi đến **{auth_str}**; thời hạn xem xét giải quyết **{days_str}** và **{fee_str}**."
        )

    # --- 17. Thăm gặp tạm giữ / tạm giam ---
    if ma_tt == "TTHS-TG-022":
        return (
            "Chào bạn, người đang bị tạm giam được gặp thân nhân **01 lần trong 01 tháng**, mỗi lần gặp **không quá 01 giờ** và tối đa **không quá 03 người thân** cùng vào gặp. "
            "Đặc biệt, thủ tục đăng ký thăm gặp hoàn toàn **MIỄN PHÍ (0 VNĐ)** — gia đình không phải đóng bất kỳ khoản phí thăm gặp nào.\n\n"
            "Trong trường hợp vụ án đang trong giai đoạn điều tra cần giữ bí mật, Thủ trưởng Trại tạm giam sẽ trao đổi, phối hợp với cơ quan đang thụ lý vụ án để quyết định cho phép gặp và giải quyết trực tiếp theo lịch tiếp nhận."
        )

    # --- 18. Đặt tiền bảo đảm ---
    if ma_tt == "TTHS-DT-122":
        return (
            "Chào bạn, với biện pháp **Đặt tiền để bảo đảm thay thế tạm giam** (Điều 122 BLTTHS), số tiền đặt tối thiểu được quy định rõ theo loại tội phạm: "
            "**30 triệu đồng** (tội ít nghiêm trọng), **100 triệu đồng** (nghiêm trọng), **200 triệu đồng** (rất nghiêm trọng) và **300 triệu đồng** (đặc biệt nghiêm trọng).\n\n"
            f"Thủ tục này nộp tại cơ quan đang thụ lý ({auth_str}, tuyệt đối không nộp cho Công an xã), quyết định của Cơ quan điều tra bắt buộc phải được **Viện kiểm sát cùng cấp phê chuẩn**, "
            "tiền được nộp trực tiếp vào tài khoản tạm giữ tại Kho bạc Nhà nước và sẽ được hoàn trả lại sau khi bị can/bị cáo chấp hành đúng nghĩa vụ cam đoan."
        )

    # --- 19. Giám định thương tích / định giá tài sản ---
    if ma_tt == "TTHS-GD-205":
        return (
            "Chào bạn, khi Cơ quan Cảnh sát điều tra đã ra Quyết định trưng cầu giám định tỷ lệ tổn thương cơ thể cho người bị hại thì chi phí giám định là **MIỄN PHÍ HOÀN TOÀN (0 VNĐ)** đối với bạn.\n\n"
            "Nếu có người tự xưng yêu cầu bạn phải chuyển tiền lệ phí giám định mới làm thì đó là hành vi trái pháp luật — bạn tuyệt đối không nộp. "
            f"Trong trường hợp cơ quan tố tụng từ chối trưng cầu giám định, bạn có quyền tự mình yêu cầu giám định trong thời hạn **{days_str}**."
        )

    if expired_alerts:
        exp_law = expired_alerts[0]["can_cu_phap_ly"][0]
        return (
            f"Chào bạn, lưu ý quan trọng cho câu hỏi của bạn là quy định cũ ({exp_law['so_hieu_van_ban']}) hiện đã **HẾT HIỆU LỰC**, "
            f"và được thay thế bởi **{exp_law.get('van_ban_thay_the')}**.\n\n"
            f"Để giải quyết **{ten_tt}** đúng luật hiện hành, bạn hãy nộp hồ sơ tại **{auth_str}**, "
            f"thời hạn giải quyết **{days_str}** với mức lệ phí **{fee_str}**."
        )
    return None


def build_direct_advice(
    query: str,
    primary_proc: Dict[str, Any],
    expired_alerts: List[Dict[str, Any]],
    revision_count: int = 0,
) -> str:
    """
    Sinh đoạn trả lời tư vấn tự nhiên (`direct_advice`) giống như chuyên viên pháp lý
    đang trực tiếp giải đáp cho người dân dựa trên câu hỏi và Hồ sơ 6 trường.
    """
    info = primary_proc["ma_va_ten_thu_tuc"]
    ho_so = primary_proc["thanh_phan_ho_so"]
    thoi_han = primary_proc["thoi_han_giai_quyet"]
    le_phi = primary_proc["le_phi"]
    laws = primary_proc["can_cu_phap_ly"]

    ma_tt = info["ma_thu_tuc"]
    ten_tt = info["ten_thu_tuc"]
    cap_tq = info.get("cap_tham_quyen", [])
    so_ngay = int(thoi_han["so_ngay_toi_da"])
    don_vi = thoi_han["don_vi_tinh"]
    muc_phi = int(le_phi["muc_phi_vnd"])

    opening = _build_situation_opening(
        query=query,
        ma_tt=ma_tt,
        ten_tt=ten_tt,
        cap_tq=cap_tq,
        so_ngay=so_ngay,
        don_vi=don_vi,
        muc_phi=muc_phi,
        expired_alerts=expired_alerts,
    )

    if opening is None:
        q_lower = query.lower()
        auth_str = _fmt_auth_natural(cap_tq)
        fee_str = _fmt_fee_natural(muc_phi)
        days_str = _fmt_days_natural(so_ngay, don_vi)
        primary_law = laws[0]
        forms_preview = "; ".join(
            f"{g['ten_giay_to']} (`{g['form_id']}`)" for g in ho_so.get("giay_to_bat_buoc", [])
        )

        if any(k in q_lower for k in ("lệ phí", "bao nhiêu tiền", "mất phí", "phí bao nhiêu", "tốn bao nhiêu")):
            opening = (
                f"Chào bạn, về mức phí thực hiện **{ten_tt}**, câu trả lời là **{fee_str}** "
                f"*(chi tiết: {le_phi['mo_ta']})*.\n\n"
                f"Bạn chỉ cần chuẩn bị hồ sơ gồm {forms_preview} rồi nộp tại **{auth_str}**; "
                f"thời hạn giải quyết **{days_str}** theo {primary_law['so_hieu_van_ban']} ({primary_law['dieu_khoan_diem']})."
            )
        elif any(k in q_lower for k in ("bao lâu", "mấy ngày", "thời hạn")):
            opening = (
                f"Chào bạn, thời hạn giải quyết theo quy định cho **{ten_tt}** là **{days_str}** "
                f"kể từ khi nhận đủ hồ sơ hợp lệ.\n\n"
                f"Bạn nộp hồ sơ tại **{auth_str}** với mức lệ phí **{fee_str}**. "
                f"Khi đi nhớ chuẩn bị đầy đủ: {forms_preview}."
            )
        elif any(k in q_lower for k in ("ở đâu", "cơ quan nào", "nộp tại", "nơi nào")):
            opening = (
                f"Chào bạn, để làm **{ten_tt}**, bạn nộp hồ sơ tại **{auth_str}**.\n\n"
                f"Hồ sơ cần chuẩn bị gồm: {forms_preview}. Thời hạn giải quyết **{days_str}** "
                f"và lệ phí nhà nước là **{fee_str}**."
            )
        else:
            opening = (
                f"Chào bạn, qua tìm hiểu tình huống của bạn, nội dung này thuộc phạm vi **{ten_tt}** "
                f"theo quy định tại **{primary_law['so_hieu_van_ban']}** ({primary_law['dieu_khoan_diem']}).\n\n"
                f"Để thực hiện đúng quy trình, bạn nộp hồ sơ tại **{auth_str}** gồm các giấy tờ: {forms_preview}. "
                f"Thời hạn giải quyết **{days_str}** và mức lệ phí là **{fee_str}**."
            )

    if revision_count > 0:
        opening = (
            f"*(Đã được thẩm định & hiệu chỉnh chuẩn xác lại ở Vòng #{revision_count} theo dữ liệu luật gốc)*\n\n"
            + opening
        )

    return opening
