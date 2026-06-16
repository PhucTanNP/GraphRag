"""Question Bank — quản lý câu hỏi mẫu và embedding của chúng.

Dùng để semantic matching:
  User query → embed → cosine similarity với tất cả câu mẫu → best intent
"""

import os
import pickle
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ─── Question-Intent Mapping ────────────────────────────────────────────
# Mỗi intent có nhiều câu hỏi mẫu (variants) để tăng độ chính xác matching

QUESTION_BANK = {
    # ── SPECS (thông số kỹ thuật chung) ~30 câu ─────────────────
    "SPECS": [
        "lốp 120/70-17 thông số thế nào",
        "lốp 2.50-17 có những thông số gì",
        "cho mình biết thông tin lốp 100/90-18",
        "lốp 110/80-14 chiều rộng bao nhiêu",
        "thông số kỹ thuật lốp 2.75-17",
        "lốp 90/90-14 specs",
        "thông tin lốp 3.00-18",
        "lốp 140/70-14 specs thế nào",
        "lốp 80/80-14 thông số",
        "kích thước lốp 100/80-14",
        "lốp 90/80-14 có thông số gì",
        "thông tin chi tiết lốp 110/70-14",
        "lốp 120/80-14 specs",
        "các thông số lốp 3.50-10",
        "lốp 70/90-17 specs thế nào",
        "thông tin lốp 2.50 r17",
        "lốp 3.00-18 đường kính ngoài",
        "lốp 100/80-14 vành bao nhiêu",
        "thông số lốp 90/80-14",
        "kích thước lốp 110/70-14",
        "lốp 120/70-17 cấu trúc gì",
        "lốp 2.75-17 thông số chiều rộng",
        "lốp 140/70-14 đường kính",
        "chiều rộng lốp 80/80-14",
        "thông số kỹ thuật lốp 70/90-17",
        "lốp 3.50-10 thông tin",
        "specs lốp 2.50-17",
        "lốp 90/90-14 vành bao nhiêu",
        "thông số lốp 110/70-14 chiều cao",
        "cấu tạo lốp 120/70-17",
    ],
    # ── SPEED (tốc độ tối đa) ~30 câu ──────────────────────────────
    "SPEED": [
        "tốc độ tối đa của lốp 120/70-17",
        "lốp 100/80-14 chạy được bao nhiêu km/h",
        "tốc độ lốp 90/90-14",
        "lốp nào nhanh nhất",
        "lốp 110/70-14 tốc độ bao nhiêu",
        "tốc độ tối đa lốp này là bao nhiêu",
        "lốp này chạy được tối đa bao nhiêu km/h",
        "cho tôi biết tốc độ lốp 2.50-17",
        "lốp 80/80-14 chạy nhanh nhất bao nhiêu",
        "tốc độ tối đa lốp 3.00-18",
        "lốp 2.75-17 chạy được bao nhiêu km",
        "lốp 70/90-17 tốc độ",
        "tốc độ lốp 120/80-14",
        "lốp 3.50-10 speed tối đa",
        "tốc độ tối đa lốp 140/70-14",
        "lốp 100/90-18 chạy bao nhiêu km",
        "vận tốc tối đa lốp 90/80-14",
        "lốp 110/80-14 max speed",
        "cho hỏi tốc độ lốp 2.50-17",
        "lốp 120/70-17 chạy được bao nhiêu km",
        "tốc độ tối đa cho lốp 3.00-18",
        "lốp 80/80-14 tối đa bao nhiêu km",
        "lốp 2.75-17 max speed là bao nhiêu",
        "tốc độ tối đa lốp 100/80-14",
        "lốp 110/70-14 chạy được bao nhiêu km",
        "lốp này vận tốc tối đa",
        "tốc độ của lốp 70/90-17",
        "lốp 140/70-14 chạy được bao nhiêu",
        "max speed lốp 90/90-14",
        "vận tốc lốp 120/80-14",
    ],
    # ── LOAD (tải trọng) ~30 câu ───────────────────────────────────
    "LOAD": [
        "tải trọng lốp 120/70-17",
        "lốp 2.50-17 chịu tải bao nhiêu kg",
        "lốp 100/90-18 tải trọng tối đa",
        "lốp nào chịu tải cao nhất",
        "lốp 110/80-14 chở được bao nhiêu kg",
        "tải trọng tối đa của lốp này",
        "lốp này chịu tải được bao nhiêu",
        "lốp 80/80-14 chịu tải bao nhiêu",
        "lốp 100/80-14 chở tối đa bao nhiêu kg",
        "tải trọng lốp 3.00-18",
        "lốp 2.75-17 chịu được bao nhiêu kg",
        "lốp 120/80-14 tải trọng",
        "lốp 90/80-14 chở được mấy kg",
        "lốp 70/90-17 tải trọng bao nhiêu",
        "lốp 110/70-14 chịu tải",
        "tải trọng lốp 3.50-10",
        "lốp 140/70-14 chở bao nhiêu kg",
        "lốp 120/70-17 chở nặng được không",
        "lốp 2.50-17 tải trọng tối đa",
        "lốp 100/80-14 tải trọng",
        "lốp 90/90-14 chịu tải bao nhiêu",
        "cho hỏi tải trọng lốp 3.00-18",
        "lốp 80/80-14 max load",
        "lốp 2.50-17 chở nặng được không",
        "tải trọng tối đa lốp 110/70-14",
        "lốp 2.75-17 tải trọng",
        "lốp 100/90-18 chịu được bao nhiêu kg",
        "lốp 90/80-14 tải bao nhiêu",
        "lốp 70/90-17 max load",
        "lốp 3.50-10 chịu tải",
    ],
    # ── PRICE (giá cả) ~30 câu ─────────────────────────────────────
    "PRICE": [
        "giá lốp 120/70-17 bao nhiêu",
        "lốp 100/80-14 giá bao nhiêu tiền",
        "báo giá lốp 2.50-17",
        "lốp rẻ nhất",
        "lốp 3.00-18 giá",
        "lốp này giá bao nhiêu",
        "cho mình hỏi giá lốp 90/90-14",
        "lốp 80/80-14 giá bán",
        "lốp 110/70-14 giá rẻ không",
        "bảng giá lốp DPLUS",
        "giá lốp 140/70-14",
        "lốp 120/80-14 bao nhiêu tiền",
        "lốp DRC giá thế nào",
        "lốp 70/90-17 giá bao nhiêu",
        "báo giá lốp 100/90-18",
        "lốp 2.75-17 giá",
        "lốp 110/80-14 giá bán bao nhiêu",
        "lốp 90/80-14 giá rẻ không",
        "bảng giá lốp DRC",
        "lốp 3.50-10 giá",
        "lốp 100/80-14 có đắt không",
        "giá lốp 120/70-17 rẻ nhất",
        "lốp nào giá rẻ nhất DPLUS",
        "giá lốp 90/90-14 bao nhiêu",
        "cho tôi báo giá lốp 120/80-14",
        "lốp 140/70-14 giá niêm yết",
        "lốp DPLUS giá bao nhiêu",
        "giá lốp 2.50-17 hiện tại",
        "lốp 80/80-14 giá mới nhất",
        "lốp 70/90-17 giá rẻ",
    ],
    # ── COMPARE (so sánh) ~30 câu ──────────────────────────────────
    "COMPARE": [
        "so sánh lốp 120/70-17 và 110/70-17",
        "lốp 2.50-17 vs 2.75-17",
        "khác nhau giữa lốp 90/90-14 và 100/80-14",
        "nên mua lốp 120/70-17 hay 110/70-17",
        "so sánh 2.50 và 2.75",
        "so sánh lốp này với lốp 100/80-14",
        "so sánh lốp DPLUS và DRC",
        "lốp nào tốt hơn 90/90-14 hay 100/80-14",
        "so sánh giá lốp 2.50-17 và 2.75-17",
        "lốp 110/70-14 so với 120/70-17",
        "lốp 3.00-18 vs 3.50-10",
        "so sánh lốp 80/80-14 và 90/90-14",
        "nên lấy lốp 100/80-14 hay 110/70-14",
        "so sánh thông số lốp 120/70-17 và 2.50-17",
        "khác biệt lốp DPLUS và DRC",
        "so sánh tốc độ lốp 2.50 và 2.75",
        "lốp 70/90-17 và 90/90-14 khác gì",
        "so sánh giá lốp DPLUS và DRC",
        "lốp 140/70-14 so với 120/70-17",
        "lốp nào tốt hơn DPLUS hay DRC",
        "so sánh chiều rộng lốp 100/80 và 110/70",
        "khác nhau giữa 3.00-18 và 3.50-10",
        "so sánh lốp 100/90-18 và 120/80-14",
        "nên mua lốp DRC hay DPLUS cho xe",
        "so sánh 2.50-17 và 2.75-17 thông số",
        "lốp 80/80-14 vs 90/80-14",
        "so sánh áp suất lốp 2.50 và 2.75",
        "lốp nào to hơn 110/70-14 hay 120/70-17",
        "so sánh tải trọng lốp DPLUS và DRC",
        "khác biệt giữa lốp 90/90-14 và 100/80-14",
    ],
    # ── PRESSURE (áp suất hơi) ~30 câu ─────────────────────────────
    "PRESSURE": [
        "áp suất lốp 120/70-17",
        "bơm lốp 100/80-14 bao nhiêu kg",
        "áp suất tiêu chuẩn lốp 2.50-17",
        "lốp 90/90-14 bơm bao nhiêu psi",
        "áp suất lốp bao nhiêu",
        "lốp 80/80-14 bơm bao nhiêu",
        "áp suất lốp 3.00-18 chuẩn",
        "lốp 110/70-14 non hơi bao nhiêu",
        "bơm lốp 2.75-17 bao nhiêu kg",
        "áp suất lốp 140/70-14",
        "lốp 120/70-17 bơm bao nhiêu kg",
        "áp suất lốp 70/90-17",
        "lốp 100/90-18 non hơi bao nhiêu",
        "bơm lốp 3.50-10 bao nhiêu kg",
        "áp suất lốp 110/80-14",
        "lốp 90/80-14 bơm psi",
        "áp suất lốp 2.50-17 tiêu chuẩn",
        "lốp 120/80-14 bơm bao nhiêu",
        "non hơi lốp 3.00-18 bao nhiêu",
        "bơm lốp 80/80-14 áp suất",
        "áp suất tiêu chuẩn lốp 110/70-14",
        "lốp 2.75-17 bơm hơi bao nhiêu",
        "áp suất lốp 100/80-14 psi",
        "lốp 140/70-14 bơm bao nhiêu kg",
        "non hơi lốp 70/90-17",
        "bơm lốp 120/70-17 bao nhiêu psi",
        "áp suất cho lốp 90/90-14",
        "lốp 3.50-10 áp suất chuẩn",
        "áp suất lốp 100/90-18 bao nhiêu",
        "bơm lốp 80/80-14 bao nhiêu kg",
    ],
    # ── BRAND (thương hiệu) ~30 câu ────────────────────────────────
    "BRAND": [
        "lốp DPLUS có tốt không",
        "thương hiệu lốp nào bền nhất",
        "lốp IRC giá bao nhiêu",
        "lốp MAXXIS chất lượng",
        "các hãng lốp xe máy",
        "lốp hãng nào tốt",
        "lốp DRC chất lượng không",
        "so sánh DPLUS và DRC",
        "lốp nào bền nhất hiện nay",
        "thương hiệu lốp xe máy nào uy tín",
        "review lốp DPLUS",
        "nên mua lốp DRC hay DPLUS",
        "lốp hãng DRC có bền không",
        "thương hiệu lốp xe máy tốt nhất",
        "lốp DPLUS giá và chất lượng",
        "các hãng lốp xe máy tại việt nam",
        "lốp DRC có tốt hơn DPLUS không",
        "nên chọn lốp hãng nào cho xe",
        "lốp IRC có bền không",
        "lốp MAXXIS và DPLUS hãng nào hơn",
        "so sánh DPLUS với IRC",
        "thương hiệu lốp nào được ưa chuộng",
        "review lốp xe máy DRC",
        "lốp DPLUS có đáng mua không",
        "lốp DRC sản xuất ở đâu",
        "hãng lốp DPLUS xem thế nào",
        "nên mua lốp DRC cho xe số",
        "lốp MAXXIS giá cao không",
        "đánh giá lốp xe máy DPLUS",
        "lốp IRC chất lượng ra sao",
    ],
    # ── MAX_LOAD (tải trọng cao nhất) ~30 câu ──────────────────────
    "MAX_LOAD": [
        "lốp nào chịu tải cao nhất",
        "lốp chịu tải tốt nhất",
        "lốp nào tải trọng lớn nhất",
        "lốp chở nặng tốt nhất",
        "tải trọng lớn nhất là lốp nào",
        "lốp nào chở được nhiều nhất",
        "lốp nào có tải trọng cao nhất DPLUS",
        "lốp nào tải trọng max",
        "lốp chở nặng nhất hiện nay",
        "tải trọng tối đa cao nhất",
        "lốp nào chịu được tải nặng nhất",
        "lốp nào load lớn nhất",
        "top lốp tải trọng cao",
        "lốp nào chở nặng nhất DRC",
        "lốp có tải trọng lớn nhất",
        "tải trọng max trong các lốp",
        "lốp nào gánh được nặng nhất",
        "lốp chịu tải cao nhất DPLUS",
        "lốp nào có tải lớn",
        "lốp nào chở được đồ nhiều",
        "lốp bán tải trọng lớn",
        "lốp nào chịu lực tốt nhất",
        "max load lốp nào to nhất",
        "lốp nào tải cao cho xe tải",
        "lốp nào mạnh nhất về tải trọng",
        "tải trọng lớn thuộc lốp nào",
        "lốp nào max load ấn tượng",
        "lốp nào chở nặng an toàn nhất",
        "lốp nào có chỉ số tải cao",
        "lốp nào siêu tải",
    ],
    # ── MAX_SPEED (tốc độ cao nhất) ~30 câu ────────────────────────
    "MAX_SPEED": [
        "lốp nào nhanh nhất",
        "lốp tốc độ cao nhất",
        "lốp nào chạy nhanh nhất",
        "tốc độ cao nhất là lốp nào",
        "lốp nào speed cao nhất",
        "lốp nào chạy max speed",
        "lốp nào có tốc độ cao nhất DPLUS",
        "lốp nhanh nhất trong các lốp",
        "lốp nào vận tốc tối đa cao",
        "lốp nào chạy speed cao",
        "lốp nào đạt tốc độ cao",
        "top lốp tốc độ cao nhất",
        "lốp nào max speed nhất",
        "lốp nào tốc độ khủng nhất",
        "lốp nào nhanh nhất DRC",
        "lốp nào vận tốc max",
        "lốp có tốc độ tối đa cao",
        "lốp nào chạy nhanh trên đường",
        "lốp nào đạt vận tốc cao nhất",
        "lốp nào speed tối đa",
        "lốp nhanh nhất thương hiệu DPLUS",
        "lốp nào chạy 150 km trở lên",
        "lốp nào có vận tốc tối đa",
        "max speed các lốp xe máy",
        "lốp nào mạnh tốc độ",
        "lốp nào chạy nhanh an toàn",
        "lốp nào có thể chạy nhanh nhất",
        "lốp nào max vận tốc",
        "lốp nào siêu tốc",
        "lốp nào speed nhất thị trường",
    ],
    # ── MAX_PRICE (giá cao nhất) ~30 câu ───────────────────────────
    "MAX_PRICE": [
        "lốp nào đắt nhất",
        "lốp giá cao nhất",
        "lốp nào mắc nhất",
        "giá lốp đắt nhất là bao nhiêu",
        "lốp nào giá đắt nhất",
        "lốp nào có giá cao nhất DPLUS",
        "lốp nào giá cao nhất hiện nay",
        "lốp mắc nhất của DRC",
        "lốp nào price cao nhất",
        "lốp nào có giá bán đắt",
        "lốp nào đắt nhất DPLUS",
        "lốp giá cao nhất thị trường",
        "lốp nào có giá lớn nhất",
        "lốp đắt nhất trong cửa hàng",
        "lốp nào có giá đắt",
        "lốp nào tiền cao nhất",
        "lốp nào max price",
        "lốp nào bán giá cao",
        "lốp nào đắt nhât",
        "lốp có giá bán cao nhất",
        "lốp nào giá tiền cao nhất",
        "lốp đắt nhất của DPLUS",
        "lốp nào có giá đắt nhất thị trường",
        "lốp nào mắc nhất DPLUS",
        "lốp nào full option đắt",
        "lốp nào giá cao",
        "lốp đắt nhất DRC",
        "lốp nào đắt đỏ nhất",
        "lốp nào trên 500k",
        "lốp nào giá chát nhất",
    ],
    # ── DRAINAGE (thoát nước / đi mưa) ~30 câu ─────────────────────
    "DRAINAGE": [
        "lốp nào thoát nước tốt",
        "lốp đi mưa tốt",
        "lốp chống trượt nước",
        "lốp phù hợp đường ướt",
        "lốp nào đi mưa êm",
        "lốp chống trơn trượt",
        "lốp nào bám đường khi ướt",
        "lốp nào thoát nước tốt nhất",
        "lốp đi đường ướt an toàn",
        "lốp nào chống trượt tốt",
        "lốp nào bám đường ướt tốt",
        "lốp nào đi mưa không sợ",
        "lốp nào rãnh thoát nước tốt",
        "lốp đường ướt êm",
        "lốp nào có rãnh sâu thoát nước",
        "bánh nào thoát nước tốt DPLUS",
        "lốp đi mưa DRC thế nào",
        "lốp nào không trượt nước",
        "lốp nào có khe rãnh lớn",
        "lốp chạy mưa phùn",
        "lốp nào chạy đường trơn tốt",
        "lốp nào chống nước",
        "lốp nào rãnh thoát nước sâu",
        "lốp nào bám dính khi ướt",
        "lốp nào ít trơn nước",
        "lốp nào mưa bão tốt",
        "lốp nào an toàn dưới mưa",
        "lốp nào đi mưa DPLUS êm",
        "lốp nào hệ thống thoát nước",
        "lốp nào wet traction tốt",
    ],
    # ── DURABILITY (độ bền) ~30 câu ────────────────────────────────
    "DURABILITY": [
        "lốp nào bền nhất",
        "lốp độ bền cao",
        "lốp đi được nhiều km",
        "lốp nào dùng lâu nhất",
        "lốp nào chạy bền",
        "lốp nào lâu mòn",
        "độ bền lốp DPLUS thế nào",
        "lốp bền nhất hiện nay",
        "lốp nào chạy được ít nhất km",
        "lốp nào tuổi thọ cao",
        "lốp nào đi lâu mòn nhất",
        "lốp bền lâu DPLUS",
        "độ bền lốp DRC",
        "lốp nào chất lượng bền",
        "lốp nào số km cao nhất",
        "lốp nào ít hư hỏng",
        "lốp nào đi bền nhất DRC",
        "lốp nào lâu phải thay",
        "lốp có độ bền cao nhất",
        "lốp nào đi được nhiều nhất",
        "lốp nào chạy road nhiều",
        "lốp nào tuổi thọ lâu",
        "lốp bền lâu DRC",
        "lốp nào dai nhất",
        "lốp nào lâu xuống cấp",
        "lốp nào đi xa bền",
        "lốp nào ít mòn gai",
        "lốp nào chạy đường dài bền",
        "lốp nào ultra durable",
        "độ bền lốp DPLUS đi được bao nhiêu km",
    ],
    # ── TUBE (có săm / không săm) ~30 câu ──────────────────────────
    "TUBE": [
        "lốp có săm không",
        "lốp không săm",
        "lốp tubeless",
        "lốp cần săm",
        "lốp nào là tubeless",
        "lốp DPLUS có săm không",
        "lốp nào không cần săm",
        "phân biệt lốp săm và không săm",
        "lốp có săm là gì",
        "lốp tubeless DPLUS",
        "lốp nào dùng săm",
        "lốp DRC có săm không",
        "lốp không săm tubeless",
        "lốp nào không cần ruột",
        "lốp nào cần bơm ruột",
        "lốp không săm có tốt không",
        "lốp nào không săm DPLUS",
        "lốp nào loại tubeless",
        "lốp xe máy có săm",
        "lốp không săm là gì",
        "lốp nào có ruột",
        "lốp tubeless DRC",
        "lốp nào không cần xăm",
        "phân biệt lốp tubeless và có săm",
        "lốp có săm hay không tốt hơn",
        "lốp nào tubeless DPLUS",
        "lốp nào có săm DRC",
        "lốp không săm chạy có an toàn không",
        "lốp tubeless có ưu điểm gì",
        "lốp nào tubeless cho xe tay ga",
    ],
    # ── SERVICE (dịch vụ) ~30 câu ──────────────────────────────────
    "SERVICE": [
        "đặt lịch thay lốp",
        "dịch vụ thay lốp",
        "phí lắp đặt bao nhiêu",
        "lắp lốp tận nơi",
        "thay lốp tại nhà",
        "đặt lịch hẹn thay lốp",
        "bảo hành lốp thế nào",
        "chính sách bảo hành lốp",
        "có giao lốp tận nơi không",
        "địa chỉ cửa hàng thay lốp",
        "dịch vụ lắp lốp miễn phí",
        "thay lốp xe máy tại nhà",
        "bảo hành lốp bao lâu",
        "đặt mua lốp online",
        "có ship lốp không",
        "lắp lốp tận nơi DPLUS",
        "giao lốp tận nhà",
        "phí vận chuyển lốp",
        "cửa hàng lốp xe gần đây",
        "địa chỉ mua lốp uy tín",
        "đặt lịch thay lốp tại nhà",
        "bảo hành lốp DPLUS",
        "có đổi trả lốp không",
        "chính sách đổi trả lốp",
        "mua lốp online giao nhanh",
        "thay lốp lưu động",
        "dịch vụ bảo dưỡng lốp",
        "đặt lốp qua điện thoại",
        "lắp lốp các quận",
        "tổng đài đặt lịch thay lốp",
    ],
}

# ─── Question Bank Manager ──────────────────────────────────────────────

class QuestionBank:
    """Manage question bank: build embeddings, save/load cache.

    Usage:
        bank = QuestionBank()
        bank.build()  # Build embeddings from QUESTION_BANK
        result = bank.match(user_query_vector)
        # → {"intent": "SPEED", "confidence": 0.85, "question": "..."}
    """

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "..", "Embeding_vector"
        )
        self._questions: list[str] = []
        self._intents: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._built = False

    # ── Public API ──────────────────────────────────────────────────────

    def build(self, embedder=None) -> None:
        """Build or load question embeddings.

        Args:
            embedder: QueryEmbedder instance. If None, lazy-import.
        """
        if self._built:
            return

        cache_path_npy = os.path.join(self.cache_dir, "question_embeddings.npy")
        cache_path_pkl = os.path.join(self.cache_dir, "question_bank.pkl")

        # Try loading from cache
        if os.path.exists(cache_path_npy) and os.path.exists(cache_path_pkl):
            try:
                self._embeddings = np.load(cache_path_npy)
                with open(cache_path_pkl, "rb") as f:
                    data = pickle.load(f)
                self._questions = data["questions"]
                self._intents = data["intents"]
                self._built = True
                logger.info(
                    f"[QuestionBank] Loaded {len(self._questions)} questions, "
                    f"{len(set(self._intents))} intents from cache"
                )
                return
            except Exception as e:
                logger.warning(f"[QuestionBank] Cache load failed: {e}")

        # Build from QUESTION_BANK
        self._questions = []
        self._intents = []
        for intent, qs in QUESTION_BANK.items():
            for q in qs:
                self._questions.append(q)
                self._intents.append(intent)

        if not self._questions:
            logger.error("[QuestionBank] No questions defined!")
            return

        # Với BM25: không cần embed batch, chỉ cần identity matrix
        # BM25 scores từ embed(query) đã là similarity trực tiếp
        n = len(self._questions)
        self._embeddings = np.eye(n, dtype=np.float32)
        logger.info("[QuestionBank] BM25 mode — identity matrix (%d, %d)", n, n)

        # Persist cache
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            np.save(cache_path_npy, self._embeddings)
            with open(cache_path_pkl, "wb") as f:
                pickle.dump({
                    "questions": self._questions,
                    "intents": self._intents,
                }, f)
            logger.info(
                f"[QuestionBank] Built and cached {len(self._questions)} questions, "
                f"{len(set(self._intents))} intents"
            )
        except Exception as e:
            logger.warning(f"[QuestionBank] Cache persist failed: {e}")

        self._built = True

    def match(self, query_vector: np.ndarray, threshold: float = 0.40) -> dict | None:
        """Match query vector against question bank.

        Args:
            query_vector: Normalized query embedding (384,).
            threshold: Minimum cosine similarity threshold.

        Returns:
            Dict with keys: intent, confidence, question
            None if no match above threshold.
        """
        if self._embeddings is None or query_vector is None:
            return None

        # Cosine similarity = dot product (normalized vectors)
        scores = np.dot(self._embeddings, query_vector)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < threshold:
            return None

        return {
            "intent": self._intents[best_idx],
            "confidence": round(best_score, 4),
            "question": self._questions[best_idx],
        }

    def match_grouped(self, query_vector: np.ndarray, threshold: float = 0.30) -> dict | None:
        """Match with intent-level scoring (average across all questions per intent).

        Args:
            query_vector: Normalized query embedding (384,).
            threshold: Minimum average confidence threshold.

        Returns:
            Dict with keys: intent, confidence, top_question
        """
        if self._embeddings is None or query_vector is None:
            return None

        scores = np.dot(self._embeddings, query_vector)

        # Group by intent
        intent_scores: dict[str, list[tuple[float, str]]] = {}
        for i, intent in enumerate(self._intents):
            if intent not in intent_scores:
                intent_scores[intent] = []
            intent_scores[intent].append((float(scores[i]), self._questions[i]))

        # Average per intent
        intent_avg = {
            intent: (sum(s for s, _ in pairs) / len(pairs), pairs)
            for intent, pairs in intent_scores.items()
        }

        best_intent = max(intent_avg, key=lambda k: intent_avg[k][0])
        best_score, best_pairs = intent_avg[best_intent]

        if best_score < threshold:
            return None

        # Get top question within that intent
        best_question = max(best_pairs, key=lambda p: p[0])[1]

        return {
            "intent": best_intent,
            "confidence": round(best_score, 4),
            "question": best_question,
        }

    def get_all_questions(self) -> list[str]:
        return self._questions.copy()

    def get_all_intents(self) -> list[str]:
        return self._intents.copy()

    def is_healthy(self) -> bool:
        return self._built and self._embeddings is not None
