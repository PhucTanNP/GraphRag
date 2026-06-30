"""Few-shot examples cho Text2Cypher.

Mỗi example gồm {question, cypher}.
Tách riêng để dễ thêm/sửa example mà không sợ ảnh hưởng pipeline.
"""

EXAMPLES = [
    # ── Giá cả ──
    {
        "question": "Lốp 120/70-17 giá bao nhiêu?",
        "cypher": "MATCH (t:Tire) WHERE t.tire_size = '120/70-17' RETURN t.tire_size, t.brand, t.pattern_code, t.sale_price_inc_vat ORDER BY t.brand",
    },
    {
        "question": "Lốp DRC giá dưới 500k",
        "cypher": "MATCH (t:Tire) WHERE t.brand = 'DRC' AND t.sale_price_inc_vat < 500000 RETURN t.tire_size, t.brand, t.pattern_code, t.sale_price_inc_vat ORDER BY t.sale_price_inc_vat",
    },
    {
        "question": "Lốp DPLUS 130/70-17 giá rẻ nhất",
        "cypher": "MATCH (t:Tire) WHERE t.tire_size = '130/70-17' AND t.brand = 'DPLUS' RETURN t.tire_size, t.brand, t.pattern_code, t.sale_price_inc_vat ORDER BY t.sale_price_inc_vat LIMIT 1",
    },
    # ── Thông số kỹ thuật ──
    {
        "question": "Lốp 120/70-17 tốc độ tối đa bao nhiêu?",
        "cypher": "MATCH (t:Tire) WHERE t.tire_size = '120/70-17' RETURN t.tire_size, t.brand, t.pattern_code, t.max_speed_kmh, t.max_load_kg",
    },
    {
        "question": "Lốp 90/90-14 chịu tải bao nhiêu kg?",
        "cypher": "MATCH (t:Tire) WHERE t.tire_size = '90/90-14' RETURN t.tire_size, t.brand, t.pattern_code, t.max_load_kg, t.vehicle_type",
    },
    # ── So sánh ──
    {
        "question": "So sánh lốp 120/70-17 và 130/70-17",
        "cypher": "MATCH (t:Tire) WHERE t.tire_size IN ['120/70-17', '130/70-17'] RETURN t.tire_size, t.brand, t.pattern_code, t.sale_price_inc_vat, t.max_speed_kmh, t.overall_width_mm ORDER BY t.tire_size",
    },
    # ── Theo thương hiệu ──
    {
        "question": "Lốp DPLUS có những size nào?",
        "cypher": "MATCH (t:Tire) WHERE t.brand = 'DPLUS' RETURN DISTINCT t.tire_size ORDER BY t.tire_size",
    },
    {
        "question": "Có bao nhiêu lốp DRC?",
        "cypher": "MATCH (t:Tire) WHERE t.brand = 'DRC' RETURN count(t) AS so_luong",
    },
    # ── Hoa lốp ──
    {
        "question": "Lốp nào có hoa D119?",
        "cypher": "MATCH (t:Tire)-[:CÓ_HOA]->(p:TirePattern {code: 'D119'}) RETURN t.tire_size, t.brand, p.code, p.loai",
    },
    {
        "question": "Lốp DPLUS có những kiểu hoa nào?",
        "cypher": "MATCH (t:Tire {brand: 'DPLUS'})-[:CÓ_HOA]->(p:TirePattern) RETURN DISTINCT p.code, p.loai ORDER BY p.code",
    },
    # ── Lợi ích hoa lốp ──
    {
        "question": "Hoa D119 có ưu điểm gì?",
        "cypher": "MATCH (p:TirePattern {code: 'D119'}) RETURN p.code, p.loai, p.loi_ich, p.phu_hop, p.dieu_kien_duong",
    },
    # ── Tra theo xe ──
    {
        "question": "Lốp cho xe SH Mode 125",
        "cypher": "MATCH (v:Vehicle {name: 'SH Mode 125'})-[r:DÙNG_LỐP_TRƯỚC|DÙNG_LỐP_SAU]->(t:Tire) RETURN v.name, t.tire_size, t.brand, t.pattern_code, t.sale_price_inc_vat, type(r) AS vi_tri ORDER BY vi_tri",
    },
    {
        "question": "Xe Vision 2022 dùng lốp gì?",
        "cypher": "MATCH (v:Vehicle) WHERE v.name CONTAINS 'Vision' RETURN v.name, v.front_tire, v.rear_tire, v.tire_type",
    },
    {
        "question": "Lốp cho xe Exciter",
        "cypher": "MATCH (v:Vehicle) WHERE toLower(v.name) CONTAINS 'exciter' RETURN v.name, v.front_tire, v.rear_tire, v.motorcycle_type",
    },
    # ── Săm ──
    {
        "question": "Săm 3.00-10 giá bao nhiêu?",
        "cypher": "MATCH (tb:Tube) WHERE tb.tube_size = '3.00-10' RETURN tb.tube_size, tb.sale_price_inc_vat",
    },
    {
        "question": "Săm dùng cho lốp 90/90-14?",
        "cypher": "MATCH (tb:Tube)-[:DÙNG_CHO]->(t:Tire) WHERE t.tire_size = '90/90-14' RETURN tb.tube_size, tb.sale_price_inc_vat, t.tire_size, t.brand",
    },
    # ── Thương hiệu xe ──
    {
        "question": "Honda có những xe nào?",
        "cypher": "MATCH (b:VehicleBrand {name: 'Honda'})<-[:THUỘC]-(v:Vehicle) RETURN b.name, v.name, v.front_tire, v.rear_tire ORDER BY v.name",
    },
    {
        "question": "Xe Piaggio có những dòng nào?",
        "cypher": "MATCH (b:VehicleBrand {name: 'Piaggio'})<-[:THUỘC]-(v:Vehicle) RETURN v.name, v.front_tire, v.rear_tire ORDER BY v.name",
    },
    # ── Loại xe ──
    {
        "question": "Lốp cho xe tay ga",
        "cypher": "MATCH (v:Vehicle {motorcycle_type: 'Scooter'})-[:DÙNG_LỐP_TRƯỚC|DÙNG_LỐP_SAU]->(t:Tire) RETURN DISTINCT t.tire_size, t.brand, t.pattern_code, t.sale_price_inc_vat ORDER BY t.tire_size",
    },
    {
        "question": "Lốp cho xe số",
        "cypher": "MATCH (v:Vehicle {motorcycle_type: 'Manual'})-[:DÙNG_LỐP_TRƯỚC|DÙNG_LỐP_SAU]->(t:Tire) RETURN DISTINCT t.tire_size, t.brand, t.pattern_code ORDER BY t.tire_size",
    },
    # ── Kết hợp ──
    {
        "question": "Xe SH Mode 125 chạy lốp D121 giá bao nhiêu?",
        "cypher": "MATCH (v:Vehicle {name: 'SH Mode 125'})-[:DÙNG_LỐP_TRƯỚC|DÙNG_LỐP_SAU]->(t:Tire)-[:CÓ_HOA]->(p:TirePattern {code: 'D121'}) RETURN v.name, t.tire_size, t.brand, t.pattern_code, p.loai, t.sale_price_inc_vat",
    },
]
