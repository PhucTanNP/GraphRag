"""Schema Neo4j — dùng cho prompt Text2Cypher.

Tách riêng để dễ kiểm soát và cập nhật khi cấu trúc graph thay đổi.
"""

SCHEMA = """Node labels:
  - Tire: tire_size, brand, pattern_code, sale_price_inc_vat, has_tube, vehicle_type, max_load_kg, max_speed_kmh, outer_diameter_mm, overall_width_mm
  - TirePattern: code, loai, loi_ich, phu_hop, dieu_kien_duong, nguon
  - Tube: tube_size, sale_price_inc_vat
  - Vehicle: name, front_tire, rear_tire, tire_type, motorcycle_type
  - VehicleBrand: name

Relationships:
  - (Tire)-[:CÓ_HOA]->(TirePattern)        -- lốp có kiểu hoa
  - (Tube)-[:DÙNG_CHO]->(Tire)              -- săm dùng cho lốp
  - (Vehicle)-[:THUỘC]->(VehicleBrand)       -- xe thuộc thương hiệu
  - (Vehicle)-[:DÙNG_LỐP_TRƯỚC]->(Tire)     -- xe dùng lốp trước
  - (Vehicle)-[:DÙNG_LỐP_SAU]->(Tire)       -- xe dùng lốp sau

Description:
  - Tire: lốp xe máy (DRC + DPLUS), có giá bán (sale_price_inc_vat), thông số kỹ thuật
  - TirePattern: kiểu hoa/gai lốp, có lợi ích (loi_ich), phù hợp (phu_hop), điều kiện đường (dieu_kien_duong)
  - Tube: săm xe, dùng cho lốp có săm
  - Vehicle: dòng xe cụ thể (Vision, SH Mode, Exciter...), có lốp trước (front_tire) và lốp sau (rear_tire)
  - VehicleBrand: thương hiệu xe (Honda, Yamaha, Piaggio, SYM...)
  - motorcycle_type: Scooter (tay ga) hoặc Manual (xe số)
  - tire_type: Tubeless hoặc Tube (loại lốp không săm / có săm)
"""
