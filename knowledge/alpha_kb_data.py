"""
alpha_kb_data.py
Danh sách alphas từ Kakushadze (2016) "101 Formulaic Alphas".
Chỉ giữ những alpha dùng fields: open/high/low/close/volume/vwap/adv20/returns
và operators đã implement. Loại bỏ các alpha cần: cap, IndClass level chi tiết
(subindustry, industry) khi không có data, adv với d khác 20.

Mapping operator names:
  Ts_Rank         → ts_rank
  Ts_ArgMax       → ts_argmax
  Ts_ArgMin       → ts_argmin
  SignedPower      → signed_power
  decay_linear    → ts_decayed_linear (hoặc decay_linear alias)
  rank(x)         → rank(x)          (cross-sectional expanding rank)
  delay(x,d)      → shift(x,d)  hoặc delay(x,d) (alias đã có)
  delta(x,d)      → ts_delta(x,d) hoặc delta(x,d) (alias)
  correlation     → ts_corr hoặc correlation (alias)
  covariance      → ts_cov hoặc covariance (alias)
  stddev          → ts_std hoặc stddev (alias)
  sum(x,d)        → ts_sum(x,d) hoặc sum_op(x,d)
  scale(x)        → scale(x)
  sign(x)         → sign(x)
  abs(x)          → abso(x)
  log(x)          → log(x)
  indneutralize(x, industry) → indneutralize(x, df['industry'])
"""

ALPHA_KB: list = [
    {
        "id": "alpha002",
        "expression": "alpha = neg(ts_corr(rank(ts_delta(log(df['volume']), 2)), rank(div(minus(df['close'], df['open']), df['open'])), 6))",
        "description": "Tương quan âm giữa hạng thay đổi khối lượng giao dịch và hạng tỷ suất sinh lời giá trong 6 ngày. Phản ánh sự phân kỳ giữa động lực khối lượng và giá.",
    },
    {
        "id": "alpha003",
        "expression": "alpha = neg(ts_corr(rank(df['open']), rank(df['volume']), 10))",
        "description": "Tương quan âm giữa hạng giá mở cửa và hạng khối lượng giao dịch trong 10 ngày.",
    },
    {
        "id": "alpha004",
        "expression": "alpha = neg(ts_rank(rank(df['low']), 9))",
        "description": "Hạng chuỗi thời gian âm của hạng chéo mặt cắt giá thấp nhất trong 9 ngày.",
    },
    {
        "id": "alpha006",
        "expression": "alpha = neg(ts_corr(df['open'], df['volume'], 10))",
        "description": "Tương quan âm giữa giá mở cửa và khối lượng giao dịch trong 10 ngày.",
    },
    {
        "id": "alpha007",
        "expression": "alpha = ts_zscore_scale(neg(ts_rank(abso(ts_delta(df['close'], 7)), 60)), 20)",
        "description": "Bán khống các cổ phiếu có biến động giá lớn gần đây khi khối lượng vượt adv20. Khai thác hiện tượng hồi quy sau các giai đoạn biến động mạnh.",
    },
    {
        "id": "alpha008",
        "expression": "alpha = neg(rank(minus(cwise_mul(ts_sum(df['open'], 5), ts_sum(df['returns'], 5)), delay(cwise_mul(ts_sum(df['open'], 5), ts_sum(df['returns'], 5)), 10))))",
        "description": "Hạng âm của sự thay đổi tích số (tổng giá mở cửa × tổng tỷ suất sinh lời) so với 10 ngày trước.",
    },
    {
        "id": "alpha011",
        "expression": "alpha = cwise_mul(add(rank(ts_max(minus(df['vwap'], df['close']), 3)), rank(ts_min(minus(df['vwap'], df['close']), 3))), rank(ts_delta(df['volume'], 3)))",
        "description": "Kết hợp biên độ cực trị của chênh lệch VWAP-giá đóng cửa với thay đổi khối lượng. Nắm bắt các mẫu giá bị dẫn dắt bởi thanh khoản.",
    },
    {
        "id": "alpha012",
        "expression": "alpha = cwise_mul(sign(ts_delta(df['volume'], 1)), neg(ts_delta(df['close'], 1)))",
        "description": "Dấu của thay đổi khối lượng nhân với biến động giá âm. Chiến lược hồi quy: khối lượng tăng kèm giá giảm.",
    },
    {
        "id": "alpha013",
        "expression": "alpha = neg(rank(ts_cov(rank(df['close']), rank(df['volume']), 5)))",
        "description": "Hạng âm của hiệp phương sai giữa hạng giá đóng cửa và hạng khối lượng. Cổ phiếu có tương quan giá-khối lượng cao có xu hướng hồi quy.",
    },
    {
        "id": "alpha014",
        "expression": "alpha = cwise_mul(neg(rank(ts_delta(df['returns'], 3))), ts_corr(df['open'], df['volume'], 10))",
        "description": "Kết hợp thay đổi động lực tỷ suất sinh lời với tương quan giữa giá mở cửa và khối lượng giao dịch.",
    },
    {
        "id": "alpha015",
        "expression": "alpha = neg(ts_sum(rank(ts_corr(rank(df['high']), rank(df['volume']), 3)), 3))",
        "description": "Tổng hạng âm của tương quan giá cao nhất-khối lượng. Phản xu hướng trong các giai đoạn giá cao kèm khối lượng lớn.",
    },
    {
        "id": "alpha016",
        "expression": "alpha = neg(rank(ts_cov(rank(df['high']), rank(df['volume']), 5)))",
        "description": "Hạng âm của hiệp phương sai giữa hạng giá cao nhất và hạng khối lượng trong 5 ngày.",
    },
    {
        "id": "alpha018",
        "expression": "alpha = neg(rank(add(add(ts_std(abso(minus(df['close'], df['open'])), 5), minus(df['close'], df['open'])), ts_corr(df['close'], df['open'], 10))))",
        "description": "Kết hợp độ biến động biên độ giá trong phiên, biến động giá và tương quan đóng cửa-mở cửa.",
    },
    {
        "id": "alpha022",
        "expression": "alpha = neg(cwise_mul(ts_delta(ts_corr(df['high'], df['volume'], 5), 5), rank(ts_std(df['close'], 20))))",
        "description": "Thay đổi tương quan giá cao nhất-khối lượng nhân với độ biến động giá đóng cửa. Phát hiện sự chuyển dịch chế độ thị trường.",
    },
    {
        "id": "alpha023",
        "expression": "alpha = ts_zscore_scale(cwise_mul(greater(df['high'], div(ts_sum(df['high'], 20), 20.0)), neg(ts_delta(df['high'], 2))), 20)",
        "description": "Khi giá cao nhất vượt đường trung bình 20 ngày, bán khống đà tăng giá cao nhất gần đây.",
    },
    {
        "id": "alpha025",
        "expression": "alpha = rank(cwise_mul(cwise_mul(cwise_mul(neg(df['returns']), df['adv20']), df['vwap']), minus(df['high'], df['close'])))",
        "description": "Hạng của tích số: tỷ suất sinh lời âm × adv20 × VWAP × (giá cao nhất - giá đóng cửa). Kết hợp đa nhân tố.",
    },
    {
        "id": "alpha026",
        "expression": "alpha = neg(ts_max(ts_corr(ts_rank(df['volume'], 5), ts_rank(df['high'], 5), 5), 3))",
        "description": "Tương quan cực đại âm giữa hạng khối lượng và hạng giá cao nhất trong 3 ngày.",
    },
    {
        "id": "alpha028",
        "expression": "alpha = scale(add(ts_corr(df['adv20'], df['low'], 5), div(add(df['high'], df['low']), 2.0)))",
        "description": "Chuẩn hóa tổng của tương quan adv20-giá thấp nhất và giá trung điểm cao-thấp.",
    },
    {
        "id": "alpha030",
        "expression": "alpha = cwise_mul(div(minus(1.0, rank(add(add(sign(minus(df['close'], delay(df['close'], 1))), sign(minus(delay(df['close'], 1), delay(df['close'], 2)))), sign(minus(delay(df['close'], 2), delay(df['close'], 3)))))), ts_sum(df['volume'], 5)), div(1.0, ts_sum(df['volume'], 20)))",
        "description": "Phản xu hướng dựa trên chuỗi tín hiệu chiều giá liên tiếp, điều chỉnh theo tỷ lệ khối lượng ngắn hạn và trung hạn.",
    },
    {
        "id": "alpha033",
        "expression": "alpha = rank(neg(div(df['open'], df['close'])))",
        "description": "Hạng của tỷ lệ giá mở cửa/đóng cửa âm. Cổ phiếu mở cửa cao so với đóng cửa có xu hướng hồi quy.",
    },
    {
        "id": "alpha034",
        "expression": "alpha = rank(add(minus(1.0, rank(div(ts_std(df['returns'], 2), ts_std(df['returns'], 5)))), minus(1.0, rank(ts_delta(df['close'], 1)))))",
        "description": "Kết hợp hạng tỷ lệ biến động ngắn hạn/trung hạn và hạng biến động giá gần nhất.",
    },
    {
        "id": "alpha035",
        "expression": "alpha = cwise_mul(cwise_mul(ts_rank(df['volume'], 32), minus(1.0, ts_rank(add(df['close'], df['high']), 16))), minus(1.0, ts_rank(df['returns'], 32)))",
        "description": "Hạng khối lượng × nghịch đảo hạng (giá đóng cửa + giá cao nhất) × nghịch đảo hạng tỷ suất sinh lời. Kết hợp động lực khối lượng và hồi quy giá.",
    },
    {
        "id": "alpha037",
        "expression": "alpha = add(rank(ts_corr(delay(minus(df['open'], df['close']), 1), df['close'], 200)), rank(minus(df['open'], df['close'])))",
        "description": "Tương quan biên độ giá trong phiên với giá đóng cửa trong 200 ngày cộng biên độ hiện tại.",
    },
    {
        "id": "alpha038",
        "expression": "alpha = cwise_mul(neg(ts_rank(df['close'], 10)), rank(div(df['close'], df['open'])))",
        "description": "Hạng chuỗi thời gian âm của giá đóng cửa nhân với hạng tỷ lệ đóng cửa/mở cửa.",
    },
    {
        "id": "alpha040",
        "expression": "alpha = cwise_mul(neg(rank(ts_std(df['high'], 10))), ts_corr(df['high'], df['volume'], 10))",
        "description": "Hạng âm của biến động giá cao nhất nhân với tương quan giá cao nhất-khối lượng.",
    },
    {
        "id": "alpha041",
        "expression": "alpha = minus(signed_power(cwise_mul(df['high'], df['low']), 0.5), df['vwap'])",
        "description": "Trung bình nhân của giá cao nhất và thấp nhất trừ VWAP. Đo lường độ lệch tâm giá trong phiên so với giá bình quân gia quyền theo khối lượng.",
    },
    {
        "id": "alpha042",
        "expression": "alpha = div(rank(minus(df['vwap'], df['close'])), rank(add(df['vwap'], df['close'])))",
        "description": "Tỷ lệ hạng chênh lệch VWAP-đóng cửa trên hạng tổng VWAP-đóng cửa. Tín hiệu hồi quy tức thì.",
    },
    {
        "id": "alpha043",
        "expression": "alpha = cwise_mul(ts_rank(div(df['volume'], df['adv20']), 20), ts_rank(neg(ts_delta(df['close'], 7)), 8))",
        "description": "Hạng tỷ lệ khối lượng/adv20 nhân với hạng nghịch đảo động lực giá. Phản xu hướng khi khối lượng cao.",
    },
    {
        "id": "alpha044",
        "expression": "alpha = neg(ts_corr(df['high'], rank(df['volume']), 5))",
        "description": "Tương quan âm giữa giá cao nhất và hạng khối lượng trong 5 ngày.",
    },
    {
        "id": "alpha046",
        "expression": "alpha = ts_zscore_scale(ts_delta(df['close'], 1), 20)",
        "description": "Tín hiệu động lực/hồi quy dựa trên gia tốc xu hướng trung hạn. Đơn giản hóa từ logic điều kiện ba nhánh trong bản gốc.",
    },
    {
        "id": "alpha050",
        "expression": "alpha = neg(ts_max(rank(ts_corr(rank(df['volume']), rank(df['vwap']), 5)), 5))",
        "description": "Giá trị cực đại âm của hạng tương quan khối lượng-VWAP trong 5 ngày.",
    },
    {
        "id": "alpha052",
        "expression": "alpha = cwise_mul(cwise_mul(add(neg(ts_min(df['low'], 5)), delay(ts_min(df['low'], 5), 5)), rank(div(minus(ts_sum(df['returns'], 240), ts_sum(df['returns'], 20)), 220.0))), ts_rank(df['volume'], 5))",
        "description": "Bật tăng từ đáy giá thấp nhất kết hợp với tỷ suất sinh lời dài hạn trừ ngắn hạn, điều chỉnh theo khối lượng giao dịch.",
    },
    {
        "id": "alpha053",
        "expression": "alpha = neg(ts_delta(div(minus(minus(df['close'], df['low']), minus(df['high'], df['close'])), add(minus(df['close'], df['low']), 1e-9)), 9))",
        "description": "Thay đổi vị trí tương đối của giá đóng cửa trong biên độ cao-thấp trong 9 ngày.",
    },
    {
        "id": "alpha054",
        "expression": "alpha = neg(div(cwise_mul(minus(df['low'], df['close']), pow_op(df['open'], 5)), cwise_mul(minus(df['low'], df['high']), add(pow_op(df['close'], 5), 1e-9))))",
        "description": "Alpha dựa trên cấu trúc giá trong phiên từ mối quan hệ giữa giá mở cửa, đóng cửa, cao nhất và thấp nhất.",
    },
    {
        "id": "alpha055",
        "expression": "alpha = neg(ts_corr(rank(div(minus(df['close'], ts_min(df['low'], 12)), add(minus(ts_max(df['high'], 12), ts_min(df['low'], 12)), 1e-9))), rank(df['volume']), 6))",
        "description": "Tương quan âm giữa hạng chỉ báo Stochastic và hạng khối lượng giao dịch trong 6 ngày.",
    },
    {
        "id": "alpha060",
        "expression": "alpha = minus(scale(rank(cwise_mul(div(minus(minus(df['close'], df['low']), minus(df['high'], df['close'])), add(minus(df['high'], df['low']), 1e-9)), df['volume']))), scale(rank(ts_argmax(df['close'], 10))))",
        "description": "Áp lực dòng tiền chuẩn hóa trừ hạng đỉnh giá gần nhất được chuẩn hóa. Kết hợp khối lượng và động lực giá.",
    },
    {
        "id": "alpha083",
        "expression": "alpha = div(cwise_mul(rank(delay(div(minus(df['high'], df['low']), add(div(ts_sum(df['close'], 5), 5.0), 1e-9)), 2)), rank(rank(df['volume']))), div(div(minus(df['high'], df['low']), add(div(ts_sum(df['close'], 5), 5.0), 1e-9)), add(minus(df['vwap'], df['close']), 1e-9)))",
        "description": "Tỷ lệ hạng biên độ cao-thấp trễ hai kỳ trên biên độ hiện tại được chuẩn hóa theo chênh lệch VWAP-đóng cửa.",
    },
    {
        "id": "alpha101",
        "expression": "alpha = div(minus(df['close'], df['open']), add(minus(df['high'], df['low']), 0.001))",
        "description": "Tỷ suất sinh lời trong phiên chuẩn hóa theo biên độ giá trong ngày. Giá trị dương khi giá đóng cửa cao hơn mở cửa so với mức biến động.",
    },
]