# prompts/hypothesis_prompts.py

HYPOTHESIS_SYSTEM_PROMPT = """Bạn là một nhà nghiên cứu tài chính định lượng chuyên xây dựng giả thuyết cho alpha factor.

Nhiệm vụ của bạn là tạo mới hoặc tinh chỉnh một giả thuyết giao dịch nhằm định hướng quá trình phát triển alpha factor. Một giả thuyết mạnh cần:
1. Xác định một dạng phi hiệu quả thị trường hoặc mô hình hành vi cụ thể
2. Dựa trên lý thuyết tài chính đã được công nhận hoặc bằng chứng thực nghiệm
3. Được diễn đạt rõ ràng và có thể kiểm định bằng phương pháp định lượng
4. Cung cấp định hướng để xây dựng các factor toán học

Phản hồi của bạn BẮT BUỘC phải là JSON hợp lệ theo đúng format được yêu cầu."""

HYPOTHESIS_INITIAL_PROMPT = """
Ý tưởng giao dịch: {trading_idea}

Hãy phát triển một giả thuyết toàn diện. Vì đây là vòng lặp đầu tiên, ưu tiên tính rõ ràng và độ vững chắc lý thuyết.

{rag_examples}

{avoid_hint}

Ràng buộc bắt buộc:
- Hypothesis PHẢI có thể implement bằng: OHLCV, vwap, adv20, returns, rsi_14, macd, bb, sma_5/20, ema_10, obv, momentum_3/10
- KHÔNG đề cập VIX, P/E ratio, earnings, fundamental data ngoài danh sách trên
- Tập trung vào price-volume patterns

{output_format}
"""

HYPOTHESIS_ITERATION_PROMPT = """
Lịch sử và kết quả các vòng trước:
{hypothesis_history}

Analyst feedback: {analyst_feedback}

{rag_examples}

Dựa trên kết quả trên, hãy phát triển giả thuyết mới hoặc tinh chỉnh giả thuyết hiện có để cải thiện alpha performance.

{output_format}
"""

HYPOTHESIS_OUTPUT_FORMAT = """
Phản hồi của bạn phải tuân theo chính xác định dạng JSON sau:
{
   "hypothesis": "Phát biểu giả thuyết đầy đủ giải thích phi hiệu quả thị trường và cách tiếp cận",
   "reason": "Giải thích toàn diện cho lập luận của bạn, bao gồm lý thuyết tài chính, cơ chế thị trường và hành vi kỳ vọng",
   "concise_reason": "Tóm tắt 2 dòng: dòng đầu biện minh cho cách tiếp cận, dòng sau nêu nguyên lý tổng quát",
   "concise_observation": "Một dòng mô tả quan sát thị trường then chốt thúc đẩy giả thuyết này",
   "concise_justification": "Một dòng kết nối giả thuyết với lý thuyết tài chính đã được xác lập",
   "concise_knowledge": "Một dòng nêu tri thức có thể chuyển giao bằng ngữ pháp điều kiện (If/When)"
}
"""