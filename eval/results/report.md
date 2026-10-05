# Kết quả đánh giá chatbot RAG

_Chạy lúc 05/10/2026 00:15 bằng `python eval/run_eval.py`._

- **Dữ liệu:** 3 file PDF tiếng Việt (8 trang) của một công ty hư cấu, nên LLM không thể biết trước nội dung.
- **Bộ câu hỏi:** 36 câu có đáp án trong tài liệu + 6 câu **không có** đáp án (để kiểm tra chatbot có bịa hay không). Câu hỏi được diễn đạt khác với văn bản gốc.
- **Cấu hình mặc định của ứng dụng (⭐):** chia đoạn theo cấu trúc + dòng ngữ cảnh, 800 ký tự, embedding `intfloat/multilingual-e5-small`, tìm kiếm hybrid, top-4, LLM `gpt-oss:20b`.

## 1. Truy xuất (retrieval)

Một câu hỏi được tính là **tìm đúng** nếu trong top-k đoạn trả về có đoạn chứa nguyên văn câu bằng chứng (evidence) của câu hỏi đó. **Hit@k** = tỉ lệ câu hỏi tìm đúng trong top-k; **MRR** = trung bình của 1/(thứ hạng của đoạn đúng đầu tiên), càng gần 1 càng tốt.

### 1.1. Cách tìm kiếm và model embedding (chia đoạn theo cấu trúc + ngữ cảnh, 800 ký tự)

| Cách tìm kiếm | Model embedding | Hit@1 | Hit@3 | Hit@5 | MRR | Thời gian / truy vấn |
|---|---|---|---|---|---|---|
| BM25 (từ khoá) | — | 94% | 100% | 100% | 0.968 | 0.4 ms |
| Vector (ngữ nghĩa) | MiniLM-L12 | 69% | 86% | 94% | 0.783 | 6.2 ms |
| Vector (ngữ nghĩa) | e5-small | 100% | 100% | 100% | 1.000 | 6.8 ms |
| Hybrid (BM25 + vector) | MiniLM-L12 | 83% | 94% | 97% | 0.894 | 7.1 ms |
| Hybrid (BM25 + vector) ⭐ | e5-small | 100% | 100% | 100% | 1.000 | 7.9 ms |

### 1.2. Cách chia đoạn (embedding e5-small)

| Cách chia đoạn | Kích thước | Số đoạn | Bằng chứng còn nguyên | Vector Hit@1 | Vector MRR | Hybrid Hit@1 | Hybrid MRR |
|---|---|---|---|---|---|---|---|
| Cắt cố định theo số ký tự | 400 | 52 | 100% | 89% | 0.940 | 89% | 0.944 |
| Cắt cố định theo số ký tự | 800 | 29 | 100% | 92% | 0.958 | 92% | 0.958 |
| Theo cấu trúc (Chương/Điều) | 400 | 69 | 100% | 97% | 0.972 | 97% | 0.979 |
| Theo cấu trúc (Chương/Điều) | 800 | 55 | 100% | 97% | 0.986 | 97% | 0.986 |
| Theo cấu trúc + dòng ngữ cảnh | 400 | 69 | 100% | 97% | 0.972 | 97% | 0.981 |
| Theo cấu trúc + dòng ngữ cảnh ⭐ | 800 | 55 | 100% | 100% | 1.000 | 100% | 1.000 |

_"Bằng chứng còn nguyên" = tỉ lệ câu hỏi mà câu bằng chứng nằm trọn trong ít nhất một đoạn (không bị cắt đôi)._

## 2. Chất lượng câu trả lời

Giám khảo là LLM `gemma4:31b` (khác họ với gpt-oss để giảm thiên vị "tự chấm bài mình"). Đáp án chuẩn của mỗi câu gồm **ý chính** (bắt buộc) và có thể có **chi tiết bổ sung** như điều kiện, ngoại lệ, thời hạn. Giám khảo chấm ý chính theo 3 mức **Đúng / Một phần / Sai**, và **Đầy đủ** nếu câu trả lời nêu được chi tiết bổ sung. **Bịa đặt** = câu trả lời sai nhưng không thừa nhận là không biết.

| Hệ thống | Đúng (36 câu) | Một phần | Sai | Đầy đủ chi tiết (19 câu) | Từ chối đúng (6 câu không có đáp án) | Số câu bịa đặt | Có trích dẫn | Thời gian LLM TB | Token prompt TB |
|---|---|---|---|---|---|---|---|---|---|
| LLM không RAG | 6% | 11% | 83% | 5% | 67% | 15/42 | 0% | 1.3 s | 155 |
| RAG + gpt-oss:20b, prompt v1 | 100% | 0% | 0% | 16% | 100% | 0/42 | 100% | 1.5 s | 849 |
| RAG + gpt-oss:20b, prompt v2 ⭐ | 100% | 0% | 0% | 42% | 100% | 0/42 | 100% | 1.3 s | 894 |
| RAG + gpt-oss:120b, prompt v2 | 100% | 0% | 0% | 42% | 100% | 0/42 | 100% | 0.9 s | 894 |

### 2.1. Cải tiến prompt sau khi phân tích lỗi

Phân tích lỗi lần đầu cho thấy prompt v1 trả lời đúng ý chính nhưng hay bỏ sót điều kiện quan trọng, ví dụ trả lời "tối đa 2 ngày làm từ xa" mà không nói "chỉ áp dụng sau thử việc". Prompt v2 (`rag/pipeline.py`) thêm quy tắc: *trả lời thẳng vào câu hỏi trước, sau đó nêu ngắn gọn các điều kiện, ngoại lệ hoặc thời hạn quan trọng liên quan trực tiếp tới câu hỏi*.

- Đầy đủ chi tiết: 16% → **42%**
- Đúng ý chính: 100% → **100%**
- Từ chối đúng câu không có đáp án: 100% → **100%**
- Thời gian LLM trung bình: 1.5 s → 1.3 s

### 2.2. Phân tích lỗi của cấu hình mặc định (RAG + gpt-oss:20b, prompt v2)

| Câu | Câu hỏi | Câu trả lời của chatbot | Ý chính | Đầy đủ | Đoạn đúng ở hạng | Nhận xét của giám khảo |
|---|---|---|---|---|---|---|
| q02 | Năm nay tôi chưa dùng hết phép thì có được chuyển sang năm sau không? | Bạn có thể chuyển tối đa 5 ngày phép chưa dùng hết trong năm sang năm sau và phải sử dụng trước ngày 31/3. [1] | Đúng | ✗ | 1 | Trả lời đúng ý chính nhưng thiếu chi tiết về việc không quy đổi ngày phép thành tiền. |
| q03 | Một tuần tôi được làm việc ở nhà tối đa mấy ngày? | Bạn được làm việc ở nhà tối đa **2 ngày mỗi tuần**. [1] | Đúng | ✗ | 1 | Câu trả lời đúng ý chính nhưng thiếu các chi tiết bổ sung về thời điểm áp dụng và cách đăng ký. |
| q05 | Lương được chuyển khoản vào ngày nào trong tháng? | Lương được chuyển khoản vào ngày 5 hằng tháng. [1] | Đúng | ✗ | 1 | Câu trả lời đúng ý chính nhưng thiếu chi tiết bổ sung về ngày nghỉ. |
| q10 | Kỹ sư phần mềm mới vào phải thử việc bao lâu? | Kỹ sư phần mềm mới vào phải thử việc **60 ngày**.   Điều 21. Thử việc quy định thời gian thử việc là 60 ngày đối với vị trí kỹ sư và chuyên viên kỹ thuật. [1] | Đúng | ✗ | 1 | Trả lời đúng ý chính nhưng thiếu chi tiết bổ sung về lương thử việc. |
| q11 | Thực tập sinh được trả phụ cấp bao nhiêu mỗi tháng? | Thực tập sinh được trả phụ cấp 4.000.000 đồng mỗi tháng. [1] | Đúng | ✗ | 1 | Câu trả lời nêu đúng số tiền phụ cấp nhưng thiếu chi tiết áp dụng cho thực tập sinh toàn thời gian. |
| q13 | Quên chấm công thì phải làm gì? | Quên chấm công, nhân viên phải gửi giải trình trên LT People trong vòng 2 ngày làm việc để quản lý trực tiếp xác nhận. [1] | Đúng | ✗ | 1 | Câu trả lời đúng ý chính nhưng thiếu các chi tiết bổ sung về số lần quên tối đa và hình thức xử lý. |
| q18 | Phụ cấp công tác trong nước mỗi ngày là bao nhiêu? | Phụ cấp công tác trong nước mỗi ngày là **300.000 đồng**. [1] | Đúng | ✗ | 1 | Câu trả lời nêu đúng ý chính nhưng bỏ sót chi tiết bổ sung về cách tính 50% cho ngày khởi hành/trở về. |
| q25 | Mời khách hàng đi ăn trong chuyến công tác thì được chi tối đa bao nhiêu? | Chi phí tiếp khách trong chuyến công tác được thanh toán tối đa 1.000.000 đồng/người/bữa. [1] | Đúng | ✗ | 1 | Câu trả lời nêu đúng ý chính nhưng thiếu các chi tiết bổ sung về yêu cầu khi quyết toán. |
| q27 | Mật khẩu phải dài tối thiểu bao nhiêu ký tự? | Mật khẩu phải có độ dài tối thiểu **12 ký tự**.   Điều 3. Mật khẩu [1] | Đúng | ✗ | 1 | Trả lời đúng ý chính nhưng thiếu các chi tiết bổ sung về thành phần mật khẩu. |
| q28 | Bao lâu thì phải đổi mật khẩu một lần? | Bạn phải đổi mật khẩu mỗi 90 ngày.   Điều 3. Mật khẩu [1] | Đúng | ✗ | 1 | Trả lời đúng ý chính nhưng thiếu chi tiết bổ sung về việc không trùng 5 mật khẩu gần nhất. |
| q31 | Nếu làm mất laptop công ty thì phải báo cho ai và trong bao lâu? | Bạn phải báo cáo ngay cho bộ phận IT (hotline 9999 hoặc email security@lamtinh.example) **trong vòng 1 giờ** kể từ khi phát hiện mất laptop.   Điều 10 của quy định an toàn thông tin quy định thời hạn và kênh báo cáo. [1] | Đúng | ✗ | 1 | Câu trả lời chính xác đầy đủ các ý chính nhưng thiếu chi tiết bổ sung về việc khóa thiết bị và xóa dữ liệu từ xa. |

### 2.3. Ví dụ: cùng một câu hỏi, có và không có RAG

**Năm nay tôi chưa dùng hết phép thì có được chuyển sang năm sau không?**

- Không RAG: _Không, phép nghỉ chưa dùng hết trong năm không được chuyển sang năm sau._ → **Sai**
- RAG: _Bạn có thể chuyển tối đa 5 ngày phép chưa dùng hết trong năm sang năm sau và phải sử dụng trước ngày 31/3. [1]_ → **Đúng**
- Đáp án chuẩn: Được, tối đa 5 ngày phép được chuyển sang năm sau.

**Lương được chuyển khoản vào ngày nào trong tháng?**

- Không RAG: _Lương được chuyển khoản vào ngày 25 của tháng._ → **Sai**
- RAG: _Lương được chuyển khoản vào ngày 5 hằng tháng. [1]_ → **Đúng**
- Đáp án chuẩn: Ngày 5 hằng tháng.

**Công ty hỗ trợ bao nhiêu tiền cho việc học mỗi năm?**

- Không RAG: _Công ty hỗ trợ 1.200.000 đ mỗi năm cho chi phí học tập của nhân viên._ → **Sai**
- RAG: _Công ty hỗ trợ 8 000 000 đồng cho mỗi nhân viên chính thức mỗi năm để chi cho khóa học, sách hoặc chứng chỉ liên quan đến công việc. Điều kiện: nhân viên phải gửi đề xuất qua biểu mẫu “Đề xuất học tập” trên LT People và được Trưởng bộ phận phê duyệt trước khi đăng ký khóa học. Ngoại lệ: đối với chứng chỉ có chi phí trên 5 000 000 đồng, nhân viên cam kết tiếp tục làm việc tại công ty ít nhất 12 tháng sau khi hoàn thành; nếu nghỉ việc trước thời hạn, nhân viên phải hoàn trả chi phí theo tỷ lệ thời gian còn lại. [1]_ → **Đúng**
- Đáp án chuẩn: 8.000.000 đồng/năm.

