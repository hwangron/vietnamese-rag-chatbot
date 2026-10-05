"""Tạo 3 file PDF mẫu (tiếng Việt) trong data/docs để demo và đánh giá chatbot.

Nội dung thuộc về một công ty HƯ CẤU ("Lam Tinh"): mọi tên, con số và email đều không có thật.
Dùng tài liệu hư cấu có chủ đích: LLM chắc chắn chưa từng thấy nội dung này khi huấn luyện,
nên nếu chatbot trả lời đúng thì đó là nhờ truy xuất tài liệu (RAG), không phải nhờ "học thuộc".

Chạy:  python scripts/make_sample_docs.py
"""

import io
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.config import DOCS_DIR  # noqa: E402

COMPANY = "Công ty Cổ phần Công nghệ Lam Tinh"
NOTE = ("Đây là tài liệu mẫu của một công ty hư cấu, được tạo ra để demo dự án chatbot RAG. "
        "Mọi tên, số liệu và địa chỉ email trong tài liệu đều không có thật.")

# Font hỗ trợ đầy đủ tiếng Việt: (thư mục, thường, đậm, nghiêng). Không tìm thấy thì dùng font dự phòng của PyMuPDF.
FONT_CANDIDATES = [
    ("C:/Windows/Fonts", "arial.ttf", "arialbd.ttf", "ariali.ttf"),
    ("/usr/share/fonts/truetype/dejavu", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf"),
    ("/System/Library/Fonts/Supplemental", "Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"),
]

BASE_CSS = """
* {font-family: vn, sans-serif; font-size: 10.5pt; line-height: 1.4;}
h1 {font-size: 20pt; color: #1f3b63; margin-bottom: 2pt;}
h2 {font-size: 13.5pt; color: #1f3b63; margin-top: 14pt; border-bottom: 1px solid #c9d3e3;}
h3 {font-size: 11pt; margin-top: 9pt; margin-bottom: 2pt;}
p {margin-top: 2pt; margin-bottom: 4pt; text-align: justify;}
li {margin-bottom: 2pt;}
.meta {color: #555; font-size: 9.5pt;}
.note {font-style: italic; color: #8a4b00; background-color: #fff4e0; padding: 4pt;}
table {border-collapse: collapse; margin: 4pt 0;}
th, td {border: 1px solid #8796ad; padding: 3pt 6pt; font-size: 9.5pt;}
th {background-color: #e8eef7;}
"""


def header(title: str, code: str, effective: str) -> str:
    return (f"<h1>{title}</h1>"
            f'<p class="meta">{COMPANY} · Mã tài liệu: {code} · Hiệu lực từ {effective}</p>'
            f'<p class="note">{NOTE}</p>')


SO_TAY = header("Sổ tay nhân viên", "HR-01, phiên bản 3.2", "01/01/2026") + """
<h2>Chương 1. Quy định chung</h2>
<h3>Điều 1. Phạm vi áp dụng</h3>
<p>Sổ tay này áp dụng cho toàn bộ nhân viên chính thức, nhân viên thử việc và thực tập sinh làm việc tại các văn phòng của Lam Tinh ở Hà Nội, Đà Nẵng và TP. Hồ Chí Minh. Những nội dung sổ tay chưa quy định được thực hiện theo Bộ luật Lao động và các văn bản pháp luật hiện hành.</p>
<p>Bộ phận Nhân sự (HR) chịu trách nhiệm giải thích và cập nhật sổ tay. Mọi thắc mắc vui lòng gửi về hr@lamtinh.example hoặc gọi máy lẻ 1010.</p>
<h3>Điều 2. Giá trị cốt lõi</h3>
<p>Lam Tinh hoạt động dựa trên ba giá trị cốt lõi: Tận tâm với khách hàng, Minh bạch trong công việc và Học hỏi không ngừng. Các giá trị này là cơ sở để đánh giá hiệu suất và xét khen thưởng hằng năm.</p>

<h2>Chương 2. Thời gian làm việc</h2>
<h3>Điều 3. Giờ làm việc</h3>
<p>Thời gian làm việc tiêu chuẩn là từ Thứ Hai đến Thứ Sáu, từ 8:30 đến 17:30, nghỉ trưa từ 12:00 đến 13:00. Nhân viên được áp dụng giờ vào linh hoạt trong khung từ 7:30 đến 9:30, với điều kiện làm đủ 8 tiếng mỗi ngày. Thứ Bảy và Chủ Nhật là ngày nghỉ hằng tuần, trừ khi dự án có kế hoạch làm việc được Trưởng bộ phận phê duyệt.</p>
<h3>Điều 4. Chấm công</h3>
<p>Nhân viên chấm công vào và ra bằng nhận diện khuôn mặt trên ứng dụng LT People. Trường hợp quên chấm công, nhân viên phải gửi giải trình trên LT People trong vòng 2 ngày làm việc để quản lý trực tiếp xác nhận.</p>
<p>Mỗi tháng được quên chấm công tối đa 3 lần. Từ lần thứ 4 trở đi, mỗi lần quên chấm công được tính là nghỉ nửa ngày không lương.</p>
<h3>Điều 5. Làm việc từ xa</h3>
<p>Sau khi kết thúc thời gian thử việc, nhân viên được làm việc từ xa tối đa 2 ngày mỗi tuần. Lịch làm việc từ xa phải được đăng ký trên LT People trước 17:00 của ngày làm việc liền trước và được quản lý trực tiếp đồng ý. Nhân viên thử việc và thực tập sinh không áp dụng chế độ làm việc từ xa.</p>
<p>Trong thời gian làm việc từ xa, nhân viên phải luôn trực tuyến trên ứng dụng nhắn tin nội bộ LT Chat và phản hồi tin nhắn công việc trong vòng 30 phút. Nhân viên làm việc từ xa ít nhất 4 ngày trong tháng được hỗ trợ chi phí Internet 200.000 đồng/tháng.</p>
<h3>Điều 6. Làm thêm giờ</h3>
<p>Làm thêm giờ phải được Trưởng bộ phận phê duyệt trước trên LT People. Tiền lương làm thêm giờ được tính như sau:</p>
<ul>
<li>Ngày làm việc bình thường: 150% tiền lương giờ.</li>
<li>Ngày nghỉ hằng tuần (Thứ Bảy, Chủ Nhật): 200% tiền lương giờ.</li>
<li>Ngày nghỉ lễ, Tết: 300% tiền lương giờ.</li>
</ul>
<p>Tổng số giờ làm thêm không vượt quá 40 giờ mỗi tháng.</p>

<h2>Chương 3. Nghỉ phép và nghỉ việc riêng</h2>
<h3>Điều 7. Nghỉ phép năm</h3>
<p>Nhân viên chính thức được hưởng 14 ngày phép năm có hưởng lương. Cứ đủ 3 năm làm việc tại công ty, số ngày phép năm được cộng thêm 1 ngày. Nhân viên làm việc chưa đủ 12 tháng được tính số ngày phép theo tỷ lệ số tháng làm việc thực tế.</p>
<p>Nhân viên đăng ký nghỉ phép trên LT People trước ít nhất 3 ngày làm việc. Nếu nghỉ từ 3 ngày liên tiếp trở lên, nhân viên phải đăng ký trước tối thiểu 2 tuần để bộ phận sắp xếp công việc.</p>
<p>Tối đa 5 ngày phép chưa dùng hết trong năm được chuyển sang năm sau và phải sử dụng trước ngày 31/3. Số ngày phép còn lại không được quy đổi thành tiền, trừ trường hợp nhân viên nghỉ việc.</p>
<h3>Điều 8. Nghỉ ốm</h3>
<p>Ngoài chế độ ốm đau của Bảo hiểm xã hội, công ty chi trả nguyên lương cho tối đa 6 ngày nghỉ ốm mỗi năm. Nếu nghỉ ốm từ 2 ngày liên tiếp trở lên, nhân viên cần nộp giấy xác nhận của cơ sở y tế cho bộ phận Nhân sự.</p>
<h3>Điều 9. Nghỉ việc riêng có hưởng lương</h3>
<ul>
<li>Bản thân kết hôn: nghỉ 3 ngày.</li>
<li>Con đẻ, con nuôi kết hôn: nghỉ 1 ngày.</li>
<li>Bố, mẹ, vợ hoặc chồng, con qua đời: nghỉ 3 ngày.</li>
<li>Nhân viên nam có vợ sinh con: nghỉ thêm 5 ngày có lương ngoài chế độ thai sản của Bảo hiểm xã hội.</li>
</ul>
<p>Nhân viên sinh con được công ty tặng quà mừng 3.000.000 đồng.</p>
<h3>Điều 10. Ngày nghỉ lễ và ngày thành lập công ty</h3>
<p>Nhân viên được nghỉ các ngày lễ, Tết theo quy định của pháp luật. Ngoài ra, ngày thành lập công ty 12/9 hằng năm là ngày nghỉ có hưởng lương của toàn công ty.</p>

<h2>Chương 4. Lương và thưởng</h2>
<h3>Điều 11. Kỳ trả lương</h3>
<p>Lương được trả mỗi tháng một lần vào ngày 5 hằng tháng bằng hình thức chuyển khoản. Nếu ngày trả lương trùng ngày nghỉ, lương được trả vào ngày làm việc liền trước. Phiếu lương được gửi qua email của nhân viên. Mọi thắc mắc về lương cần gửi bộ phận Nhân sự trong vòng 7 ngày kể từ ngày nhận phiếu lương.</p>
<h3>Điều 12. Thưởng tháng 13 và thưởng hiệu suất</h3>
<p>Thưởng tháng 13 bằng 1 tháng lương cơ bản, được chi trả cùng kỳ lương tháng 1 năm sau. Nhân viên làm việc chưa đủ 12 tháng trong năm được tính theo tỷ lệ, với điều kiện đã làm việc tối thiểu 3 tháng.</p>
<p>Thưởng hiệu suất được xét hai lần mỗi năm vào tháng 6 và tháng 12, dựa trên kết quả đánh giá KPI:</p>
<ul>
<li>Xếp loại A (xuất sắc): thưởng 2 tháng lương.</li>
<li>Xếp loại B (tốt): thưởng 1 tháng lương.</li>
<li>Xếp loại C (đạt): thưởng 0,5 tháng lương.</li>
<li>Xếp loại D (chưa đạt): không có thưởng hiệu suất.</li>
</ul>
<h3>Điều 13. Xét tăng lương</h3>
<p>Công ty xét tăng lương mỗi năm một lần vào tháng 4, căn cứ vào kết quả đánh giá KPI cuối năm trước và mặt bằng lương thị trường.</p>
<h3>Điều 14. Thưởng giới thiệu ứng viên</h3>
<p>Nhân viên giới thiệu ứng viên được tuyển dụng thành công sẽ nhận thưởng 5.000.000 đồng đối với vị trí thông thường và 10.000.000 đồng đối với vị trí cấp cao (từ Senior trở lên). Khoản thưởng được chi trả sau khi ứng viên hoàn thành thời gian thử việc.</p>

<h2>Chương 5. Phúc lợi</h2>
<h3>Điều 15. Bảo hiểm sức khỏe</h3>
<p>Nhân viên chính thức được tham gia gói bảo hiểm sức khỏe LT Care với quyền lợi điều trị nội trú tối đa 200.000.000 đồng/năm. Nhân viên có thể mua thêm bảo hiểm cho người thân với giá ưu đãi; công ty hỗ trợ 50% phí bảo hiểm cho con dưới 18 tuổi.</p>
<h3>Điều 16. Khám sức khỏe định kỳ</h3>
<p>Công ty tổ chức khám sức khỏe định kỳ cho toàn bộ nhân viên mỗi năm một lần vào tháng 10.</p>
<h3>Điều 17. Ăn trưa và gửi xe</h3>
<p>Nhân viên được hỗ trợ tiền ăn trưa 35.000 đồng cho mỗi ngày làm việc thực tế tại văn phòng; khoản hỗ trợ được cộng vào lương hằng tháng. Công ty chi trả phí gửi xe máy tại tòa nhà văn phòng.</p>
<h3>Điều 18. Ngân sách học tập</h3>
<p>Mỗi nhân viên chính thức có ngân sách học tập 8.000.000 đồng/năm để chi cho khóa học, sách hoặc chứng chỉ liên quan đến công việc. Nhân viên gửi đề xuất qua biểu mẫu "Đề xuất học tập" trên LT People và cần được Trưởng bộ phận phê duyệt trước khi đăng ký khóa học.</p>
<p>Đối với chứng chỉ có chi phí trên 5.000.000 đồng, nhân viên cam kết tiếp tục làm việc tại công ty ít nhất 12 tháng sau khi hoàn thành; nếu nghỉ việc trước thời hạn, nhân viên hoàn trả chi phí theo tỷ lệ thời gian còn lại.</p>
<h3>Điều 19. Hoạt động gắn kết</h3>
<p>Công ty tổ chức du lịch hằng năm cho toàn thể nhân viên, thường vào tháng 6 hoặc tháng 7. Mỗi phòng ban có quỹ hoạt động gắn kết 300.000 đồng/người/quý.</p>
<h3>Điều 20. Thiết bị làm việc</h3>
<p>Nhân viên được cấp máy tính xách tay khi nhận việc. Máy được thay mới sau 4 năm sử dụng. Nhân viên chịu trách nhiệm bảo quản thiết bị và tuân thủ Quy định an toàn thông tin của công ty.</p>

<h2>Chương 6. Thử việc, thực tập và nghỉ việc</h2>
<h3>Điều 21. Thử việc</h3>
<p>Thời gian thử việc là 60 ngày đối với vị trí kỹ sư và chuyên viên kỹ thuật, 30 ngày đối với các vị trí khác. Lương thử việc bằng 90% lương chính thức. Quản lý trực tiếp đánh giá kết quả thử việc và thông báo cho nhân viên chậm nhất 3 ngày làm việc trước khi kết thúc thời gian thử việc.</p>
<h3>Điều 22. Chương trình thực tập</h3>
<p>Chương trình thực tập kéo dài 3 tháng. Thực tập sinh toàn thời gian được hưởng phụ cấp 4.000.000 đồng/tháng và được phân công một người hướng dẫn (mentor) trong suốt quá trình thực tập. Thực tập sinh có kết quả đánh giá tốt được ưu tiên xét tuyển làm nhân viên chính thức mà không cần qua vòng phỏng vấn kỹ thuật.</p>
<h3>Điều 23. Nghỉ việc</h3>
<p>Nhân viên muốn nghỉ việc phải báo trước 30 ngày đối với hợp đồng lao động xác định thời hạn và 45 ngày đối với hợp đồng lao động không xác định thời hạn. Trước ngày nghỉ, nhân viên phải bàn giao công việc, tài liệu và thiết bị theo biên bản bàn giao.</p>
<p>Công ty thanh toán tiền lương còn lại và tiền phép năm chưa sử dụng trong vòng 14 ngày làm việc kể từ ngày chấm dứt hợp đồng. Nhân viên nghỉ việc sẽ có buổi phỏng vấn thôi việc với bộ phận Nhân sự.</p>

<h2>Chương 7. Quy tắc ứng xử</h2>
<h3>Điều 24. Trang phục</h3>
<p>Từ Thứ Hai đến Thứ Năm, nhân viên mặc trang phục lịch sự. Thứ Sáu nhân viên được mặc trang phục tự do. Khi gặp khách hàng, nhân viên mặc trang phục công sở.</p>
<h3>Điều 25. Quà tặng từ đối tác</h3>
<p>Nhân viên không được nhận quà tặng từ khách hàng hoặc đối tác có giá trị trên 500.000 đồng. Trường hợp không thể từ chối, nhân viên phải báo cáo bộ phận Pháp chế trong vòng 3 ngày làm việc.</p>
<h3>Điều 26. Kênh phản ánh</h3>
<p>Nhân viên có thể phản ánh các hành vi vi phạm qua hộp thư ẩn danh speakup@lamtinh.example. Công ty cam kết bảo mật danh tính và không trả đũa người phản ánh trung thực.</p>
"""

CONG_TAC_PHI = header("Quy định công tác phí", "TC-07", "01/03/2026") + """
<h3>Điều 1. Phạm vi áp dụng</h3>
<p>Quy định này áp dụng cho các chuyến công tác trong nước và nước ngoài đã được phê duyệt của nhân viên Lam Tinh. Bộ phận Kế toán chịu trách nhiệm hướng dẫn và kiểm soát việc thanh toán; liên hệ ketoan@lamtinh.example.</p>
<h3>Điều 2. Đề nghị và phê duyệt công tác</h3>
<p>Nhân viên tạo "Đề nghị công tác" trên hệ thống LT Expense, kèm mục đích, lịch trình và dự toán chi phí. Đề nghị phải được gửi ít nhất 5 ngày làm việc trước chuyến đi trong nước và 15 ngày làm việc trước chuyến đi nước ngoài.</p>
<p>Quản lý trực tiếp là người phê duyệt chuyến công tác. Chuyến công tác nước ngoài hoặc chuyến có tổng dự toán trên 30.000.000 đồng phải được Giám đốc Tài chính phê duyệt thêm.</p>
<h3>Điều 3. Phương tiện đi lại</h3>
<ul>
<li>Chặng đường trên 300 km: đi máy bay hạng phổ thông, đặt vé qua bộ phận Hành chính.</li>
<li>Chặng đường từ 300 km trở xuống: ưu tiên tàu hỏa, xe khách hoặc xe của công ty.</li>
<li>Vé máy bay hạng thương gia chỉ áp dụng cho cấp Giám đốc trở lên hoặc chuyến bay có thời gian bay trên 6 giờ.</li>
<li>Taxi hoặc xe công nghệ được thanh toán theo thực tế khi di chuyển giữa sân bay, khách sạn và nơi làm việc, tối đa 500.000 đồng/ngày.</li>
<li>Nhân viên tự lái xe cá nhân đi công tác được hỗ trợ 4.000 đồng/km theo quãng đường thực tế.</li>
</ul>
<h3>Điều 4. Định mức tiền phòng khách sạn</h3>
<p>Tiền phòng khách sạn được thanh toán theo thực tế nhưng không vượt quá định mức mỗi đêm dưới đây:</p>
<table>
<tr><th>Cấp bậc</th><th>Hà Nội, TP. Hồ Chí Minh</th><th>Tỉnh, thành phố khác</th><th>Nước ngoài</th></tr>
<tr><td>Nhân viên, Trưởng nhóm</td><td>1.200.000 đồng</td><td>900.000 đồng</td><td>120 USD</td></tr>
<tr><td>Trưởng phòng</td><td>1.800.000 đồng</td><td>1.300.000 đồng</td><td>180 USD</td></tr>
<tr><td>Giám đốc trở lên</td><td>2.500.000 đồng</td><td>2.000.000 đồng</td><td>250 USD</td></tr>
</table>
<p>Khi có từ hai nhân viên cùng giới đi chung một chuyến công tác, các nhân viên ở ghép phòng đôi và được áp dụng định mức bằng 1,5 lần định mức một người.</p>
<h3>Điều 5. Phụ cấp công tác</h3>
<p>Nhân viên đi công tác được hưởng phụ cấp ăn uống và chi tiêu cá nhân 300.000 đồng/ngày đối với công tác trong nước và 50 USD/ngày đối với công tác nước ngoài. Ngày khởi hành sau 12:00 hoặc ngày trở về trước 12:00 được tính 50% mức phụ cấp.</p>
<h3>Điều 6. Tạm ứng</h3>
<p>Nhân viên được tạm ứng tối đa 70% tổng dự toán đã được phê duyệt. Đề nghị tạm ứng gửi trên LT Expense trước ngày khởi hành ít nhất 3 ngày làm việc; tiền tạm ứng được chuyển khoản vào tài khoản nhận lương.</p>
<h3>Điều 7. Quyết toán chi phí</h3>
<p>Nhân viên nộp hồ sơ quyết toán trên LT Expense trong vòng 7 ngày làm việc kể từ ngày về. Hồ sơ gồm hóa đơn điện tử ghi đúng tên và mã số thuế của công ty, thẻ lên máy bay (boarding pass) và các chứng từ liên quan. Các khoản chi dưới 200.000 đồng có thể dùng biên lai thay cho hóa đơn.</p>
<p>Nếu quá 30 ngày kể từ ngày về mà nhân viên chưa quyết toán, khoản tạm ứng sẽ được khấu trừ vào lương của kỳ gần nhất.</p>
<h3>Điều 8. Các khoản không được thanh toán</h3>
<ul>
<li>Đồ uống và đồ ăn trong minibar của khách sạn.</li>
<li>Dịch vụ giặt là, trừ chuyến công tác kéo dài trên 5 ngày.</li>
<li>Đồ uống có cồn, trừ trường hợp tiếp khách đã được phê duyệt.</li>
<li>Tiền phạt vi phạm giao thông hoặc vi phạm quy định của hãng vận chuyển.</li>
<li>Chi phí cho người thân đi cùng và chi phí nâng hạng ghế, nâng hạng phòng.</li>
</ul>
<h3>Điều 9. Chi phí tiếp khách</h3>
<p>Chi phí tiếp khách trong chuyến công tác được thanh toán tối đa 1.000.000 đồng/người/bữa. Nhân viên phải ghi rõ tên khách, đơn vị và mục đích buổi tiếp khách khi quyết toán.</p>
<h3>Điều 10. Bảo hiểm du lịch</h3>
<p>Công ty mua bảo hiểm du lịch cho mọi chuyến công tác nước ngoài. Nhân viên gửi thông tin hộ chiếu cho bộ phận Hành chính trước ngày khởi hành ít nhất 7 ngày.</p>
<h3>Điều 11. Xử lý vi phạm</h3>
<p>Nhân viên kê khai chi phí không trung thực phải hoàn trả toàn bộ số tiền đã nhận và bị xem xét kỷ luật theo Nội quy lao động.</p>
"""

AN_TOAN_THONG_TIN = header("Quy định an toàn thông tin", "IT-02", "15/02/2026") + """
<h3>Điều 1. Mục đích và phạm vi</h3>
<p>Quy định này nhằm bảo vệ thông tin của công ty và khách hàng, áp dụng cho toàn bộ nhân viên, cộng tác viên và nhà thầu có quyền truy cập vào hệ thống của Lam Tinh. Bộ phận An ninh thông tin (Security) chịu trách nhiệm giám sát việc thực hiện.</p>
<h3>Điều 2. Phân loại dữ liệu</h3>
<p>Dữ liệu của công ty được phân thành 4 cấp độ:</p>
<ul>
<li>Công khai: thông tin đã được công bố, ví dụ tài liệu giới thiệu sản phẩm, tin tuyển dụng.</li>
<li>Nội bộ: thông tin dùng trong công ty, ví dụ quy trình làm việc, thông báo nội bộ.</li>
<li>Bí mật: thông tin kinh doanh nhạy cảm, ví dụ hợp đồng, báo giá, kế hoạch kinh doanh.</li>
<li>Tuyệt mật: mã nguồn sản phẩm lõi, dữ liệu cá nhân của khách hàng và thông tin lương của nhân viên.</li>
</ul>
<p>Dữ liệu Tuyệt mật chỉ được lưu trong thư mục mã hóa trên LT Drive và không được gửi ra ngoài công ty qua email.</p>
<h3>Điều 3. Mật khẩu</h3>
<p>Mật khẩu phải có độ dài tối thiểu 12 ký tự, bao gồm chữ hoa, chữ thường, chữ số và ký tự đặc biệt. Mật khẩu phải được thay đổi định kỳ 90 ngày một lần và không được trùng với 5 mật khẩu gần nhất. Tài khoản sẽ bị khóa tạm thời nếu nhập sai mật khẩu 5 lần trong vòng 15 phút.</p>
<p>Nhân viên sử dụng trình quản lý mật khẩu do công ty cung cấp và tuyệt đối không chia sẻ mật khẩu cho người khác, kể cả đồng nghiệp hoặc bộ phận IT.</p>
<h3>Điều 4. Xác thực đa yếu tố</h3>
<p>Xác thực đa yếu tố (MFA) là bắt buộc khi đăng nhập email, VPN, LT Drive và hệ thống quản lý mã nguồn. Nhân viên sử dụng ứng dụng xác thực (authenticator) trên điện thoại; công ty không sử dụng mã xác thực qua SMS.</p>
<h3>Điều 5. Sử dụng máy tính của công ty</h3>
<ul>
<li>Máy tính phải bật chế độ tự động khóa màn hình sau 5 phút không sử dụng.</li>
<li>Ổ cứng phải được mã hóa bằng BitLocker (Windows) hoặc FileVault (macOS).</li>
<li>Chỉ cài phần mềm trong danh mục được phép; phần mềm khác phải gửi yêu cầu qua IT Helpdesk.</li>
<li>Không sử dụng USB hoặc ổ cứng di động để sao chép dữ liệu, trừ khi được IT cấp phép tạm thời với thời hạn tối đa 7 ngày.</li>
</ul>
<h3>Điều 6. Thiết bị cá nhân</h3>
<p>Nhân viên chỉ được truy cập email và LT Chat trên điện thoại cá nhân sau khi cài đặt ứng dụng quản lý thiết bị LT Secure. Không được lưu dữ liệu Bí mật hoặc Tuyệt mật trên thiết bị cá nhân.</p>
<h3>Điều 7. Làm việc từ xa và mạng Internet</h3>
<p>Khi làm việc ngoài văn phòng, nhân viên bắt buộc kết nối VPN của công ty trước khi truy cập hệ thống nội bộ. Không sử dụng Wi-Fi công cộng khi chưa bật VPN và không xử lý dữ liệu Tuyệt mật tại nơi công cộng.</p>
<h3>Điều 8. Email lừa đảo (phishing)</h3>
<p>Khi nhận được email đáng ngờ, nhân viên không mở tệp đính kèm, không nhấp vào đường link và báo cáo bằng nút "Report Phishing" trên ứng dụng email hoặc chuyển tiếp email tới phishing@lamtinh.example.</p>
<p>Bộ phận Security tổ chức diễn tập phishing mỗi quý một lần. Nhân viên nhấp vào đường link trong bài diễn tập phải hoàn thành lại khóa đào tạo an toàn thông tin trong vòng 7 ngày.</p>
<h3>Điều 9. Sử dụng công cụ trí tuệ nhân tạo (AI)</h3>
<p>Nhân viên không được nhập mã nguồn, dữ liệu khách hàng hoặc bất kỳ dữ liệu Bí mật, Tuyệt mật nào vào các công cụ AI công cộng như chatbot hoặc dịch vụ dịch thuật trực tuyến. Khi cần dùng AI cho công việc, nhân viên sử dụng nền tảng AI nội bộ LT Assistant đã được bộ phận Security phê duyệt.</p>
<h3>Điều 10. Báo cáo sự cố an toàn thông tin</h3>
<p>Khi phát hiện sự cố như mất thiết bị, lộ mật khẩu hoặc nghi ngờ máy tính nhiễm mã độc, nhân viên phải báo cáo trong vòng 1 giờ qua hotline IT (máy lẻ 9999) hoặc email security@lamtinh.example. Trường hợp mất máy tính xách tay, bộ phận IT sẽ khóa thiết bị và xóa dữ liệu từ xa.</p>
<h3>Điều 11. Đào tạo</h3>
<p>Nhân viên mới phải hoàn thành khóa đào tạo nhận thức an toàn thông tin trong 2 tuần đầu tiên và tham gia đào tạo lại hằng năm. Bài kiểm tra cuối khóa cần đạt tối thiểu 80% số điểm.</p>
<h3>Điều 12. Quản lý quyền truy cập</h3>
<p>Quyền truy cập được cấp theo nguyên tắc tối thiểu cần thiết cho công việc và được rà soát 6 tháng một lần. Khi nhân viên nghỉ việc, toàn bộ tài khoản sẽ bị vô hiệu hóa trong vòng 24 giờ kể từ thời điểm chấm dứt hợp đồng.</p>
<h3>Điều 13. Xử lý vi phạm</h3>
<ul>
<li>Vi phạm lần đầu: nhắc nhở bằng văn bản.</li>
<li>Vi phạm lần thứ hai: khiển trách và không xét thưởng hiệu suất của kỳ đánh giá đó.</li>
<li>Vi phạm nghiêm trọng, ví dụ làm lộ dữ liệu Tuyệt mật: có thể bị chấm dứt hợp đồng lao động.</li>
</ul>
"""

DOCUMENTS = [
    ("so_tay_nhan_vien.pdf", "Sổ tay nhân viên", SO_TAY),
    ("quy_dinh_cong_tac_phi.pdf", "Quy định công tác phí", CONG_TAC_PHI),
    ("quy_dinh_an_toan_thong_tin.pdf", "Quy định an toàn thông tin", AN_TOAN_THONG_TIN),
]


def find_font() -> tuple[str, pymupdf.Archive | None]:
    """Trả về (CSS @font-face, kho font) cho font hỗ trợ tiếng Việt đầu tiên tìm thấy."""
    for folder, regular, bold, italic in FONT_CANDIDATES:
        if (Path(folder) / regular).exists():
            css = (f"@font-face {{font-family: vn; src: url('{regular}');}}"
                   f"@font-face {{font-family: vn; src: url('{bold}'); font-weight: bold;}}"
                   f"@font-face {{font-family: vn; src: url('{italic}'); font-style: italic;}}")
            return css, pymupdf.Archive(folder)
    print("Không tìm thấy Arial/DejaVu, dùng font dự phòng của PyMuPDF.")
    return "", None


def build_pdf(html: str, title: str, out_path: Path, font_css: str, archive) -> int:
    # 1) Dàn trang HTML thành PDF (khổ A4, lề 2 cm)
    buffer = io.BytesIO()
    story = pymupdf.Story(html=html, user_css=font_css + BASE_CSS, archive=archive)
    writer = pymupdf.DocumentWriter(buffer)
    mediabox = pymupdf.paper_rect("a4")
    content_area = mediabox + (56, 56, -56, -70)
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(content_area)
        story.draw(device)
        writer.end_page()
    writer.close()

    # 2) Thêm footer có số trang, metadata, rồi nén font cho file nhỏ gọn
    doc = pymupdf.open("pdf", buffer.getvalue())
    for number, page in enumerate(doc, start=1):
        footer = f"Lam Tinh (công ty hư cấu) · {title} · Trang {number}/{doc.page_count}"
        page.insert_htmlbox(
            pymupdf.Rect(56, 800, 539, 825),
            f'<p style="font-size: 8pt; color: #777; text-align: center;">{footer}</p>',
            css=font_css + "* {font-family: vn, sans-serif;}",
            archive=archive,
        )
    doc.set_metadata({
        "title": title,
        "author": f"{COMPANY} (công ty hư cấu)",
        "subject": "Tài liệu mẫu cho dự án chatbot RAG",
        "creator": "scripts/make_sample_docs.py",
    })
    doc.subset_fonts()
    doc.save(out_path, garbage=4, deflate=True)
    pages = doc.page_count
    doc.close()
    return pages


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    font_css, archive = find_font()
    for filename, title, html in DOCUMENTS:
        out_path = DOCS_DIR / filename
        pages = build_pdf(html, title, out_path, font_css, archive)
        print(f"✓ {out_path.relative_to(ROOT)}: {pages} trang, {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
