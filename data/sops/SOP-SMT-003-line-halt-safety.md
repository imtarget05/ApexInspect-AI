# SOP-SMT-003: Quy Trình An Toàn Dừng Khẩn Cấp & Khởi Động Lại Dây Chuyền (Line Halt Protocol)

**Mã tài liệu**: SOP-SMT-003  
**Phiên bản**: 4.0  
**Khu vực áp dụng**: Toàn bộ dây chuyền sản xuất SMT  
**Mức độ rủi ro**: RẤT CAO (MANDATORY COMPLIANCE)

---

## 1. Mục đích & Nguyên tắc an toàn
Đảm bảo khi phát hiện chuỗi lỗi nghiêm trọng lặp đi lặp lại hoặc sự cố thiết bị cơ khí, quyết định dừng dây chuyền phải được thực thi an toàn, không làm cháy bo mạch đang nằm trong lò hàn và có sự phê duyệt có trách nhiệm của Quản đốc ca (Human-In-The-Loop).

## 2. Tiêu chí kích hoạt dừng dây chuyền tự động từ AI Agent
Hệ thống AI Agent được ủy quyền phát lệnh đề xuất dừng chuyền khi:
1. Phát hiện $\ge 3$ sản phẩm liên tiếp cùng dính lỗi nghiêm trọng (Hàn chập, vỡ linh kiện).
2. Tỷ lệ lỗi (Defect Rate) trung bình trong 10 phút vượt ngưỡng $15\%$.
3. Nhiệt độ trạm hoặc cảm biến áp suất báo vượt ngưỡng nguy hiểm.

## 3. Trình tự thực thi lệnh dừng chuyền (Step-by-Step)
1. **Bước 1 (Đề xuất có kiểm soát)**:
   - AI Agent lập ticket MES với mức độ `CRITICAL`.
   - Gửi yêu cầu phê duyệt đến màn hình điều hành của Quản đốc ca.
2. **Bước 2 (Xác nhận của con người - HITL)**:
   - Quản đốc ca kiểm tra hình ảnh lỗi trên màn hình.
   - Bấm `[PHÊ DUYỆT DỪNG CHUYỀN]` (Approve).
3. **Bước 3 (Thực thi dừng chuyền theo quy trình an toàn)**:
   - Máy in kem hàn và máy Pick & Place dừng nạp phôi mới ngay lập tức.
   - Băng tải lò hàn hồi lưu **KHÔNG ĐƯỢC DỪNG ĐỘT NGỘT** mà phải tiếp tục chạy chậm để đưa toàn bộ bo mạch đang ở trong buồng nhiệt ra ngoài an toàn (tránh cháy nổ bo mạch).
   - Hệ thống MES chuyển trạng thái dây chuyền thành `HALTED`.
4. **Bước 4 (Khởi động lại dây chuyền)**:
   - Chỉ được khởi động lại sau khi kỹ sư bảo trì khắc phục xong lỗi và Quản đốc ca ký biên bản nghiệm thu trực tiếp trên hệ thống MES.
