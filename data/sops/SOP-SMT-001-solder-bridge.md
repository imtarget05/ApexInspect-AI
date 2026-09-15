# SOP-SMT-001: Xử Lý Lỗi Hàn Chập (Solder Bridge / Short Circuit) Trên Dây Chuyền SMT

**Mã tài liệu**: SOP-SMT-001  
**Phiên bản**: 3.2  
**Khu vực áp dụng**: Trạm hàn hồi lưu (Reflow Oven) & Máy in kem hàn (Solder Paste Printer)  
**Mức độ rủi ro**: CAO (CRITICAL)

---

## 1. Định nghĩa lỗi
Hiện tượng hai hoặc nhiều chân linh kiện (IC, điện trở, tụ điện) bị dính liền bởi thiếc hàn dư thừa, tạo thành một mạch ngắn dẫn điện không mong muốn, có nguy cơ làm cháy hỏng toàn bộ bo mạch khi cấp nguồn.

## 2. Các nguyên nhân gốc rễ (Root Causes)
1. **Lỗi máy in kem hàn (Stencil Printing)**:
   - Tấm Stencil bị bám dính thiếc hàn ở mặt dưới do lâu chưa lau tự động.
   - Áp lực dao gạt (Squeegee pressure) quá lớn làm tràn kem hàn ra ngoài pad.
2. **Lỗi hồ nhiệt độ lò hàn hồi lưu (Reflow Profile)**:
   - Nhiệt độ gia nhiệt sơ bộ (Pre-heat) tăng quá nhanh làm nổ dung môi trong kem hàn (flux splattering).
   - Nhiệt độ đỉnh (Peak temperature) vượt ngưỡng quy định (> 260°C).
3. **Lỗi thiết bị gắp đặt (Pick & Place)**:
   - Áp lực cắm linh kiện quá sâu đẩy kem hàn phòi sang hai bên.

## 3. Quy trình xử lý tiêu chuẩn (Corrective Actions)
- **Nếu xuất hiện 1 sản phẩm lỗi đơn lẻ**:
  - Gắn cờ vi phạm, điều hướng bo mạch sang trạm Sửa chữa thủ công (Manual Rework Station).
  - Dùng dây hút thiếc (Desoldering braid) và dung dịch trợ hàn để tách cầu chì thiếc.
- **Nếu xuất hiện $\ge 3$ sản phẩm liên tiếp hoặc tỷ lệ lỗi $> 10\%$ trong 15 phút**:
  - **Bắt buộc kích hoạt lệnh tạm dừng dây chuyền (HALT LINE)**.
  - Kỹ sư bảo trì lập tức thực hiện quy trình lau tấm Stencil bằng cồn Isopropyl (IPA).
  - Kiểm tra độ nhớt của kem hàn và hiệu chuẩn lại hồ nhiệt độ lò hàn.
  - Tuyệt đối không cho phép dây chuyền tái khởi động khi chưa có chữ ký xác nhận của Quản đốc ca.
