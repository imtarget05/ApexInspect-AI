# SOP-SMT-002: Xử Lý Lỗi Thiếu Linh Kiện (Missing Component / Missing Hole)

**Mã tài liệu**: SOP-SMT-002  
**Phiên bản**: 2.1  
**Khu vực áp dụng**: Trạm gắp đặt linh kiện tự động (Pick & Place Machine)  
**Mức độ rủi ro**: TRUNG BÌNH (MEDIUM)

---

## 1. Định nghĩa lỗi
Tình trạng một hoặc nhiều vị trí linh kiện trên bo mạch PCB bị bỏ trống hoàn toàn hoặc thiếu lỗ khoan linh kiện xuyên lỗ, dẫn đến mạch hở (open circuit) và mất chức năng khối mạch.

## 2. Các nguyên nhân gốc rễ (Root Causes)
1. **Lỗi đầu hút chân không (Vacuum Nozzle)**:
   - Đầu kim hút bị bám bụi bẩn, tắc nghẽn làm giảm áp suất hút.
   - Đầu kim hút bị cong vênh hoặc mòn đầu cao su.
2. **Hết cuộn nạp liệu (Feeder Exhaustion / Jam)**:
   - Băng chuyền cuộn linh kiện (Tape Feeder) bị kẹt màng bóng, không trượt nạp linh kiện mới.
   - Hết linh kiện trên cuộn Feeder mà cảm biến báo hết chưa kịp phát hiện.
3. **Lỗi camera định vị linh kiện (Vision Alignment Error)**:
   - Ống kính camera dưới đáy máy Pick & Place bị mờ do dính bụi kem hàn, không nhận diện được góc xoay linh kiện.

## 3. Quy trình xử lý tiêu chuẩn (Corrective Actions)
- **Hành động tức thời**:
  - Ghi nhận mã vị trí linh kiện bị thiếu (ví dụ: `R102`, `C204`, `U12`).
  - Kiểm tra Feeder tương ứng xem có bị kẹt băng linh kiện hay không.
- **Nếu lỗi lặp lại $\ge 3$ lần trên cùng vị trí**:
  - Tạm dừng trạm Pick & Place để vệ sinh đầu hút chân không bằng khí nén sạch.
  - Chạy chu trình tự căn chỉnh camera quang học (Auto-calibration routine).
  - Điều hướng các bo mạch thiếu linh kiện sang khu vực bổ sung linh kiện thủ công trước khi vào lò hàn.
