# SOP-SMT-006: Bảo Trì Đầu Hút Chân Không & Xử Lý Lỗi Mất Linh Kiện Máy Gắp Đặt SMD

**Mã tài liệu**: SOP-SMT-006  
**Tiêu chuẩn tham chiếu**: IPC-9850 (Surface Mount Placement Equipment Characterization)  
**Khu vực áp dụng**: Máy gắp đặt linh kiện tốc độ cao (Modular SMD Placement System)  
**Mức độ rủi ro**: TRUNG BÌNH - CAO (MEDIUM - CRITICAL)

---

## 1. Định nghĩa lỗi
Hiện tượng linh kiện dán bề mặt (SMD chip, IC QFP/BGA) bị thiếu trên bo mạch (Missing Component), bị đặt lệch góc (Skew/Misalignment), hoặc dựng đứng như bia mộ (Tombstoning) sau khi qua trạm gắp đặt.

## 2. Nguyên nhân gốc rễ
1. **Tắc nghẽn đầu hút chân không (Vacuum Nozzle Clogging)**:
   - Bụi kem hàn hoặc cặn bẩn bám dính vào lỗ hút đường kính siêu nhỏ ($0.3\text{ mm}$ đối với linh kiện 0201/01005).
   - Áp suất chân không đo được sụt giảm dưới $-75\text{ kPa}$.
2. **Hao mòn cơ học đầu hút (Nozzle Tip Wear)**:
   - Miếng đệm cao su hoặc đầu gốm bị mẻ, rách gây hở khí khi di chuyển ở gia tốc cao ($> 3G$).
3. **Camera căn chỉnh quang học (Optical Vision Alignment)**:
   - Đèn chiếu sáng LED hoặc lăng kính quang học của máy bị bám bụi bẩn, dẫn đến nhận diện sai tọa độ góc quay $\theta$.

## 3. Quy trình bảo dưỡng & Xử lý sự cố
1. **Hành động phản ứng nhanh**:
   - Khi hệ thống camera AOI báo lỗi mất linh kiện liên tiếp $\ge 3$ sản phẩm:
   - Tự động kích hoạt cơ cấu phân luồng sang khay sửa chữa (Rework Station).
   - Kiểm tra số hiệu đầu hút (Nozzle ID) và đầu gắn feeder tương ứng trong nhật ký máy.
2. **Quy trình vệ sinh đầu hút**:
   - Tháo cassette chứa đầu hút, đặt vào bể làm sạch sóng siêu âm (Ultrasonic cleaner) chứa dung dịch cồn IPA trong $15\text{ phút}$.
   - Thổi khô bằng khí nén sạch áp suất cao ($0.5\text{ MPa}$).
3. **Hiệu chuẩn lại quang học (Vision Calibration)**:
   - Thực hiện quy trình cân chỉnh chuẩn tâm bằng đĩa kính định chuẩn (Glass Calibration Plate).
   - Chạy thử 5 bo mạch kiểm tra trước khi hoàn tất lệnh phê duyệt MES.
