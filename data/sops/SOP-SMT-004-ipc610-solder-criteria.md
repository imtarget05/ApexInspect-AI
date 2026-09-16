# SOP-SMT-004: Tiêu Chuẩn Đánh Giá Mối Hàn Theo IPC-A-610 Class 3

**Mã tài liệu**: SOP-SMT-004  
**Tiêu chuẩn tham chiếu**: IPC-A-610G / J-STD-001 Class 3 (High Reliability Electronic Products)  
**Khu vực áp dụng**: Trạm kiểm tra quang học AOI & Trạm hàn SMT  
**Mức độ rủi ro**: CAO (CRITICAL)

---

## 1. Mục đích & Phạm vi
Quy định các tiêu chí nghiệm thu chất lượng mối hàn bề mặt (SMD/SMT) trên bo mạch PCB công nghiệp theo cấp độ IPC Class 3 (thiết bị y tế, viễn thông, ô tô và hàng không vũ trụ).

## 2. Tiêu chí nghiệm thu IPC-A-610 Class 3
1. **Góc ướt thiếc hàn (Wetting Angle)**:
   - Mối hàn đạt chuẩn phải có góc ướt $\theta \le 90^\circ$ (tối ưu: $15^\circ - 45^\circ$).
   - Nghiêm cấm hiện tượng thiếc hàn không bám (non-wetting) hoặc tách lớp co rút (de-wetting).
2. **Khoảng cách điện môi tối thiểu (Minimum Electrical Clearance)**:
   - Khoảng cách giữa hai đường mạch hoặc chân linh kiện liền kề không được nhỏ hơn $0.13\text{ mm}$ (5 mils) hoặc 75% khoảng cách thiết kế CAD.
   - Bất kỳ cầu nối thiếc hàn (solder bridge) hoặc râu đồng (spur) vi phạm khoảng cách này đều bị phân loại là lỗi DEFECT (Không chấp nhận).
3. **Chiều cao góc lượn hàn (Fillet Height)**:
   - Chiều cao góc lượn chân linh kiện chíp (chip component) phải đạt tối thiểu $25\%$ chiều dày linh kiện hoặc $0.5\text{ mm}$.

## 3. Quy trình phản ứng khi phát hiện vi phạm
- **Vi phạm Class 2 (Cảnh báo)**:
  - Tăng tần suất lấy mẫu kiểm tra AOI lên $100\%$ trong 3 lô tiếp theo.
  - Gửi cảnh báo chất lượng đến kỹ sư công nghệ SMT phụ trách ca.
- **Vi phạm Class 3 (Hàn chập / Vi phạm khoảng cách an toàn)**:
  - Hệ thống AI tự động phân loại mức độ sự cố **CRITICAL**.
  - Đề xuất lệnh dừng dây chuyền nếu phát hiện chuỗi lỗi lặp lại.
  - Kiểm tra áp suất dao gạt và độ biến dạng của tấm Stencil tại trạm in kem hàn.
