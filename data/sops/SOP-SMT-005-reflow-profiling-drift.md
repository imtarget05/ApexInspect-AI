# SOP-SMT-005: Kiểm Soát & Hiệu Chuẩn Lệch Đường Cong Nhiệt Lò Hàn Hồi Lưu (Reflow Profile Drift)

**Mã tài liệu**: SOP-SMT-005  
**Tiêu chuẩn tham chiếu**: IPC-7530A (Guidelines for Temperature Profiling for Mass Soldering Processes)  
**Khu vực áp dụng**: Lò hàn hồi lưu đối lưu khí nóng (10-Zone Reflow Oven)  
**Mức độ rủi ro**: CAO (CRITICAL)

---

## 1. Định nghĩa sự cố trôi dạt nhiệt (Profile Drift)
Hiện tượng nhiệt độ thực tế tại các vùng gia nhiệt (Zones 1-10) hoặc trên bề mặt bo mạch bị sai lệch $> \pm 3^\circ\text{C}$ so với đường cong chuẩn đã thiết lập, dẫn đến tăng đột biến các lỗi hàn (hàn nguội, bi thiếc, hoặc quá nhiệt cháy mạch).

## 2. Các thông số kiểm soát giới hạn (PWI - Process Window Index)
- **Tốc độ dốc nhiệt gia nhiệt sơ bộ (Preheat Ramp Rate)**: $1.0 - 2.5^\circ\text{C/giây}$. Vượt quá ngưỡng này gây bắn tung tóe kem hàn (solder spatter) dẫn đến lỗi bi hàn và cầu hàn chập.
- **Thời gian ngâm nhiệt dung môi (Soak Time, 150 - 200°C)**: $60 - 120\text{ giây}$.
- **Thời gian trên điểm nóng chảy (Time Above Liquidus - TAL, > 217°C)**: $45 - 75\text{ giây}$ đối với hợp kim SAC305 (Sn96.5/Ag3.0/Cu0.5).
- **Nhiệt độ đỉnh (Peak Temperature)**: $235 - 245^\circ\text{C}$. Không được vượt quá $255^\circ\text{C}$ để tránh biến dạng PCB.

## 3. Quy trình khắc phục khi kích hoạt cảnh báo trôi tỷ lệ lỗi (Yield Drift)
1. **Dừng nạp bo mạch mới (Feed Hold)**:
   - Khi hệ thống AI phát hiện tỷ lệ lỗi trôi dạt $> 15\%$, tạm ngưng cấp bo mạch từ máy gắp đặt vào lò hàn.
2. **Đo đạc kiểm tra nhiệt độ độc lập (KIC / ECD Profiler)**:
   - Cho bo mạch kiểm chuẩn gắn 6 đầu nhiệt ngẫu liên (thermocouples) chạy qua lò.
   - Phân tích PWI. Nếu PWI $> 80\%$, tiến hành hiệu chuẩn bù nhiệt tại từng vùng.
3. **Kiểm tra quạt đối lưu và thanh gia nhiệt**:
   - Kiểm tra tốc độ vòng quay động cơ quạt đối lưu tuần hoàn khí.
   - Vệ sinh đường ống thu hồi dung môi trợ hàn (flux recovery system).
