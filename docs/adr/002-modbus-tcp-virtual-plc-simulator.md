# docs/adr/002-modbus-tcp-virtual-plc-simulator.md
# ADR-002: Modbus TCP + Virtual PLC Simulator cho Industrial OT

## Status
Accepted (2026-09-16)

## Context
Hệ thống cần giao tiếp với industrial PLC (conveyor, reject diverter, andon tower light) qua Modbus TCP — protocol phổ biến trong SMT line. Tuy nhiên:

- Phòng thí nghiệm / dev environment không có hardware PLC thật.
- Cần test tích hợp Modbus mà không phụ thuộc hardware.
- Cần simulation fallback khi chạy production mà PLC offline.

## Decision
Thêm 2 thành phần:

1. **`PLCBridge`** (`src/industrial/plc_bridge.py`): Modbus TCP client bản quyền, hỗ trợ coil write/read, holding register, với **simulation fallback mode** khi không connect được.
2. **`VirtualModbusServer`** (`src/industrial/simulator_plc.py`): Server Modbus giả lập chạy local, dùng cho test integration và development.

## Decision drivers
- **pymodbus** là thư viện Python Modbus phổ biến, đã có trong requirements.txt.
- Virtual simulator cho phép test end-to-end tanpa hardware — quan trọng cho CI/CD và developer onboarding.
- Simulation fallback đảm bảo system không crash khi PLC offline (graceful degradation).

## Trade-offs
- Virtual simulator không phải real PLC — timing, error behavior khác hardware thật.
- Modbus TCP là protocol cũ, không có encryption/auth built-in — cần network-level security (VLAN, firewall).
- Simulation mode phải được flag rõ ràng để tránh confuse operator.

## Consequences
- Test `test_plc_bridge.py` cover cả simulation mode và virtual server.
- Production deployment phải document cách cấu hình PLC thật + network.
- Fallback mode ghi log rõ ràng: `SIMULATED` vs `DISPATCHED`.