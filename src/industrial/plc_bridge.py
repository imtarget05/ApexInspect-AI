import os
import time
from typing import Dict, Any, Optional

try:
    from pymodbus.client import ModbusTcpClient
    PYMODBUS_AVAILABLE = True
except ImportError:
    PYMODBUS_AVAILABLE = False


class PLCBridge:
    """
    Industrial Modbus TCP PLC Communication Bridge.
    Connects ApexInspect AI to SMT line controllers (e.g., Siemens S7-1200, Omron NX1P2, Beckhoff).
    Controls conveyor interlocks, pneumatic reject diverters, and andon tower lights.
    """

    # Coils (Discrete Outputs)
    COIL_CONVEYOR_RUN = 0      # 0 = Stopped, 1 = Running
    COIL_HALT_LINE = 1         # 1 = Emergency / Quality Halt triggered
    COIL_REWORK_DIVERT = 2     # 1 = Pulse pneumatic cylinder to reject bin
    COIL_TOWER_RED = 3         # 1 = Red Tower Light (Critical Fault)
    COIL_TOWER_YELLOW = 4      # 1 = Yellow Tower Light (Warning/Action Required)
    COIL_TOWER_GREEN = 5       # 1 = Green Tower Light (Normal Production)

    # Holding Registers
    REG_TOTAL_INSPECTED = 0    # Total count of items inspected
    REG_DEFECT_COUNT = 1       # Total defect count
    REG_YIELD_RATE = 2         # Yield rate (x100, e.g. 9850 = 98.50%)
    REG_LAST_DEFECT_CODE = 3   # Defect code index

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 1.0,
        simulation_fallback: bool = True,
        mode: Optional[str] = None,
    ):
        self.host = host or os.getenv("PLC_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("PLC_PORT", "5020"))
        self.timeout = timeout
        # ``simulation_fallback`` remains accepted for callers using the old
        # constructor, but it can never turn a failed hardware write into a
        # simulated success.  Explicit hardware mode is opt-in.
        self.simulation_fallback = simulation_fallback
        configured_mode = mode or os.getenv("APEX_PLC_MODE", "simulation")
        self.mode = self._parse_mode(configured_mode)
        self.client: Optional[Any] = None
        self._is_hardware_connected = False
        self._hardware_unavailable_reason = "SIMULATION_MODE"

        # Internal simulated memory state for offline fallback & testing
        self._simulated_coils = {
            self.COIL_CONVEYOR_RUN: True,
            self.COIL_HALT_LINE: False,
            self.COIL_REWORK_DIVERT: False,
            self.COIL_TOWER_RED: False,
            self.COIL_TOWER_YELLOW: False,
            self.COIL_TOWER_GREEN: True,
        }
        self._simulated_registers = {
            self.REG_TOTAL_INSPECTED: 0,
            self.REG_DEFECT_COUNT: 0,
            self.REG_YIELD_RATE: 10000,
            self.REG_LAST_DEFECT_CODE: 0,
        }

        if self.mode == "HARDWARE":
            self.connect()

    @staticmethod
    def _parse_mode(value: str) -> str:
        normalized = value.strip().upper()
        if normalized in {"SIMULATION", "SIMULATED"}:
            return "SIMULATION"
        if normalized == "HARDWARE":
            return "HARDWARE"
        raise ValueError("APEX_PLC_MODE must be 'simulation' or 'hardware'")

    def connect(self) -> bool:
        """Attempts to connect to the Modbus TCP PLC server."""
        if self.mode != "HARDWARE":
            self._is_hardware_connected = False
            self._hardware_unavailable_reason = "SIMULATION_MODE"
            return False
        if not PYMODBUS_AVAILABLE:
            self._is_hardware_connected = False
            self._hardware_unavailable_reason = "PYMODBUS_UNAVAILABLE"
            return False

        try:
            self.client = ModbusTcpClient(self.host, port=self.port, timeout=self.timeout)
            connected = self.client.connect()
            self._is_hardware_connected = bool(connected)
            self._hardware_unavailable_reason = None if connected else "PLC_UNAVAILABLE"
            return self._is_hardware_connected
        except Exception as exc:
            self._is_hardware_connected = False
            self._hardware_unavailable_reason = f"PLC_CONNECTION_ERROR: {exc}"
            return False

    def disconnect(self) -> None:
        """Closes the Modbus TCP connection."""
        if self.client and hasattr(self.client, "close"):
            try:
                self.client.close()
            except Exception:
                pass
        self._is_hardware_connected = False

    @property
    def is_connected(self) -> bool:
        """Returns True if connected to real PLC hardware or server."""
        if self.mode != "HARDWARE":
            return False
        if self.client and hasattr(self.client, "connected"):
            return bool(self.client.connected)
        return self._is_hardware_connected

    def _result(self, action: str, line_id: str, status: str, **details: Any) -> Dict[str, Any]:
        return {
            "action": action,
            "line_id": line_id,
            "mode": self.mode,
            "status": status,
            "hardware_connected": self.is_connected,
            **details,
        }

    @staticmethod
    def _write_succeeded(response: Any) -> bool:
        """Modbus writes must acknowledge success; an error response is a failure."""
        return response is not None and not (
            hasattr(response, "isError") and response.isError()
        )

    def _write_coils(self, writes: tuple[tuple[int, bool], ...]) -> bool:
        """Write a command sequence, stopping immediately when the PLC rejects one coil."""
        for address, value in writes:
            if not self._write_succeeded(self.client.write_coil(address, value)):
                return False
        return True

    def _hardware_unavailable_result(self, action: str, line_id: str) -> Dict[str, Any]:
        return self._result(
            action,
            line_id,
            "FAILED",
            error="PLC_UNAVAILABLE",
            hardware_unavailable_reason=self._hardware_unavailable_reason,
        )

    def halt_line(self, line_id: str = "SMT-LINE-01") -> Dict[str, Any]:
        """
        Commands the PLC to immediately cut power to the conveyor motor and engage RED tower light.
        """
        if self.mode == "HARDWARE":
            if not self.is_connected:
                return self._hardware_unavailable_result("HALT_LINE", line_id)
            try:
                writes = (
                    (self.COIL_HALT_LINE, True),
                    (self.COIL_CONVEYOR_RUN, False),
                    (self.COIL_TOWER_RED, True),
                    (self.COIL_TOWER_GREEN, False),
                    (self.COIL_TOWER_YELLOW, False),
                )
                if not self._write_coils(writes):
                    return self._result("HALT_LINE", line_id, "FAILED", error="PLC_WRITE_REJECTED")
            except Exception as exc:
                return self._result("HALT_LINE", line_id, "FAILED", error=str(exc))

        # Update local / simulated state
        self._simulated_coils[self.COIL_HALT_LINE] = True
        self._simulated_coils[self.COIL_CONVEYOR_RUN] = False
        self._simulated_coils[self.COIL_TOWER_RED] = True
        self._simulated_coils[self.COIL_TOWER_GREEN] = False
        self._simulated_coils[self.COIL_TOWER_YELLOW] = False

        return self._result(
            "HALT_LINE", line_id, "DISPATCHED" if self.mode == "HARDWARE" else "SIMULATED",
            conveyor_running=False,
            tower_light="RED",
        )

    def resume_line(self, line_id: str = "SMT-LINE-01") -> Dict[str, Any]:
        """
        Releases safety interlocks, turns GREEN tower light ON, and signals conveyor ready to run.
        """
        if self.mode == "HARDWARE":
            if not self.is_connected:
                return self._hardware_unavailable_result("RESUME_LINE", line_id)
            try:
                writes = (
                    (self.COIL_HALT_LINE, False),
                    (self.COIL_CONVEYOR_RUN, True),
                    (self.COIL_TOWER_RED, False),
                    (self.COIL_TOWER_YELLOW, False),
                    (self.COIL_TOWER_GREEN, True),
                )
                if not self._write_coils(writes):
                    return self._result("RESUME_LINE", line_id, "FAILED", error="PLC_WRITE_REJECTED")
            except Exception as exc:
                return self._result("RESUME_LINE", line_id, "FAILED", error=str(exc))

        self._simulated_coils[self.COIL_HALT_LINE] = False
        self._simulated_coils[self.COIL_CONVEYOR_RUN] = True
        self._simulated_coils[self.COIL_TOWER_RED] = False
        self._simulated_coils[self.COIL_TOWER_YELLOW] = False
        self._simulated_coils[self.COIL_TOWER_GREEN] = True

        return self._result(
            "RESUME_LINE", line_id, "DISPATCHED" if self.mode == "HARDWARE" else "SIMULATED",
            conveyor_running=True,
            tower_light="GREEN",
        )

    def divert_rework(self, line_id: str = "SMT-LINE-01") -> Dict[str, Any]:
        """
        Actuates pneumatic diverter gate to push defective PCB onto rework conveyor.
        """
        if self.mode == "HARDWARE":
            if not self.is_connected:
                return self._hardware_unavailable_result("ROUTE_REWORK", line_id)
            try:
                writes = (
                    (self.COIL_REWORK_DIVERT, True),
                    (self.COIL_TOWER_YELLOW, True),
                )
                if not self._write_coils(writes):
                    return self._result("ROUTE_REWORK", line_id, "FAILED", error="PLC_WRITE_REJECTED")
            except Exception as exc:
                return self._result("ROUTE_REWORK", line_id, "FAILED", error=str(exc))

        self._simulated_coils[self.COIL_REWORK_DIVERT] = True
        self._simulated_coils[self.COIL_TOWER_YELLOW] = True

        return self._result(
            "ROUTE_REWORK", line_id, "DISPATCHED" if self.mode == "HARDWARE" else "SIMULATED",
            diverter_active=True,
            tower_light="YELLOW",
        )

    def update_telemetry(self, total: int, defects: int, yield_rate: float, defect_code: int = 0) -> Dict[str, Any]:
        """Syncs real-time edge telemetry into PLC holding registers for SCADA visualization."""
        scaled_yield = max(0, min(10000, int(round(yield_rate * 100))))
        if self.mode == "HARDWARE":
            if not self.is_connected:
                return self._hardware_unavailable_result("UPDATE_TELEMETRY", "")
            try:
                response = self.client.write_registers(
                    self.REG_TOTAL_INSPECTED,
                    [total, defects, scaled_yield, defect_code]
                )
                if not self._write_succeeded(response):
                    return self._result("UPDATE_TELEMETRY", "", "FAILED", error="PLC_WRITE_REJECTED")
            except Exception as exc:
                return self._result("UPDATE_TELEMETRY", "", "FAILED", error=str(exc))

        self._simulated_registers[self.REG_TOTAL_INSPECTED] = total
        self._simulated_registers[self.REG_DEFECT_COUNT] = defects
        self._simulated_registers[self.REG_YIELD_RATE] = scaled_yield
        self._simulated_registers[self.REG_LAST_DEFECT_CODE] = defect_code
        return self._result(
            "UPDATE_TELEMETRY",
            "",
            "DISPATCHED" if self.mode == "HARDWARE" else "SIMULATED",
            total=total,
            defects=defects,
            yield_rate=scaled_yield / 100,
            defect_code=defect_code,
        )

    def read_plc_status(self) -> Dict[str, Any]:
        """Reads digital outputs and telemetry from PLC or simulated memory."""
        if self.is_connected:
            try:
                coils_res = self.client.read_coils(0, count=6)
                if not coils_res.isError():
                    bits = coils_res.bits
                    return {
                        "hardware_connected": True,
                        "conveyor_running": bool(bits[self.COIL_CONVEYOR_RUN]),
                        "halt_triggered": bool(bits[self.COIL_HALT_LINE]),
                        "rework_divert": bool(bits[self.COIL_REWORK_DIVERT]),
                        "tower_light": "RED" if bits[self.COIL_TOWER_RED] else ("YELLOW" if bits[self.COIL_TOWER_YELLOW] else "GREEN")
                    }
            except Exception as exc:
                self._hardware_unavailable_reason = f"PLC_READ_ERROR: {exc}"

        # Return simulated state
        coils = self._simulated_coils
        tower = "RED" if coils[self.COIL_TOWER_RED] else ("YELLOW" if coils[self.COIL_TOWER_YELLOW] else "GREEN")
        return {
            "hardware_connected": False,
            "mode": self.mode,
            "hardware_unavailable_reason": self._hardware_unavailable_reason,
            "conveyor_running": coils[self.COIL_CONVEYOR_RUN],
            "halt_triggered": coils[self.COIL_HALT_LINE],
            "rework_divert": coils[self.COIL_REWORK_DIVERT],
            "tower_light": tower
        }
