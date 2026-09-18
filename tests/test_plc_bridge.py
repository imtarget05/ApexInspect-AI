import os
import sys
import unittest
import time
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.industrial.plc_bridge import PLCBridge
from src.industrial.simulator_plc import VirtualModbusServer


class TestPLCBridge(unittest.TestCase):
    """Verifies Modbus TCP industrial communication and simulation fallback."""

    def test_plc_bridge_offline_simulation_mode(self):
        """PLCBridge must gracefully handle offline hardware using simulated registers without crashing."""
        # Port 59999 has no running service
        bridge = PLCBridge(host="127.0.0.1", port=59999, timeout=0.2, simulation_fallback=True)
        self.assertFalse(bridge.is_connected)

        # Halt line in simulation mode
        halt_res = bridge.halt_line("SMT-LINE-01")
        self.assertEqual(halt_res["action"], "HALT_LINE")
        self.assertEqual(halt_res["status"], "SIMULATED")
        self.assertFalse(halt_res["conveyor_running"])
        self.assertEqual(halt_res["tower_light"], "RED")

        # Resume line in simulation mode
        resume_res = bridge.resume_line("SMT-LINE-01")
        self.assertEqual(resume_res["action"], "RESUME_LINE")
        self.assertEqual(resume_res["status"], "SIMULATED")
        self.assertTrue(resume_res["conveyor_running"])
        self.assertEqual(resume_res["tower_light"], "GREEN")

        # Divert rework in simulation mode
        divert_res = bridge.divert_rework("SMT-LINE-01")
        self.assertEqual(divert_res["action"], "ROUTE_REWORK")
        self.assertEqual(divert_res["tower_light"], "YELLOW")

    def test_plc_bridge_with_virtual_modbus_server(self):
        """PLCBridge must successfully connect, write coils, and sync registers with VirtualModbusServer."""
        if not __import__("src.industrial.plc_bridge", fromlist=["PYMODBUS_AVAILABLE"]).PYMODBUS_AVAILABLE:
            self.skipTest("pymodbus is not installed")
        test_port = 5035
        server = VirtualModbusServer(host="127.0.0.1", port=test_port)
        server_started = server.start(daemon=True)
        self.assertTrue(server_started)

        try:
            bridge = PLCBridge(host="127.0.0.1", port=test_port, timeout=1.0, mode="hardware")
            self.assertTrue(bridge.is_connected)

            # Test Halt line actuation
            halt_res = bridge.halt_line("SMT-LINE-01")
            self.assertEqual(halt_res["status"], "DISPATCHED")
            self.assertTrue(halt_res["hardware_connected"])
            self.assertEqual(halt_res["tower_light"], "RED")

            status = bridge.read_plc_status()
            self.assertTrue(status["halt_triggered"])
            self.assertFalse(status["conveyor_running"])
            self.assertEqual(status["tower_light"], "RED")

            # Test Resume line actuation
            resume_res = bridge.resume_line("SMT-LINE-01")
            self.assertEqual(resume_res["status"], "DISPATCHED")
            self.assertTrue(resume_res["conveyor_running"])
            self.assertEqual(resume_res["tower_light"], "GREEN")

            # Test Telemetry holding registers
            result = bridge.update_telemetry(total=100, defects=5, yield_rate=95.0, defect_code=2)
            self.assertEqual(result["status"], "DISPATCHED")

            bridge.disconnect()
            self.assertFalse(bridge.is_connected)
        finally:
            server.stop()

    def test_hardware_mode_rejects_unavailable_plc(self):
        """Changing an unavailable hardware result into simulation must fail this test."""
        bridge = PLCBridge(host="127.0.0.1", port=59999, timeout=0.01, mode="hardware")

        result = bridge.halt_line()

        self.assertEqual(result["mode"], "HARDWARE")
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error"], "PLC_UNAVAILABLE")
        self.assertFalse(result["hardware_connected"])

    def test_simulation_mode_never_connects_or_writes_to_hardware(self):
        """Removing simulation isolation must fail this test."""
        with patch("src.industrial.plc_bridge.ModbusTcpClient", create=True) as client_class:
            bridge = PLCBridge(mode="simulation")
            result = bridge.halt_line()

        client_class.assert_not_called()
        self.assertEqual(result["mode"], "SIMULATION")
        self.assertEqual(result["status"], "SIMULATED")

    def test_hardware_write_exception_returns_failed_without_simulation_claim(self):
        """Swallowing a hardware write failure as SIMULATED must fail this test."""
        bridge = PLCBridge(mode="simulation")
        bridge.mode = "HARDWARE"
        bridge._is_hardware_connected = True

        class FailingClient:
            connected = True

            @staticmethod
            def write_coil(*_args):
                raise RuntimeError("wire disconnected")

        bridge.client = FailingClient()
        result = bridge.divert_rework()

        self.assertEqual(result["mode"], "HARDWARE")
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("wire disconnected", result["error"])

    def test_hardware_stops_remaining_coil_writes_after_rejected_response(self):
        """Continuing a multi-coil command after a rejection must fail this test."""
        bridge = PLCBridge(mode="simulation")
        bridge.mode = "HARDWARE"
        bridge._is_hardware_connected = True

        class RejectedResponse:
            @staticmethod
            def isError():
                return True

        class RejectingClient:
            connected = True
            calls = []

            @classmethod
            def write_coil(cls, address, value):
                cls.calls.append((address, value))
                return RejectedResponse()

        bridge.client = RejectingClient()
        result = bridge.halt_line()

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error"], "PLC_WRITE_REJECTED")
        self.assertEqual(RejectingClient.calls, [(bridge.COIL_HALT_LINE, True)])

    def test_telemetry_write_exception_returns_structured_failure(self):
        """Returning success after a telemetry write exception must fail this test."""
        bridge = PLCBridge(mode="simulation")
        bridge.mode = "HARDWARE"
        bridge._is_hardware_connected = True

        class FailingClient:
            connected = True

            @staticmethod
            def write_registers(*_args):
                raise RuntimeError("register write failed")

        bridge.client = FailingClient()
        result = bridge.update_telemetry(total=100, defects=5, yield_rate=95.0)

        self.assertEqual(result["mode"], "HARDWARE")
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("register write failed", result["error"])


if __name__ == "__main__":
    unittest.main()
