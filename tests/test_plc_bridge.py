import os
import sys
import unittest
import time

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
        test_port = 5035
        server = VirtualModbusServer(host="127.0.0.1", port=test_port)
        server_started = server.start(daemon=True)
        self.assertTrue(server_started)

        try:
            bridge = PLCBridge(host="127.0.0.1", port=test_port, timeout=1.0)
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
            ok = bridge.update_telemetry(total=100, defects=5, yield_rate=95.0, defect_code=2)
            self.assertTrue(ok)

            bridge.disconnect()
            self.assertFalse(bridge.is_connected)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
