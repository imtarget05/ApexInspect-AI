import asyncio
import threading
import time
from typing import Optional, Tuple, Any

try:
    from pymodbus.server import StartAsyncTcpServer, ServerAsyncStop
    from pymodbus.datastore import (
        ModbusSequentialDataBlock,
        ModbusServerContext,
        ModbusDeviceContext
    )
    PYMODBUS_AVAILABLE = True
except ImportError:
    PYMODBUS_AVAILABLE = False


class VirtualModbusServer:
    """
    Virtual Modbus TCP PLC Server for testing, local development, and CI/CD pipelines.
    Emulates a physical Siemens/Omron SMT conveyor PLC controller on localhost.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5020):
        self.host = host
        self.port = port
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.store: Optional[Any] = None
        self.context: Optional[Any] = None

    def start(self, daemon: bool = True) -> bool:
        """Starts the virtual Modbus TCP server in a background thread."""
        if not PYMODBUS_AVAILABLE:
            print("[VirtualPLC] pymodbus is not installed. Cannot start virtual server.")
            return False

        if self.running:
            return True

        self._loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(self._loop)
            self.store = ModbusDeviceContext(
                di=ModbusSequentialDataBlock(1, [0] * 50),
                co=ModbusSequentialDataBlock(1, [1, 0, 0, 0, 0, 1] + [0] * 44),  # Default: conveyor running, tower green
                hr=ModbusSequentialDataBlock(1, [0] * 50),
                ir=ModbusSequentialDataBlock(1, [0] * 50),
            )
            self.context = ModbusServerContext(devices=self.store, single=True)
            self.running = True

            async def _start():
                await StartAsyncTcpServer(
                    context=self.context,
                    address=(self.host, self.port)
                )

            try:
                self._loop.run_until_complete(_start())
            except Exception as e:
                self.running = False

        self._thread = threading.Thread(target=_run, daemon=daemon)
        self._thread.start()

        # Wait briefly for socket binding
        time.sleep(0.3)
        return True

    def stop(self) -> None:
        """Stops the virtual Modbus server."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


if __name__ == "__main__":
    server = VirtualModbusServer(port=5020)
    print(f"[VirtualPLC] Starting Virtual Modbus TCP Server on 127.0.0.1:5020...")
    server.start(daemon=False)
