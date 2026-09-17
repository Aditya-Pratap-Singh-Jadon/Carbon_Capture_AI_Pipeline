"""SerialSource: pyserial-backed TelemetrySource implementation.

COM port, baud rate, and timeout are configurable; nothing is hardcoded.
Handles: connection failure, decoding errors, empty lines, timeout, clean shutdown.
"""

import serial
from typing import Iterator

from carbon_capture.input.source.base import TelemetrySource
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)


class SerialSource(TelemetrySource):
    """
    Reads raw telemetry lines from a configurable serial port.

    Compatible with any producer that speaks the agreed telemetry contract
    (Digital Twin over virtual COM, future Arduino over physical COM, etc.).
    The class has no knowledge of Digital Twin internals.
    """

    def __init__(
        self,
        port: str,
        baud_rate: int = 115200,
        timeout: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        port      : Serial port identifier (e.g. "COM11", "/dev/ttyUSB0").
        baud_rate : Line speed.  Must match the producer's configuration.
        timeout   : Per-read timeout in seconds.  1.0 s is a sensible default.
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self._connection: serial.Serial | None = None
        self._running = False

    # ------------------------------------------------------------------
    # TelemetrySource interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the serial port.  Raises serial.SerialException on failure."""
        try:
            self._connection = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
            )
            self._running = True
            logger.info(
                f"[INPUT] Connected to {self.port} at {self.baud_rate} baud "
                f"(timeout={self.timeout}s)."
            )
        except serial.SerialException as exc:
            logger.error(f"[INPUT] Serial connection failure on {self.port}: {exc}")
            raise

    def disconnect(self) -> None:
        """Close the serial port cleanly."""
        self._running = False
        if self._connection and self._connection.is_open:
            self._connection.close()
            logger.info(f"[INPUT] Disconnected from {self.port}.")

    def read_stream(self) -> Iterator[str]:
        """
        Yield raw decoded lines continuously until stopped or interrupted.

        - Empty / whitespace-only lines are silently skipped.
        - Decoding errors are handled with 'replace' strategy; bad bytes become U+FFFD.
        - A SerialException aborts the stream and logs an error.
        - KeyboardInterrupt (Ctrl+C) causes a clean shutdown.
        """
        if not self._connection or not self._connection.is_open:
            raise RuntimeError(
                f"Cannot read stream: not connected to {self.port}."
            )

        logger.info(f"[INPUT] Streaming from {self.port} …")

        while self._running:
            try:
                raw: bytes = self._connection.readline()
            except serial.SerialException as exc:
                logger.error(f"[INPUT] Serial read error on {self.port}: {exc}")
                break
            except KeyboardInterrupt:
                logger.info("[INPUT] Stream terminated by user (Ctrl+C).")
                break

            if not raw:
                # Timeout fired without a newline — no data this cycle.
                continue

            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            yield line

    # ------------------------------------------------------------------
    # Context manager support (optional convenience)
    # ------------------------------------------------------------------

    def __enter__(self) -> "SerialSource":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()
