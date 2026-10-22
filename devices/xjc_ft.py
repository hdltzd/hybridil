"""XJC six-axis force/torque sensor driver.

The adapter implements the protocol used by the supplied XJC reference code.
It can either open a local serial device directly, or own a local ``socat``
PTY that connects to the AUBO controller's ``ser2net`` TCP endpoint. It never
changes device permissions or starts remote services. Sensor zeroing and
streaming are explicit operations.

Returned wrench order and units are ``[Fx, Fy, Fz, Tx, Ty, Tz]`` in N and Nm.
"""

import os
import struct
import subprocess
import time
from numbers import Integral, Real
from typing import Dict, Optional, Sequence

import numpy as np


class XjcError(RuntimeError):
    """Base XJC sensor error."""


class XjcConnectionError(XjcError):
    """The configured serial transport could not be opened."""


class XjcProtocolError(XjcError):
    """A sensor packet was incomplete or failed validation."""


class SocatSerialBridge:
    """Create ``local_port`` as a PTY connected to remote ser2net TCP."""

    def __init__(self, config: Dict) -> None:
        self.config = dict(config)
        self.process = None
        self.created_link = False
        self.local_port = self.config.get("local_port", "/tmp/ttyRobotTool")

    def start(self) -> str:
        host = self.config.get("host")
        tcp_port = self.config.get("tcp_port", 5000)
        if not isinstance(host, str) or not host:
            raise XjcConnectionError("XJC socat transport.host is required")
        if not isinstance(tcp_port, Integral) or not 0 < tcp_port <= 65535:
            raise XjcConnectionError("XJC socat tcp_port must be in 1..65535")
        if not isinstance(self.local_port, str) or not self.local_port.startswith("/"):
            raise XjcConnectionError("XJC socat local_port must be an absolute path")

        if os.path.exists(self.local_port):
            if self.config.get("reuse_existing", True):
                return self.local_port
            raise XjcConnectionError(
                "XJC local PTY {!r} already exists".format(self.local_port)
            )
        if os.path.lexists(self.local_port):
            raise XjcConnectionError(
                "XJC local PTY {!r} is a stale link; remove it after confirming no "
                "socat process owns it".format(self.local_port)
            )

        try:
            self.process = subprocess.Popen(
                [
                    "socat",
                    "PTY,raw,echo=0,link={}".format(self.local_port),
                    "TCP:{}:{}".format(host, int(tcp_port)),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as error:
            raise XjcConnectionError("The XJC socat transport requires 'socat'") from error

        timeout_s = float(self.config.get("startup_timeout_s", 3.0))
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            self.close()
            raise ValueError("XJC socat startup_timeout_s must be positive")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if os.path.exists(self.local_port):
                self.created_link = True
                return self.local_port
            if self.process.poll() is not None:
                error_text = self.process.stderr.read() if self.process.stderr else ""
                self.close()
                raise XjcConnectionError(
                    "socat could not connect XJC transport {}:{}: {}".format(
                        host,
                        tcp_port,
                        error_text.strip(),
                    )
                )
            time.sleep(0.02)
        self.close()
        raise XjcConnectionError(
            "Timed out creating XJC PTY {} for {}:{}".format(
                self.local_port,
                host,
                tcp_port,
            )
        )

    def close(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1.0)
            self.process = None
        if self.created_link and os.path.islink(self.local_port):
            os.unlink(self.local_port)
        self.created_link = False


class XjcForceTorqueSensor:
    """Minimal synchronous XJC protocol adapter."""

    SUPPORTED_RATES = (100, 250, 500)
    STREAM_HEADER = b"\x20\x4e"
    DISABLE_STREAM_COMMAND = b"\xff" * 11

    ZERO_COMMANDS = {
        0x01: bytes.fromhex("01 10 46 1C 00 02 04 00 00 00 00 E8 95"),
        0x09: bytes.fromhex("09 10 46 1C 00 02 04 00 00 00 00 C2 F5"),
    }
    POLL_COMMANDS = {
        0x01: bytes.fromhex("01 04 00 00 00 0C F0 0F"),
        0x09: bytes.fromhex("09 04 00 54 00 0C B0 97"),
    }
    STREAM_COMMANDS = {
        (0x01, 100): bytes.fromhex("01 10 01 9A 00 01 02 00 00 AB 6A"),
        (0x01, 250): bytes.fromhex("01 10 01 9A 00 01 02 00 01 6A AA"),
        (0x01, 500): bytes.fromhex("01 10 01 9A 00 01 02 00 02 2A AB"),
        (0x09, 100): bytes.fromhex("09 10 01 9A 00 01 02 00 00 CC AA"),
        (0x09, 250): bytes.fromhex("09 10 01 9A 00 01 02 00 01 0D 6A"),
        (0x09, 500): bytes.fromhex("09 10 01 9A 00 01 02 00 02 4D 6B"),
    }

    def __init__(
        self,
        port: Optional[str] = "/dev/ttyUSB0",
        baudrate: int = 115200,
        timeout_s: float = 0.1,
        slave_address: int = 0x01,
        sample_rate: int = 250,
        usb_selector: Optional[str] = None,
        transport: Optional[Dict] = None,
    ) -> None:
        self._validate_parameters(
            port,
            baudrate,
            timeout_s,
            slave_address,
            sample_rate,
            usb_selector,
        )
        if transport is not None and not isinstance(transport, dict):
            raise ValueError("transport must be an object or null")
        try:
            import serial
            import serial.tools.list_ports
        except ImportError as error:
            raise ImportError(
                "pyserial is required to create an XjcForceTorqueSensor"
            ) from error

        self.serial_module = serial
        self.baudrate = int(baudrate)
        self.timeout_s = float(timeout_s)
        self.slave_address = int(slave_address)
        self.sample_rate = int(sample_rate)
        self.streaming_enabled = False
        self.serial = None
        self.bridge = None
        self._closed = False

        selected_port = port
        if transport is not None:
            transport_type = transport.get("type", "direct")
            if transport_type == "socat_tcp":
                self.bridge = SocatSerialBridge(transport)
                selected_port = self.bridge.start()
            elif transport_type != "direct":
                raise ValueError(
                    "Unsupported XJC transport.type {!r}".format(transport_type)
                )
        if usb_selector is not None:
            selected_port = self.find_port(usb_selector)
        if selected_port is None:
            raise XjcConnectionError("No XJC serial port was selected")
        self.port = selected_port

        try:
            if "://" in self.port:
                self.serial = serial.serial_for_url(
                    self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout_s,
                )
            else:
                self.serial = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout_s,
                )
        except Exception as error:
            self.close()
            raise XjcConnectionError(
                "Failed to open XJC transport {!r}".format(self.port)
            ) from error

    @classmethod
    def _validate_parameters(
        cls,
        port,
        baudrate,
        timeout_s,
        slave_address,
        sample_rate,
        usb_selector,
    ) -> None:
        if port is not None and (not isinstance(port, str) or not port):
            raise ValueError("port must be a non-empty string or None")
        if not isinstance(baudrate, Integral) or baudrate <= 0:
            raise ValueError("baudrate must be a positive integer")
        if not isinstance(timeout_s, Real) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if slave_address not in cls.ZERO_COMMANDS:
            raise ValueError("slave_address must be 0x01 or 0x09")
        if sample_rate not in cls.SUPPORTED_RATES:
            raise ValueError("sample_rate must be 100, 250, or 500 Hz")
        if usb_selector is not None and (
            not isinstance(usb_selector, str) or not usb_selector
        ):
            raise ValueError("usb_selector must be a non-empty string or None")

    def find_port(self, selector: str) -> str:
        """Find a serial device by serial, location, HWID, or description."""
        matches = []
        for port in self.serial_module.tools.list_ports.comports():
            searchable = (
                getattr(port, "device", None),
                getattr(port, "serial_number", None),
                getattr(port, "location", None),
                getattr(port, "hwid", None),
                getattr(port, "description", None),
            )
            if any(selector == value or selector in str(value) for value in searchable if value):
                matches.append(port.device)
        if not matches:
            raise XjcConnectionError(
                "No serial device matched usb_selector={!r}".format(selector)
            )
        if len(matches) > 1:
            raise XjcConnectionError(
                "usb_selector={!r} matched multiple ports: {}".format(
                    selector,
                    matches,
                )
            )
        return matches[0]

    @staticmethod
    def crc16(data: bytes):
        crc = 0xFFFF
        for value in data:
            crc ^= value
            for _ in range(8):
                crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        return (crc >> 8) & 0xFF, crc & 0xFF

    def _require_open(self) -> None:
        if self._closed or self.serial is None or not self.serial.is_open:
            raise XjcConnectionError("XJC serial transport is closed")

    def _flush_input(self) -> None:
        self._require_open()
        self.serial.reset_input_buffer()

    def _read_exact(self, byte_count: int) -> bytes:
        deadline = time.monotonic() + self.timeout_s
        output = bytearray()
        while len(output) < byte_count and time.monotonic() < deadline:
            output.extend(self.serial.read(byte_count - len(output)))
        if len(output) != byte_count:
            raise XjcProtocolError(
                "Expected {} XJC bytes, received {}".format(
                    byte_count,
                    len(output),
                )
            )
        return bytes(output)

    @classmethod
    def _validate_crc(cls, packet: bytes) -> None:
        if len(packet) < 3:
            raise XjcProtocolError("XJC packet is too short")
        crc_high, crc_low = cls.crc16(packet[:-2])
        if packet[-2] != crc_low or packet[-1] != crc_high:
            raise XjcProtocolError("XJC packet CRC validation failed")

    def set_zero(self) -> None:
        """Explicitly zero the sensor; streaming must be disabled first."""
        self._require_open()
        if self.streaming_enabled:
            raise XjcProtocolError("Disable XJC streaming before zeroing")
        self._flush_input()
        self.serial.reset_output_buffer()
        self.serial.write(self.ZERO_COMMANDS[self.slave_address])
        response = self._read_exact(8)
        self._validate_crc(response)
        if response[0] != self.slave_address or response[1] != 0x10:
            raise XjcProtocolError("Unexpected XJC zero response")

    def read_polling(self) -> np.ndarray:
        """Request and return one big-endian float32 wrench sample."""
        self._require_open()
        if self.streaming_enabled:
            raise XjcProtocolError("Disable XJC streaming before polling")
        self._flush_input()
        self.serial.reset_output_buffer()
        self.serial.write(self.POLL_COMMANDS[self.slave_address])
        response = self._read_exact(29)
        self._validate_crc(response)
        if (
            response[0] != self.slave_address
            or response[1] != 0x04
            or response[2] != 24
        ):
            raise XjcProtocolError("Unexpected XJC polling response")
        return np.asarray(struct.unpack(">ffffff", response[3:27]), dtype=np.float64)

    def enable_streaming(self) -> None:
        self._require_open()
        if self.streaming_enabled:
            return
        self._flush_input()
        command = self.STREAM_COMMANDS[(self.slave_address, self.sample_rate)]
        self.serial.write(command)
        self.streaming_enabled = True

    def disable_streaming(self) -> None:
        if self._closed or self.serial is None or not self.serial.is_open:
            return
        self.serial.write(self.DISABLE_STREAM_COMMAND)
        self.serial.flush()
        self.serial.reset_input_buffer()
        self.streaming_enabled = False

    def read_streaming(self) -> np.ndarray:
        """Read one passive 16-byte XJC wrench packet."""
        self._require_open()
        if not self.streaming_enabled:
            raise XjcProtocolError("Enable XJC streaming before reading")

        deadline = time.monotonic() + self.timeout_s
        previous = None
        while time.monotonic() < deadline:
            current = self.serial.read(1)
            if not current:
                continue
            if previous == self.STREAM_HEADER[:1] and current == self.STREAM_HEADER[1:]:
                packet = self.STREAM_HEADER + self._read_exact(14)
                self._validate_crc(packet)
                raw = [
                    int.from_bytes(packet[index:index + 2], "little", signed=True)
                    for index in range(2, 14, 2)
                ]
                return np.concatenate(
                    [
                        np.asarray(raw[:3], dtype=np.float64) / 10.0,
                        np.asarray(raw[3:], dtype=np.float64) / 1000.0,
                    ]
                )
            previous = current
        raise XjcProtocolError("Timed out waiting for an XJC streaming header")

    def read(self) -> np.ndarray:
        return self.read_streaming() if self.streaming_enabled else self.read_polling()

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.serial is not None and self.serial.is_open:
                try:
                    self.disable_streaming()
                except Exception:
                    pass
                self.serial.close()
        finally:
            if self.bridge is not None:
                self.bridge.close()
            self._closed = True

    disconnect = close

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
