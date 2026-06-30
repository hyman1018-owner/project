"""WebSocket bridge from AutoCamTracker V1.75 to the DockKit iOS app."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import ipaddress
import re
import socket
import subprocess
import threading
from time import monotonic, time
from typing import Any, Callable

from autocamtracker.core.telemetry_logger import TelemetryLogger

SOURCE_VERSION = "1.75"
CAMERA_FRAME_ENVELOPE_MAGIC = b"ACTF1"
CAMERA_FRAME_ENVELOPE_HEADER_BYTES = len(CAMERA_FRAME_ENVELOPE_MAGIC) + 8
FRAMING_ZOOM_FACTORS = {"wide": 1.0, "medium": 1.6, "close": 2.4}
CENTER_ZOOM_FACTOR = FRAMING_ZOOM_FACTORS["wide"]
LOST_ZOOM_HOLD_SECONDS = 1.0
LOST_ZOOM_RAMP_SECONDS = 2.0
COASTING_COMMAND_FRAMES = 12

_last_locked_zoom_factor = CENTER_ZOOM_FACTOR
_last_unlocked_at: float | None = None


def zoom_factor_for_framing(framing_mode: str | None) -> float:
    """Return the fixed iPhone display zoom for a desktop framing mode."""

    return FRAMING_ZOOM_FACTORS.get(framing_mode or "medium", FRAMING_ZOOM_FACTORS["medium"])


@dataclass(frozen=True)
class TrackingServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    path: str = "/ws/tracking"
    publish_hz: float = 20.0


@dataclass(frozen=True)
class MotorStatus:
    docked: bool
    manual_ready: bool
    system_tracking_enabled: bool | None
    last_error: str | None
    timestamp_ms: int
    current_velocity: dict[str, Any] | None = None
    last_command: dict[str, Any] | None = None
    last_stop_reason: str | None = None
    camera_zoom_factor: float | None = None
    camera_display_zoom_factor: float | None = None

    @property
    def ready(self) -> bool:
        return self.docked and self.manual_ready and self.system_tracking_enabled is False


def tracking_message(
    *,
    target_locked: bool,
    error_x: float = 0.0,
    error_y: float = 0.0,
    confidence: float = 0.0,
    target_id: int | None = None,
    sequence: int = 0,
    frame_width: int | None = None,
    frame_height: int | None = None,
    target_x: float | None = None,
    target_y: float | None = None,
    bbox_width: float | None = None,
    bbox_height: float | None = None,
    zoom_factor: float | None = None,
    predicted: bool = False,
) -> dict[str, Any]:
    """Build the versioned wire message consumed by TrackingCommand.swift."""

    message = {
        "type": "tracking",
        "version": "1.0",
        "source_version": SOURCE_VERSION,
        "sequence": sequence,
        "target_locked": bool(target_locked),
        "target_id": target_id,
        "error_x": max(-1.0, min(1.0, float(error_x))),
        "error_y": max(-1.0, min(1.0, float(error_y))),
        "confidence": max(0.0, min(1.0, float(confidence))),
        "timestamp_ms": int(time() * 1000),
    }
    if frame_width is not None and frame_height is not None:
        message.update(
            {
                "frame_width": int(frame_width),
                "frame_height": int(frame_height),
                "target_x": max(0.0, min(1.0, float(target_x or 0.0))),
                "target_y": max(0.0, min(1.0, float(target_y or 0.0))),
                "bbox_width": max(0.0, min(1.0, float(bbox_width or 0.0))),
                "bbox_height": max(0.0, min(1.0, float(bbox_height or 0.0))),
            }
        )
    if zoom_factor is not None:
        message["zoom_factor"] = max(0.1, min(10.0, float(zoom_factor)))
    if predicted:
        message["predicted_target"] = True
    return message


def frame_tracking_message(frame_data, frame_shape, sequence: int = 0) -> dict[str, Any]:
    """Convert pixel-space framing status into normalized gimbal error."""

    global _last_locked_zoom_factor, _last_unlocked_at

    frame_h, frame_w = frame_shape[:2]
    targets = frame_data.selected_targets
    fresh_target = next(
        (
            target
            for target in targets
            if (
                (target.status == "tracking" and target.lost_frame_count == 0)
                or (target.status == "coasting" and target.lost_frame_count <= COASTING_COMMAND_FRAMES)
            )
        ),
        None,
    )
    framing_mode = getattr(getattr(frame_data, "framing_status", None), "framing_mode", "medium")
    if fresh_target is None or frame_data.tracking_status != "tracking":
        now = monotonic()
        if _last_unlocked_at is None:
            _last_unlocked_at = now
        elapsed = now - _last_unlocked_at
        if elapsed <= LOST_ZOOM_HOLD_SECONDS:
            zoom_factor = _last_locked_zoom_factor
        else:
            ramp = min(1.0, (elapsed - LOST_ZOOM_HOLD_SECONDS) / LOST_ZOOM_RAMP_SECONDS)
            zoom_factor = _last_locked_zoom_factor + (CENTER_ZOOM_FACTOR - _last_locked_zoom_factor) * ramp
        return tracking_message(
            target_locked=False,
            sequence=sequence,
            zoom_factor=zoom_factor,
        )

    status = frame_data.framing_status
    bbox = fresh_target.bbox
    bbox_width = (bbox[2] - bbox[0]) / max(1.0, frame_w)
    target_id = frame_data.selected_global_vehicle_id
    if target_id is None:
        target_id = frame_data.selected_local_track_id
    zoom_factor = zoom_factor_for_framing(framing_mode)
    _last_locked_zoom_factor = zoom_factor
    _last_unlocked_at = None
    return tracking_message(
        target_locked=True,
        target_id=target_id,
        error_x=status.error_x / max(1.0, frame_w / 2.0),
        error_y=status.error_y / max(1.0, frame_h / 2.0),
        confidence=fresh_target.confidence,
        sequence=sequence,
        frame_width=frame_w,
        frame_height=frame_h,
        target_x=fresh_target.center[0] / max(1.0, frame_w),
        target_y=fresh_target.center[1] / max(1.0, frame_h),
        bbox_width=bbox_width,
        bbox_height=(bbox[3] - bbox[1]) / max(1.0, frame_h),
        zoom_factor=zoom_factor,
        predicted=fresh_target.status == "coasting",
    )


class TrackingWebSocketServer:
    """Runs a small asyncio WebSocket server without blocking Tkinter."""

    def __init__(
        self,
        config: TrackingServerConfig | None = None,
        on_status: Callable[[str], None] | None = None,
        on_control: Callable[[dict[str, Any]], None] | None = None,
        telemetry_logger: TelemetryLogger | None = None,
    ) -> None:
        self.config = config or TrackingServerConfig()
        self.on_status = on_status
        self.on_control = on_control
        self.telemetry_logger = telemetry_logger
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._clients: set[Any] = set()
        self._frame_lock = threading.Lock()
        self._motor_status_lock = threading.Lock()
        self._latest_motor_status: MotorStatus | None = None
        self._latest_frame_bytes: bytes | None = None
        self._latest_frame_info: dict[str, Any] = {}
        self._latest_decoded_frame_info: dict[str, Any] = {}
        self._latest_desktop_state: dict[str, Any] | None = None
        self._received_frame_count = 0
        self._sequence = 0
        self._last_publish_at = 0.0
        self._running = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def motor_status(self) -> MotorStatus | None:
        with self._motor_status_lock:
            return self._latest_motor_status

    @property
    def motor_ready(self) -> bool:
        status = self.motor_status
        return bool(status and status.ready)

    @property
    def local_urls(self) -> list[str]:
        interface_addresses = self._active_interface_addresses()
        addresses: set[str] = {address for _, address in interface_addresses}
        hostname = socket.gethostname()
        local_name = hostname if hostname.endswith(".local") else f"{hostname}.local"
        try:
            addresses.update(socket.gethostbyname_ex(hostname)[2])
        except OSError:
            pass
        usable = [address for address in addresses if ":" not in address and not address.startswith("127.")]
        link_local = sorted(address for address in usable if ipaddress.ip_address(address).is_link_local)
        private = sorted(
            address
            for address in usable
            if ipaddress.ip_address(address).is_private and address not in link_local
        )
        other = sorted(address for address in usable if address not in link_local and address not in private)
        # Prefer the normal LAN address. A 169.254 link-local address may be
        # present whenever an iPhone is attached by USB, but it is not always
        # routable from the app and previously became the misleading default.
        urls = [
            f"ws://{address}:{self.config.port}{self.config.path}"
            for address in (*private, *link_local, *other)
        ]
        urls.append(f"ws://{local_name}:{self.config.port}{self.config.path}")
        return urls

    @property
    def preferred_url(self) -> str:
        return self.local_urls[0]

    @staticmethod
    def _active_interface_addresses() -> list[tuple[str, str]]:
        """Return active macOS IPv4 interfaces, including USB link-local addresses."""

        try:
            result = subprocess.run(
                ["ifconfig"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            return []

        interfaces: dict[str, dict[str, Any]] = {}
        current: str | None = None
        for line in result.stdout.splitlines():
            match = re.match(r"^([a-zA-Z0-9]+):", line)
            if match:
                current = match.group(1)
                interfaces[current] = {"addresses": [], "active": False}
                continue
            if current is None:
                continue
            address_match = re.match(r"\s+inet (\d+\.\d+\.\d+\.\d+)\b", line)
            if address_match:
                interfaces[current]["addresses"].append(address_match.group(1))
            if line.strip() == "status: active":
                interfaces[current]["active"] = True

        return [
            (name, address)
            for name, state in interfaces.items()
            if state["active"]
            for address in state["addresses"]
            if not address.startswith("127.")
        ]

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._thread_main, name="tracking-websocket", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        loop = self._loop
        stop_event = self._stop_event
        if loop is not None and stop_event is not None and loop.is_running():
            loop.call_soon_threadsafe(stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._running.clear()

    def publish_frame(self, frame_data, frame_shape) -> None:
        interval = 1.0 / max(1.0, self.config.publish_hz)
        now = monotonic()
        if now - self._last_publish_at < interval:
            return
        self._last_publish_at = now
        self._sequence += 1
        payload = frame_tracking_message(frame_data, frame_shape, self._sequence)
        if getattr(frame_data, "receive_latency_ms", None) is not None:
            payload["receive_latency_ms"] = round(float(frame_data.receive_latency_ms), 2)
        if getattr(frame_data, "decode_time_ms", 0.0):
            payload["decode_time_ms"] = round(float(frame_data.decode_time_ms), 2)
        self.publish(payload)

    def publish_test_pulse(self, error_x: float = 0.12) -> None:
        self._sequence += 1
        self.publish(
            tracking_message(
                target_locked=True,
                target_id=999,
                error_x=error_x,
                confidence=1.0,
                sequence=self._sequence,
            )
        )

    def publish_stop(self, zoom_factor: float | None = CENTER_ZOOM_FACTOR) -> None:
        self._sequence += 1
        self.publish(
            tracking_message(
                target_locked=False,
                sequence=self._sequence,
                zoom_factor=zoom_factor,
            )
        )

    def publish_control(self, action: str) -> None:
        self.publish(
            {
                "type": "control",
                "action": action,
                "timestamp_ms": int(time() * 1000),
            }
        )

    def read_latest_frame(self):
        """Decode and consume only the newest iPhone JPEG frame."""

        with self._frame_lock:
            data = self._latest_frame_bytes
            info = dict(self._latest_frame_info)
            self._latest_frame_bytes = None
        if data is None:
            return None

        import cv2
        import numpy as np

        decoded_started_at = monotonic()
        encoded = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        decoded_at = monotonic()
        capture_timestamp_ms = info.get("capture_timestamp_ms")
        receive_latency_ms = None
        if capture_timestamp_ms is not None:
            receive_latency_ms = max(0.0, time() * 1000.0 - float(capture_timestamp_ms))
        self._latest_decoded_frame_info = {
            **info,
            "decode_time_ms": (decoded_at - decoded_started_at) * 1000.0,
            "receive_latency_ms": receive_latency_ms,
            "decoded_monotonic_s": decoded_at,
        }
        return frame

    def latest_frame_timing(self) -> dict[str, Any]:
        return dict(self._latest_decoded_frame_info)

    def publish(self, payload: dict[str, Any]) -> None:
        if payload.get("type") == "desktop_state":
            self._latest_desktop_state = dict(payload)
        if payload.get("type") in {"tracking", "desktop_state"}:
            self._log("ws_send", payload=payload)
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except Exception as exc:  # pragma: no cover - surfaced in the desktop UI
            self._notify(f"iPhone server failed: {exc}")
        finally:
            self._running.clear()
            self._loop = None
            self._stop_event = None

    async def _serve(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError as exc:
            raise RuntimeError("Install the 'websockets' dependency first") from exc

        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        async with serve(self._handle_client, self.config.host, self.config.port):
            self._running.set()
            self._notify("Waiting for iPhone")
            await self._stop_event.wait()

        self._clients.clear()

    async def _handle_client(self, websocket) -> None:
        from websockets.exceptions import ConnectionClosed

        request_path = getattr(getattr(websocket, "request", None), "path", "")
        if request_path.split("?", 1)[0] != self.config.path:
            await websocket.close(code=1008, reason="Unsupported path")
            return

        self._clients.add(websocket)
        self._log("ws_client_connected", path=request_path, client_count=len(self._clients))
        self._notify(f"iPhone connected ({len(self._clients)})")
        await websocket.send(json.dumps(tracking_message(target_locked=False, sequence=self._sequence)))
        if self._latest_desktop_state is not None:
            await websocket.send(json.dumps(self._latest_desktop_state, separators=(",", ":")))
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    self._accept_camera_frame(message)
                elif isinstance(message, str):
                    self._accept_text_message(message)
        except ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            if not self._clients:
                with self._motor_status_lock:
                    self._latest_motor_status = None
            self._log("ws_client_disconnected", client_count=len(self._clients))
            self._notify("iPhone disconnected" if not self._clients else f"iPhone connected ({len(self._clients)})")

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        if not self._clients:
            return
        message = json.dumps(payload, separators=(",", ":"))
        clients = list(self._clients)
        results = await asyncio.gather(*(client.send(message) for client in clients), return_exceptions=True)
        for client, result in zip(clients, results):
            if isinstance(result, Exception):
                self._clients.discard(client)

    def _accept_camera_frame(self, data: bytes) -> None:
        capture_timestamp_ms = None
        if data.startswith(CAMERA_FRAME_ENVELOPE_MAGIC):
            if len(data) < CAMERA_FRAME_ENVELOPE_HEADER_BYTES + 4:
                return
            capture_timestamp_ms = int.from_bytes(
                data[len(CAMERA_FRAME_ENVELOPE_MAGIC):CAMERA_FRAME_ENVELOPE_HEADER_BYTES],
                byteorder="big",
                signed=False,
            )
            data = data[CAMERA_FRAME_ENVELOPE_HEADER_BYTES:]
        if len(data) < 4 or len(data) > 2_000_000 or not data.startswith(b"\xff\xd8"):
            return
        received_at = monotonic()
        with self._frame_lock:
            self._latest_frame_bytes = data
            self._latest_frame_info = {
                "frame_count": self._received_frame_count + 1,
                "frame_bytes": len(data),
                "capture_timestamp_ms": capture_timestamp_ms,
                "received_monotonic_s": received_at,
            }
            self._received_frame_count += 1
            first_frame = self._received_frame_count == 1
        if first_frame:
            self._log("camera_frame_received", frame_bytes=len(data), frame_count=self._received_frame_count)
            self._notify("iPhone video receiving")
        elif self._received_frame_count % 150 == 0:
            self._log("camera_frame_received", frame_bytes=len(data), frame_count=self._received_frame_count)

    def _accept_text_message(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            return
        if payload.get("type") == "motor_status":
            self._accept_motor_status(payload)
        elif payload.get("type") == "control":
            self._accept_control(payload)

    def _accept_motor_status(self, payload: dict[str, Any] | str) -> None:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                return
        if payload.get("type") != "motor_status":
            return
        status = MotorStatus(
            docked=bool(payload.get("docked", False)),
            manual_ready=bool(payload.get("manual_ready", False)),
            system_tracking_enabled=payload.get("system_tracking_enabled"),
            last_error=str(payload["last_error"]) if payload.get("last_error") else None,
            timestamp_ms=int(payload.get("timestamp_ms", 0)),
            current_velocity=_dict_or_none(payload.get("current_velocity")),
            last_command=_dict_or_none(payload.get("last_command")),
            last_stop_reason=str(payload["last_stop_reason"]) if payload.get("last_stop_reason") else None,
            camera_zoom_factor=_float_or_none(payload.get("camera_zoom_factor")),
            camera_display_zoom_factor=_float_or_none(payload.get("camera_display_zoom_factor")),
        )
        with self._motor_status_lock:
            self._latest_motor_status = status
        self._log("motor_status", status=status)
        state = "motor ready" if status.ready else "motor not ready"
        if status.last_error:
            state = f"motor error: {status.last_error}"
        self._notify(f"iPhone connected · {state}")

    def _accept_control(self, payload: dict[str, Any]) -> None:
        action = str(payload.get("action") or "").strip()
        if not action or self.on_control is None:
            return
        self._log("ws_control", payload=payload)
        self.on_control(dict(payload))
        self._notify(f"iPhone control: {action}")

    def _notify(self, message: str) -> None:
        if self.on_status is not None:
            self.on_status(message)

    def _log(self, event: str, **fields: Any) -> None:
        if self.telemetry_logger is not None:
            self.telemetry_logger.log(event, **fields)


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
