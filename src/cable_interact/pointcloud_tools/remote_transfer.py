"""Asynchronously copy completed capture directories to a remote PC with scp."""

from __future__ import annotations

import queue
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional


LogFunction = Callable[[str], None]


class ScpTransferQueue:
    """Run scp transfers on one background worker so capture callbacks stay fast."""

    def __init__(
        self,
        destination: str,
        *,
        timeout_sec: float = 300.0,
        log_info: LogFunction = print,
        log_error: LogFunction = print,
    ) -> None:
        destination = destination.strip()
        if not destination:
            raise ValueError("scp destination must not be empty")
        if ":" not in destination:
            raise ValueError(
                "scp destination must look like user@host:/remote/folder/"
            )
        self.destination = destination
        self.timeout_sec = timeout_sec
        self.log_info = log_info
        self.log_error = log_error
        self._queue: queue.Queue[Optional[Path]] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="scp-transfer-worker",
            daemon=True,
        )
        self._thread.start()
        self.log_info(f"Background SCP upload enabled: {self.destination}")

    def submit(self, path: Path) -> None:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            self.log_error(f"Skipping SCP upload because path does not exist: {path}")
            return
        self._queue.put(path)
        self.log_info(f"Queued SCP upload: {path}")

    def close(self, *, wait: bool = True) -> None:
        self._queue.put(None)
        if wait:
            self.log_info("Waiting for queued SCP uploads to finish...")
            self._thread.join()

    def _run(self) -> None:
        while True:
            path = self._queue.get()
            try:
                if path is None:
                    return
                self._copy(path)
            finally:
                self._queue.task_done()

    def _copy(self, path: Path) -> None:
        command = [
            "scp",
            "-r",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            str(path),
            self.destination,
        ]
        self.log_info(f"Uploading with SCP: {path.name}")
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            self.log_error(f"SCP upload timed out after {self.timeout_sec:.1f}s: {path}")
            return
        except OSError as exc:
            self.log_error(f"Could not start scp for {path}: {exc}")
            return
        if result.returncode == 0:
            self.log_info(f"SCP upload complete: {path}")
            return
        detail = result.stderr.strip() or result.stdout.strip() or "unknown scp error"
        self.log_error(f"SCP upload failed for {path}: {detail}")
