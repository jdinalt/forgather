"""
DiLoCo HTTP Parameter Server.

Standalone process that holds global model parameters, receives pseudo-gradients
from workers, and applies an outer optimizer step. Supports two modes:

- **Synchronous** (default): Workers block at a barrier until all have submitted.
  Server averages all pseudo-gradients and applies one outer optimizer step.

- **Asynchronous**: Workers submit and receive updated params immediately without
  waiting. Uses Delayed Nesterov (DN) momentum to avoid momentum amplification
  from stale gradients, and Dynamic Local Updates (DyLU) to adapt sync frequency
  per worker based on relative speed.

Usage:
    # Synchronous (default)
    server = DiLoCoServer(model_state_dict, num_workers=3, port=8512)

    # Asynchronous with DN momentum
    server = DiLoCoServer(model_state_dict, num_workers=3, port=8512,
                          async_mode=True, dn_buffer_size=3)

    server.run()   # Blocking
    server.start() # Non-blocking (background thread)
"""

import io
import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    """Registered worker metadata."""

    worker_id: str
    hostname: str
    registered_at: float
    last_heartbeat: float
    sync_round: int = 0
    last_sync_server_round: int = 0  # Server round when this worker last synced
    steps_per_second: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


def _default_outer_optimizer_factory(params):
    """Default outer optimizer: SGD with Nesterov momentum (DiLoCo paper defaults)."""
    return torch.optim.SGD(params, lr=0.7, momentum=0.9, nesterov=True)


def _serialize_state_dict(state_dict: Dict[str, torch.Tensor]) -> bytes:
    """Serialize a state dict to bytes using torch.save."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def _deserialize_state_dict(data: bytes) -> Dict[str, torch.Tensor]:
    """Deserialize bytes to a state dict using torch.load."""
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=True)


def _read_request_body(handler: BaseHTTPRequestHandler) -> bytes:
    """Read the full request body from an HTTP handler."""
    content_length = int(handler.headers.get("Content-Length", 0))
    return handler.rfile.read(content_length)


def _send_json_response(handler: BaseHTTPRequestHandler, data: dict, status: int = 200):
    """Send a JSON response."""
    body = json.dumps(data).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_tensor_response(handler: BaseHTTPRequestHandler, state_dict: Dict[str, torch.Tensor]):
    """Send a state dict as an octet-stream response."""
    data = _serialize_state_dict(state_dict)
    handler.send_response(200)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class DiLoCoServer:
    """
    Central DiLoCo parameter server.

    In synchronous mode, holds global model parameters, accepts pseudo-gradient
    submissions from workers, averages them when all workers have submitted
    (barrier), applies the outer optimizer, and returns updated global parameters.

    In asynchronous mode, applies each worker's pseudo-gradients immediately upon
    receipt. Uses Delayed Nesterov (DN) momentum to prevent momentum amplification
    from stale async gradients, and Dynamic Local Updates (DyLU) to recommend
    per-worker sync frequencies based on relative training speeds.

    Args:
        model_state_dict: Initial model parameters (e.g., from model.state_dict()).
        num_workers: Expected number of workers.
        port: HTTP server port. If None, auto-selects an available port.
        outer_optimizer_factory: Callable that takes a parameter list and returns
            a torch.optim.Optimizer. Defaults to SGD(lr=0.7, momentum=0.9, nesterov=True).
        host: Host address to bind to. Defaults to "0.0.0.0".
        save_dir: Directory for periodic server state saves. None disables.
        save_every_n_rounds: Save server state every N sync rounds.
        async_mode: If True, apply pseudo-gradients immediately without barrier.
        dn_buffer_size: Delayed Nesterov buffer size. In async mode, buffer this
            many pseudo-gradient submissions before applying the outer optimizer
            with momentum. Between buffered steps, apply simple gradient descent
            (no momentum). Set to 0 to disable DN (apply momentum every step).
            Only used in async mode.
        dylu_enabled: If True, compute per-worker recommended sync_every based
            on relative training speeds (Dynamic Local Updates). Only used in
            async mode.
        dylu_base_sync_every: Base sync_every for the fastest worker (H in paper).
            Slower workers get proportionally smaller values. Only used when
            dylu_enabled=True.
    """

    def __init__(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        num_workers: int,
        port: Optional[int] = None,
        outer_optimizer_factory: Optional[Callable] = None,
        host: str = "0.0.0.0",
        save_dir: Optional[str] = None,
        save_every_n_rounds: int = 10,
        async_mode: bool = False,
        dn_buffer_size: int = 0,
        dylu_enabled: bool = False,
        dylu_base_sync_every: int = 500,
    ):
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")

        self.num_workers = num_workers
        self.host = host
        self.port = port or self._find_available_port()
        self.save_dir = save_dir
        self.save_every_n_rounds = save_every_n_rounds
        self.async_mode = async_mode
        self.dn_buffer_size = dn_buffer_size
        self.dylu_enabled = dylu_enabled
        self.dylu_base_sync_every = dylu_base_sync_every

        # Global parameters - stored as nn.Parameters for the optimizer
        self._param_names: List[str] = list(model_state_dict.keys())
        self._param_list = torch.nn.ParameterList(
            [torch.nn.Parameter(v.clone().float().cpu(), requires_grad=False) for v in model_state_dict.values()]
        )

        # Outer optimizer
        factory = outer_optimizer_factory or _default_outer_optimizer_factory
        self.outer_optimizer = factory(self._param_list.parameters())

        # Extract outer LR for use in DN direct gradient steps
        self._outer_lr = self.outer_optimizer.param_groups[0]["lr"]

        # Worker registry
        self._workers: Dict[str, WorkerInfo] = {}
        self._workers_lock = threading.Lock()

        # Sync state - uses a Condition for proper barrier synchronization.
        # Each round is tracked by number; completed round results are stored
        # so threads that wake up late still get the correct result.
        self._sync_round = 0
        self._pending_pseudograds: Dict[str, Dict[str, torch.Tensor]] = {}
        self._sync_cond = threading.Condition()
        self._completed_rounds: Dict[int, Dict[str, torch.Tensor]] = {}

        # Async state
        self._async_lock = threading.Lock()
        self._total_submissions = 0  # Total pseudo-gradient submissions received

        # Delayed Nesterov (DN) state - buffer pseudo-gradients, apply momentum
        # only every dn_buffer_size submissions to avoid momentum amplification
        # from stale async gradients.
        self._dn_grad_buffer: List[Dict[str, torch.Tensor]] = []

        # Server state
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False
        self._started_at: Optional[float] = None

    def _find_available_port(self, start_port: int = 8512, max_attempts: int = 100) -> int:
        """Find an available port."""
        for i in range(max_attempts):
            port = start_port + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No available port in range {start_port}-{start_port + max_attempts}")

    def get_global_params(self) -> Dict[str, torch.Tensor]:
        """Get current global parameters as a state dict."""
        return {
            name: param.data.clone()
            for name, param in zip(self._param_names, self._param_list)
        }

    def _apply_outer_optimizer(self):
        """Average pending pseudo-gradients and apply the outer optimizer step."""
        n = len(self._pending_pseudograds)
        if n == 0:
            return

        # Average pseudo-gradients and set as .grad
        for i, name in enumerate(self._param_names):
            avg_grad = None
            for worker_pseudograds in self._pending_pseudograds.values():
                pg = worker_pseudograds[name].float()
                if avg_grad is None:
                    avg_grad = pg.clone()
                else:
                    avg_grad.add_(pg)
            avg_grad.div_(n)
            self._param_list[i].grad = avg_grad

        self.outer_optimizer.step()
        self.outer_optimizer.zero_grad()

        self._sync_round += 1
        self._pending_pseudograds.clear()

        logger.info(f"Outer optimizer step complete. Sync round: {self._sync_round}")

        # Periodic save
        if self.save_dir and self._sync_round % self.save_every_n_rounds == 0:
            self.save_state()

    def _apply_async_pseudograd(self, worker_id: str, pseudograds: Dict[str, torch.Tensor]):
        """
        Apply a single worker's pseudo-gradients in async mode.

        If DN is disabled (dn_buffer_size=0), applies the outer optimizer with
        full momentum on every submission.

        If DN is enabled, buffers pseudo-gradients and alternates between:
        - Direct gradient steps (param -= lr * grad) for intermediate submissions
        - Full outer optimizer steps (with momentum) every dn_buffer_size submissions
        """
        if self.dn_buffer_size <= 0:
            # No DN: apply outer optimizer directly on each submission
            for i, name in enumerate(self._param_names):
                self._param_list[i].grad = pseudograds[name].float()
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
        else:
            # DN: buffer gradients, alternate between direct and momentum steps
            self._dn_grad_buffer.append(pseudograds)

            if len(self._dn_grad_buffer) >= self.dn_buffer_size:
                # Full momentum step: average the buffer, apply outer optimizer
                n = len(self._dn_grad_buffer)
                for i, name in enumerate(self._param_names):
                    avg_grad = None
                    for buffered_pg in self._dn_grad_buffer:
                        pg = buffered_pg[name].float()
                        if avg_grad is None:
                            avg_grad = pg.clone()
                        else:
                            avg_grad.add_(pg)
                    avg_grad.div_(n)
                    self._param_list[i].grad = avg_grad

                self.outer_optimizer.step()
                self.outer_optimizer.zero_grad()
                self._dn_grad_buffer.clear()
            else:
                # Intermediate step: direct gradient descent (no momentum)
                # param -= lr * grad
                with torch.no_grad():
                    for i, name in enumerate(self._param_names):
                        self._param_list[i].data.sub_(
                            self._outer_lr * pseudograds[name].float()
                        )

        self._sync_round += 1
        self._total_submissions += 1

        # Periodic save
        if self.save_dir and self._sync_round % self.save_every_n_rounds == 0:
            self.save_state()

    def _compute_dylu_sync_every(self, worker_id: str) -> Optional[int]:
        """
        Compute recommended sync_every for a worker using Dynamic Local Updates.

        DyLU adjusts each worker's sync frequency proportional to its speed
        relative to the fastest worker: H_w = floor((v_w / v_max) * H_base).
        This ensures faster workers contribute more updates while slower workers
        don't become bottlenecks.

        Returns None if not enough speed data is available.
        """
        if not self.dylu_enabled:
            return None

        with self._workers_lock:
            speeds = {
                wid: w.steps_per_second
                for wid, w in self._workers.items()
                if w.steps_per_second > 0
            }

        if not speeds or worker_id not in speeds:
            return None

        max_speed = max(speeds.values())
        if max_speed <= 0:
            return None

        worker_speed = speeds[worker_id]
        recommended = max(1, int((worker_speed / max_speed) * self.dylu_base_sync_every))
        return recommended

    def _handle_register(self, handler: BaseHTTPRequestHandler):
        """Handle worker registration."""
        body = _read_request_body(handler)
        info = json.loads(body.decode("utf-8"))
        worker_id = info["worker_id"]

        with self._workers_lock:
            if worker_id in self._workers:
                logger.warning(f"Worker {worker_id} re-registering (replacing old entry)")

            self._workers[worker_id] = WorkerInfo(
                worker_id=worker_id,
                hostname=info.get("hostname", "unknown"),
                registered_at=time.time(),
                last_heartbeat=time.time(),
                extra=info.get("extra", {}),
            )
            num_registered = len(self._workers)

        logger.info(f"Worker {worker_id} registered ({num_registered}/{self.num_workers})")

        # Return global params
        if self.async_mode:
            with self._async_lock:
                _send_tensor_response(handler, self.get_global_params())
        else:
            _send_tensor_response(handler, self.get_global_params())

    def _handle_submit_pseudograd(self, handler: BaseHTTPRequestHandler):
        """
        Handle pseudo-gradient submission.

        In sync mode: blocks until all workers submit, then applies the outer
        optimizer and returns updated global params to all workers.

        In async mode: applies the pseudo-gradient immediately and returns
        updated global params without waiting.
        """
        # Read the request: length-prefixed JSON header + tensor payload
        body = _read_request_body(handler)

        header_len = struct.unpack("!I", body[:4])[0]
        header = json.loads(body[4 : 4 + header_len].decode("utf-8"))
        tensor_data = body[4 + header_len :]

        worker_id = header["worker_id"]
        pseudograds = _deserialize_state_dict(tensor_data)

        if self.async_mode:
            self._handle_submit_async(handler, worker_id, pseudograds)
        else:
            self._handle_submit_sync(handler, worker_id, pseudograds)

    def _handle_submit_sync(self, handler, worker_id, pseudograds):
        """Synchronous pseudo-gradient submission with barrier."""
        with self._sync_cond:
            my_round = self._sync_round

            self._pending_pseudograds[worker_id] = pseudograds

            with self._workers_lock:
                if worker_id in self._workers:
                    self._workers[worker_id].last_heartbeat = time.time()
                    self._workers[worker_id].sync_round += 1
                    self._workers[worker_id].last_sync_server_round = my_round + 1

            submitted = len(self._pending_pseudograds)
            logger.info(
                f"Worker {worker_id} submitted pseudograds "
                f"({submitted}/{self.num_workers}) for round {my_round}"
            )

            if submitted >= self.num_workers:
                self._apply_outer_optimizer()
                self._completed_rounds[my_round] = self.get_global_params()

                old_rounds = [r for r in self._completed_rounds if r < my_round - 1]
                for r in old_rounds:
                    del self._completed_rounds[r]

                self._sync_cond.notify_all()

            while my_round not in self._completed_rounds:
                if not self._sync_cond.wait(timeout=600):
                    _send_json_response(handler, {"error": "Sync timeout"}, 504)
                    return

            global_params = self._completed_rounds[my_round]

        _send_tensor_response(handler, global_params)

    def _handle_submit_async(self, handler, worker_id, pseudograds):
        """Asynchronous pseudo-gradient submission - apply immediately."""
        with self._async_lock:
            # Compute staleness: how many server updates since this worker last synced
            staleness = 0
            with self._workers_lock:
                if worker_id in self._workers:
                    w = self._workers[worker_id]
                    staleness = self._sync_round - w.last_sync_server_round
                    w.last_heartbeat = time.time()
                    w.sync_round += 1
                    w.last_sync_server_round = self._sync_round + 1

            logger.info(
                f"Worker {worker_id} submitted pseudograds (async), "
                f"staleness={staleness}, server_round={self._sync_round}"
            )

            # Apply this worker's pseudo-gradients immediately
            self._apply_async_pseudograd(worker_id, pseudograds)

            global_params = self.get_global_params()

        _send_tensor_response(handler, global_params)

    def _handle_get_global_params(self, handler: BaseHTTPRequestHandler):
        """Handle request for current global parameters."""
        if self.async_mode:
            with self._async_lock:
                _send_tensor_response(handler, self.get_global_params())
        else:
            _send_tensor_response(handler, self.get_global_params())

    def _handle_heartbeat(self, handler: BaseHTTPRequestHandler):
        """Handle worker heartbeat."""
        body = _read_request_body(handler)
        info = json.loads(body.decode("utf-8"))
        worker_id = info["worker_id"]

        with self._workers_lock:
            if worker_id in self._workers:
                self._workers[worker_id].last_heartbeat = time.time()
                if "steps_per_second" in info:
                    self._workers[worker_id].steps_per_second = info["steps_per_second"]

        # Compute DyLU recommendation if enabled
        recommended_sync_every = self._compute_dylu_sync_every(worker_id)

        response = {
            "status": "ok",
            "sync_round": self._sync_round,
            "num_workers": self.num_workers,
            "num_registered": len(self._workers),
        }
        if recommended_sync_every is not None:
            response["recommended_sync_every"] = recommended_sync_every

        _send_json_response(handler, response)

    def _handle_deregister(self, handler: BaseHTTPRequestHandler):
        """Handle worker deregistration."""
        body = _read_request_body(handler)
        info = json.loads(body.decode("utf-8"))
        worker_id = info["worker_id"]

        with self._workers_lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
                logger.info(f"Worker {worker_id} deregistered")

        _send_json_response(handler, {"status": "ok"})

    def _handle_status(self, handler: BaseHTTPRequestHandler):
        """Handle status request."""
        with self._workers_lock:
            workers = {
                wid: {
                    "hostname": w.hostname,
                    "registered_at": w.registered_at,
                    "last_heartbeat": w.last_heartbeat,
                    "sync_round": w.sync_round,
                    "last_sync_server_round": w.last_sync_server_round,
                    "steps_per_second": w.steps_per_second,
                }
                for wid, w in self._workers.items()
            }

        if self.async_mode:
            with self._async_lock:
                pending = []
                dn_buffered = len(self._dn_grad_buffer)
        else:
            with self._sync_cond:
                pending = list(self._pending_pseudograds.keys())
            dn_buffered = 0

        response = {
            "status": "running",
            "mode": "async" if self.async_mode else "sync",
            "sync_round": self._sync_round,
            "num_workers": self.num_workers,
            "num_registered": len(workers),
            "workers": workers,
            "pending_submissions": pending,
            "started_at": self._started_at,
            "uptime_seconds": time.time() - self._started_at if self._started_at else 0,
        }

        if self.async_mode:
            response["total_submissions"] = self._total_submissions
            response["dn_buffer_size"] = self.dn_buffer_size
            response["dn_buffered"] = dn_buffered
            response["dylu_enabled"] = self.dylu_enabled
            if self.dylu_enabled:
                response["dylu_base_sync_every"] = self.dylu_base_sync_every

        _send_json_response(handler, response)

    def _create_handler(self):
        """Create a request handler class bound to this server instance."""
        server_ref = self

        class DiLoCoRequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Route HTTP logs through our logger
                logger.debug(format, *args)

            def do_POST(self):
                try:
                    path = self.path.rstrip("/")
                    if path == "/register":
                        server_ref._handle_register(self)
                    elif path == "/submit_pseudograd":
                        server_ref._handle_submit_pseudograd(self)
                    elif path == "/heartbeat":
                        server_ref._handle_heartbeat(self)
                    elif path == "/deregister":
                        server_ref._handle_deregister(self)
                    else:
                        _send_json_response(self, {"error": f"Unknown endpoint: {path}"}, 404)
                except Exception as e:
                    logger.error(f"Error handling POST {self.path}: {e}", exc_info=True)
                    _send_json_response(self, {"error": str(e)}, 500)

            def do_GET(self):
                try:
                    path = self.path.rstrip("/")
                    if path == "/global_params":
                        server_ref._handle_get_global_params(self)
                    elif path == "/status":
                        server_ref._handle_status(self)
                    else:
                        _send_json_response(self, {"error": f"Unknown endpoint: {path}"}, 404)
                except Exception as e:
                    logger.error(f"Error handling GET {self.path}: {e}", exc_info=True)
                    _send_json_response(self, {"error": str(e)}, 500)

        return DiLoCoRequestHandler

    def save_state(self, path: Optional[str] = None):
        """Save server state (global params + outer optimizer) to disk."""
        import os

        save_path = path or self.save_dir
        if save_path is None:
            logger.warning("No save path specified, skipping save")
            return

        os.makedirs(save_path, exist_ok=True)

        state = {
            "global_params": self.get_global_params(),
            "outer_optimizer": self.outer_optimizer.state_dict(),
            "sync_round": self._sync_round,
            "num_workers": self.num_workers,
            "param_names": self._param_names,
            "async_mode": self.async_mode,
            "total_submissions": self._total_submissions,
        }

        save_file = os.path.join(save_path, f"diloco_server_state_round{self._sync_round}.pt")
        torch.save(state, save_file)
        logger.info(f"Server state saved to {save_file}")

        # Also save a "latest" copy
        latest_file = os.path.join(save_path, "diloco_server_state_latest.pt")
        torch.save(state, latest_file)

    def load_state(self, path: str):
        """Load server state from disk."""
        state = torch.load(path, map_location="cpu", weights_only=False)

        # Restore global params
        for i, name in enumerate(self._param_names):
            self._param_list[i].data.copy_(state["global_params"][name])

        # Restore outer optimizer
        self.outer_optimizer.load_state_dict(state["outer_optimizer"])

        self._sync_round = state["sync_round"]
        self._total_submissions = state.get("total_submissions", 0)
        logger.info(f"Server state loaded from {path}, at round {self._sync_round}")

    def run(self):
        """Run the server (blocking). Call this from the main process."""
        handler_class = self._create_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler_class)
        self._server.daemon_threads = True
        self._running = True
        self._started_at = time.time()

        mode = "async" if self.async_mode else "sync"
        logger.info(f"DiLoCo server starting on {self.host}:{self.port} (mode={mode})")
        logger.info(f"Expecting {self.num_workers} worker(s)")
        if self.async_mode and self.dn_buffer_size > 0:
            logger.info(f"Delayed Nesterov: buffer_size={self.dn_buffer_size}")
        if self.dylu_enabled:
            logger.info(f"DyLU enabled: base_sync_every={self.dylu_base_sync_every}")

        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server interrupted")
        finally:
            self._running = False
            self._server.server_close()
            logger.info("Server stopped")

    def start(self):
        """Start the server in a background thread (non-blocking)."""
        handler_class = self._create_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler_class)
        self._server.daemon_threads = True
        self._running = True
        self._started_at = time.time()

        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()

        mode = "async" if self.async_mode else "sync"
        logger.info(f"DiLoCo server started on {self.host}:{self.port} (mode={mode}, background)")
        logger.info(f"Expecting {self.num_workers} worker(s)")

    def stop(self):
        """Stop the background server."""
        if self._server:
            self._server.shutdown()
            self._running = False
            if self._server_thread:
                self._server_thread.join(timeout=5)
            self._server.server_close()
            logger.info("Server stopped")

    @property
    def address(self) -> str:
        """Return the server address as host:port."""
        return f"{self.host}:{self.port}"
