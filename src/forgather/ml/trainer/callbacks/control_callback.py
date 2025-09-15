"""
TrainerControlCallback - Enable external control of running training jobs

Provides HTTP-based API for sending control commands to running distributed training jobs.
Only rank 0 runs the HTTP server and broadcasts commands to other ranks.
"""

import asyncio
import json
import logging
import os
import signal
import socket
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import aiohttp
    import aiohttp.web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

import torch
import torch.distributed as dist

from ..trainer_types import (
    MinimalTrainingArguments,
    TrainerState,
    TrainerControl,
    TrainerCallback,
)

logger = logging.getLogger(__name__)


@dataclass
class ControlCommand:
    """Represents a control command sent to the trainer."""

    command: str
    timestamp: float
    data: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "timestamp": self.timestamp,
            "data": self.data or {},
        }


COMMAND_CODES = {
    "graceful_stop": 1,
    "save_checkpoint": 2,
    "save_and_stop": 3,
    "status": 4,
    "abort": 5,
}

COMMAND_NAMES = {v: k for k, v in COMMAND_CODES.items()}


class TrainerControlCallback(TrainerCallback):
    """
    Callback that enables external control of training jobs via HTTP API.

    Features:
    - Graceful stop: Stop training cleanly after current step
    - Save checkpoint: Trigger checkpoint save (with evaluation if needed)
    - Save and stop: Save checkpoint then stop training
    - Status queries: Get current training status

    Only rank 0 runs the HTTP server. Commands are broadcast to all ranks
    via torch.distributed for coordination.
    """

    def __init__(self, job_id: str = None, port: int = None, enable_http: bool = None):
        """
        Initialize the control callback.

        Args:
            job_id: Unique identifier for this training job. Auto-generated if None.
            port: HTTP server port. Auto-selected if None.
            enable_http: Whether to enable HTTP server. Auto-detected based on aiohttp availability.
        """
        super().__init__()

        if enable_http is None:
            enable_http = AIOHTTP_AVAILABLE

        if enable_http and not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, disabling HTTP control server")
            enable_http = False

        self.enable_http = enable_http
        self.job_id = job_id or self._generate_job_id()
        self.port = port
        self.control_dir = Path.home() / ".forgather" / "jobs" / self.job_id

        # Command handling
        self.command_queue: Optional[asyncio.Queue] = None
        self.pending_commands: List[ControlCommand] = []

        # HTTP server
        self.server_task: Optional[asyncio.Task] = None
        self.server_runner: Optional[aiohttp.web.AppRunner] = None
        self.server_thread: Optional[threading.Thread] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None

        # State tracking
        self.trainer_args: Optional[MinimalTrainingArguments] = None
        self.trainer_state: Optional[TrainerState] = None
        self.last_status: Dict[str, Any] = {}

    def _generate_job_id(self) -> str:
        """Generate a unique job ID."""
        import platform

        timestamp = int(time.time())
        hostname = platform.node()
        pid = os.getpid()
        return f"job_{timestamp}_{hostname}_{pid}"

    def _find_available_port(
        self, start_port: int = 8900, max_attempts: int = 100
    ) -> int:
        """Find an available port for the HTTP server."""
        for i in range(max_attempts):
            port = start_port + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                continue
        raise RuntimeError(
            f"Could not find available port in range {start_port}-{start_port + max_attempts}"
        )

    def _write_endpoint_file(self, port: int):
        """Write endpoint information for service discovery."""
        import platform

        self.control_dir.mkdir(parents=True, exist_ok=True)
        endpoint_file = self.control_dir / "endpoint.json"

        endpoint_info = {
            "job_id": self.job_id,
            "host": platform.node(),
            "port": port,
            "pid": os.getpid(),
            "started_at": time.time(),
        }

        with open(endpoint_file, "w") as f:
            json.dump(endpoint_info, f, indent=2)

        logger.info(
            f"Trainer control endpoint: http://{endpoint_info['host']}:{port}/jobs/{self.job_id}"
        )

    def _setup_signal_handler(self):
        """Setup signal handler for lightweight command notification."""

        def signal_handler(signum, frame):
            if self.event_loop and not self.event_loop.is_closed():
                self.event_loop.call_soon_threadsafe(self._check_for_signals)

        signal.signal(signal.SIGUSR1, signal_handler)

    def _check_for_signals(self):
        """Check for signal-based commands (placeholder for future enhancement)."""
        pass

    def _get_device(self):
        """Get the correct device for tensor operations in distributed training."""
        if (
            self.trainer_args
            and hasattr(self.trainer_args, "device")
            and self.trainer_args.device is not None
        ):
            return self.trainer_args.device
        elif torch.cuda.is_available():
            return torch.cuda.current_device()
        else:
            return "cpu"

    async def _run_http_server(self):
        """Run the HTTP server in async mode."""
        try:
            self.command_queue = asyncio.Queue()

            app = aiohttp.web.Application()
            app.router.add_post(
                f"/jobs/{self.job_id}/control", self._handle_control_request
            )
            app.router.add_get(
                f"/jobs/{self.job_id}/status", self._handle_status_request
            )
            app.router.add_get("/jobs", self._handle_list_jobs)

            self.server_runner = aiohttp.web.AppRunner(app)
            await self.server_runner.setup()

            if self.port is None:
                self.port = self._find_available_port()

            site = aiohttp.web.TCPSite(self.server_runner, "0.0.0.0", self.port)
            await site.start()

            self._write_endpoint_file(self.port)
            logger.info(f"Trainer control server started on port {self.port}")

            # Keep server running
            while True:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("HTTP server shutting down")
            raise
        except Exception as e:
            logger.error(f"HTTP server error: {e}")
            raise

    def _run_server_thread(self):
        """Run HTTP server in separate thread."""
        asyncio.set_event_loop(self.event_loop)
        try:
            self.event_loop.run_until_complete(self._run_http_server())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Server thread error: {e}")

    async def _handle_control_request(
        self, request: aiohttp.web.Request
    ) -> aiohttp.web.Response:
        """Handle incoming control commands."""
        try:
            data = await request.json()
            command = data.get("command")

            if command not in COMMAND_CODES:
                return aiohttp.web.json_response(
                    {
                        "error": f"Unknown command: {command}",
                        "valid_commands": list(COMMAND_CODES.keys()),
                    },
                    status=400,
                )

            control_command = ControlCommand(
                command=command, timestamp=time.time(), data=data.get("data", {})
            )

            await self.command_queue.put(control_command)

            return aiohttp.web.json_response(
                {
                    "status": "acknowledged",
                    "command": command,
                    "timestamp": control_command.timestamp,
                    "message": f"Command {command} queued for execution",
                }
            )

        except Exception as e:
            logger.error(f"Error handling control request: {e}")
            return aiohttp.web.json_response({"error": str(e)}, status=500)

    async def _handle_status_request(
        self, request: aiohttp.web.Request
    ) -> aiohttp.web.Response:
        """Handle status requests."""
        try:
            status = {
                "job_id": self.job_id,
                "status": "running",
                "timestamp": time.time(),
                **self.last_status,
            }

            if self.trainer_state:
                status.update(
                    {
                        "global_step": self.trainer_state.global_step,
                        "epoch": self.trainer_state.epoch,
                        "max_steps": self.trainer_state.max_steps,
                    }
                )

            return aiohttp.web.json_response(status)

        except Exception as e:
            logger.error(f"Error handling status request: {e}")
            return aiohttp.web.json_response({"error": str(e)}, status=500)

    async def _handle_list_jobs(
        self, request: aiohttp.web.Request
    ) -> aiohttp.web.Response:
        """Handle job listing requests."""
        try:
            jobs_dir = Path.home() / ".forgather" / "jobs"
            jobs = []

            if jobs_dir.exists():
                for job_dir in jobs_dir.iterdir():
                    if job_dir.is_dir():
                        endpoint_file = job_dir / "endpoint.json"
                        if endpoint_file.exists():
                            try:
                                with open(endpoint_file) as f:
                                    job_info = json.load(f)
                                    jobs.append(job_info)
                            except Exception as e:
                                logger.warning(
                                    f"Could not read job info from {endpoint_file}: {e}"
                                )

            return aiohttp.web.json_response({"jobs": jobs})

        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            return aiohttp.web.json_response({"error": str(e)}, status=500)

    def _check_commands_non_blocking(self):
        """Check for new commands without blocking."""
        if not self.command_queue:
            return

        try:
            while True:
                try:
                    command = self.command_queue.get_nowait()
                    self.pending_commands.append(command)
                    logger.info(f"Received command: {command.command}")
                except asyncio.QueueEmpty:
                    break
        except Exception as e:
            logger.error(f"Error checking commands: {e}")

    def _pack_commands_for_broadcast(
        self, commands: List[ControlCommand]
    ) -> torch.Tensor:
        """Pack commands into tensor for broadcast."""
        if not commands:
            data = [0]
        else:
            # Pack: [num_commands, cmd1_code, cmd1_timestamp, cmd2_code, cmd2_timestamp, ...]
            data = [len(commands)]
            for cmd in commands:
                data.extend([COMMAND_CODES[cmd.command], int(cmd.timestamp)])

        # Create tensor on correct device for distributed communication
        device = self._get_device()
        return torch.tensor(data, dtype=torch.long, device=device)

    def _unpack_commands_from_broadcast(
        self, tensor: torch.Tensor
    ) -> List[ControlCommand]:
        """Unpack commands from broadcast tensor."""
        data = tensor.tolist()
        if not data or data[0] == 0:
            return []

        commands = []
        num_commands = data[0]
        idx = 1

        for _ in range(num_commands):
            command_code = data[idx]
            timestamp = data[idx + 1]
            idx += 2

            command = ControlCommand(
                command=COMMAND_NAMES[command_code], timestamp=float(timestamp)
            )
            commands.append(command)

        return commands

    def _broadcast_and_handle_commands(self, control: TrainerControl) -> TrainerControl:
        """Broadcast commands to all ranks and handle them."""
        commands_to_process = []

        if not torch.distributed.is_initialized():
            # Single process mode - just check for commands on rank 0
            if (
                self.trainer_state
                and self.trainer_state.is_world_process_zero
                and self.enable_http
            ):
                self._check_commands_non_blocking()
                commands_to_process = self.pending_commands.copy()
                self.pending_commands.clear()
        else:
            device = self._get_device()

            # Rank 0: Broadcast pending commands
            if self.trainer_state and self.trainer_state.is_world_process_zero:
                if self.enable_http:
                    self._check_commands_non_blocking()

                commands_tensor = self._pack_commands_for_broadcast(
                    self.pending_commands
                )

                # First broadcast the tensor size so other ranks can allocate correctly
                size_tensor = torch.tensor(
                    [commands_tensor.size(0)], dtype=torch.long, device=device
                )
                torch.distributed.broadcast(size_tensor, src=0)

                # Then broadcast the actual commands
                torch.distributed.broadcast(commands_tensor, src=0)

                commands_to_process = self.pending_commands.copy()
                self.pending_commands.clear()

            # All other ranks: Receive broadcast commands
            else:
                # First receive the size
                size_tensor = torch.tensor([0], dtype=torch.long, device=device)
                torch.distributed.broadcast(size_tensor, src=0)

                # Then receive the commands tensor with the correct size
                tensor_size = size_tensor.item()
                if tensor_size > 0:
                    commands_tensor = torch.zeros(
                        tensor_size, dtype=torch.long, device=device
                    )
                    torch.distributed.broadcast(commands_tensor, src=0)
                    commands_to_process = self._unpack_commands_from_broadcast(
                        commands_tensor
                    )

        # All ranks: Process commands
        for command in commands_to_process:
            control = self._apply_command(command, control)

        return control

    def _apply_command(
        self, command: ControlCommand, control: TrainerControl
    ) -> TrainerControl:
        """Apply a control command to the trainer control state."""
        logger.info(f"Applying command: {command.command}")

        if command.command == "graceful_stop":
            control.should_training_stop = True
            logger.info(
                "Graceful stop requested - training will stop after current step"
            )

        elif command.command == "save_checkpoint":
            control.should_save = True
            # If we're tracking best model, trigger evaluation first
            if (
                self.trainer_args
                and self.trainer_args.load_best_model_at_end
                and self.trainer_args.eval_strategy.value != "no"
            ):
                control.should_evaluate = True
            logger.info("Checkpoint save requested")

        elif command.command == "save_and_stop":
            control.should_save = True
            if (
                self.trainer_args
                and self.trainer_args.load_best_model_at_end
                and self.trainer_args.eval_strategy.value != "no"
            ):
                control.should_evaluate = True
            control.should_training_stop = True
            logger.info("Save and stop requested - will save checkpoint then stop")

        elif command.command == "abort":
            # Check if the TrainerControl supports the forgather extension
            if hasattr(control, "should_abort_without_save"):
                control.should_abort_without_save = True
                control.should_training_stop = True
                logger.info(
                    "Abort requested - training will stop WITHOUT saving checkpoint"
                )
            else:
                # Fallback for standard HF TrainerControl
                control.should_training_stop = True
                logger.warning(
                    "Abort requested but TrainerControl doesn't support abort_without_save - will stop gracefully"
                )

        return control

    # TrainerCallback interface methods

    def on_train_begin(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Initialize control system when training begins."""
        self.trainer_args = args
        self.trainer_state = state

        # Only rank 0 runs the HTTP server
        if state.is_world_process_zero and self.enable_http:
            try:
                self.event_loop = asyncio.new_event_loop()
                self.server_thread = threading.Thread(
                    target=self._run_server_thread, daemon=True
                )
                self.server_thread.start()

                # Give server time to start
                time.sleep(0.1)

                logger.info(f"Trainer control system initialized for job {self.job_id}")

            except Exception as e:
                logger.error(f"Failed to start control system: {e}")
                self.enable_http = False

    def on_log(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict = None,
        **kwargs,
    ):
        """Check for control commands on each log event."""
        self.trainer_state = state

        # Update status for queries
        if logs:
            self.last_status.update(logs)
            self.last_status["timestamp"] = time.time()

        # Check for and broadcast commands
        return self._broadcast_and_handle_commands(control)

    def on_train_end(
        self,
        args: MinimalTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Clean up when training ends."""
        if state.is_world_process_zero and self.enable_http:
            try:
                # Cancel server task
                if self.server_task and not self.server_task.done():
                    self.server_task.cancel()

                # Shutdown server
                if self.server_runner:
                    asyncio.run_coroutine_threadsafe(
                        self.server_runner.cleanup(), self.event_loop
                    ).result(timeout=5)

                # Stop event loop
                if self.event_loop and not self.event_loop.is_closed():
                    self.event_loop.call_soon_threadsafe(self.event_loop.stop)

                # Clean up endpoint file
                endpoint_file = self.control_dir / "endpoint.json"
                if endpoint_file.exists():
                    endpoint_file.unlink()

                logger.info("Trainer control system shutdown complete")

            except Exception as e:
                logger.warning(f"Error during control system shutdown: {e}")
