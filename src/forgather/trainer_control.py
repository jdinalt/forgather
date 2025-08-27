"""
Client API for controlling remote training jobs.

Provides a clean interface for sending control commands to running distributed training jobs
via HTTP API or other communication mechanisms.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class JobInfo:
    """Information about a training job."""

    job_id: str
    host: str
    port: int
    pid: int
    started_at: float

    @property
    def endpoint_url(self) -> str:
        """Get the base URL for this job's control API."""
        return f"http://{self.host}:{self.port}/jobs/{self.job_id}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobInfo":
        """Create JobInfo from dictionary."""
        return cls(
            job_id=data["job_id"],
            host=data["host"],
            port=data["port"],
            pid=data["pid"],
            started_at=data["started_at"],
        )


@dataclass
class ControlResponse:
    """Response from a control command."""

    success: bool
    message: str
    data: Dict[str, Any] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class TrainerControlClient(ABC):
    """Abstract base class for trainer control clients."""

    @abstractmethod
    def send_command(
        self, job_id: str, command: str, data: Dict[str, Any] = None
    ) -> ControlResponse:
        """Send a control command to a training job."""
        pass

    @abstractmethod
    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a training job."""
        pass

    @abstractmethod
    def list_jobs(self) -> List[JobInfo]:
        """List all discoverable training jobs."""
        pass

    def graceful_stop(self, job_id: str) -> ControlResponse:
        """Send graceful stop command to a training job."""
        return self.send_command(job_id, "graceful_stop")

    def save_checkpoint(self, job_id: str) -> ControlResponse:
        """Request checkpoint save from a training job."""
        return self.send_command(job_id, "save_checkpoint")

    def save_and_stop(self, job_id: str) -> ControlResponse:
        """Request checkpoint save and graceful stop from a training job."""
        return self.send_command(job_id, "save_and_stop")

    def abort(self, job_id: str) -> ControlResponse:
        """Abort training job without saving."""
        return self.send_command(job_id, "abort")


class HTTPTrainerControlClient(TrainerControlClient):
    """HTTP-based trainer control client."""

    def __init__(self, timeout: float = 30.0):
        """
        Initialize HTTP client.

        Args:
            timeout: HTTP request timeout in seconds.
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required for HTTPTrainerControlClient")

        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "forgather-trainer-control",
            }
        )

    def _get_job_info(self, job_id: str) -> JobInfo:
        """Get job information from discovery files."""
        jobs_dir = Path.home() / ".forgather" / "jobs"
        job_dir = jobs_dir / job_id
        endpoint_file = job_dir / "endpoint.json"

        if not endpoint_file.exists():
            raise ValueError(f"No endpoint information found for job {job_id}")

        try:
            with open(endpoint_file) as f:
                data = json.load(f)
            return JobInfo.from_dict(data)
        except Exception as e:
            raise ValueError(f"Failed to read job endpoint info: {e}")

    def send_command(
        self, job_id: str, command: str, data: Dict[str, Any] = None
    ) -> ControlResponse:
        """Send a control command via HTTP."""
        try:
            job_info = self._get_job_info(job_id)
            url = f"{job_info.endpoint_url}/control"

            payload = {"command": command, "data": data or {}}

            response = self.session.post(url, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                result = response.json()
                return ControlResponse(
                    success=True,
                    message=result.get("message", "Command sent successfully"),
                    data=result,
                    timestamp=result.get("timestamp"),
                )
            else:
                error_data = {}
                try:
                    error_data = response.json()
                except:
                    error_data = {"error": response.text}

                return ControlResponse(
                    success=False,
                    message=f"HTTP {response.status_code}: {error_data.get('error', 'Unknown error')}",
                    data=error_data,
                )

        except Exception as e:
            logger.error(f"Failed to send command {command} to job {job_id}: {e}")
            return ControlResponse(
                success=False, message=f"Failed to send command: {e}"
            )

    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status via HTTP."""
        try:
            job_info = self._get_job_info(job_id)
            url = f"{job_info.endpoint_url}/status"

            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Failed to get status for job {job_id}: {e}")
            return {"error": str(e), "job_id": job_id, "status": "unknown"}

    def list_jobs(self) -> List[JobInfo]:
        """List jobs by scanning endpoint files, sorted by start time (newest first)."""
        jobs = []
        jobs_dir = Path.home() / ".forgather" / "jobs"

        if not jobs_dir.exists():
            return jobs

        for job_dir in jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue

            endpoint_file = job_dir / "endpoint.json"
            if not endpoint_file.exists():
                continue

            try:
                with open(endpoint_file) as f:
                    data = json.load(f)
                jobs.append(JobInfo.from_dict(data))
            except Exception as e:
                logger.warning(f"Could not read job info from {endpoint_file}: {e}")

        # Sort by start time, newest first
        jobs.sort(key=lambda job: job.started_at, reverse=True)
        return jobs

    def list_jobs_remote(self, host: str, port: int) -> List[JobInfo]:
        """List jobs by querying a remote trainer's HTTP API."""
        try:
            url = f"http://{host}:{port}/jobs"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            return [JobInfo.from_dict(job_data) for job_data in data.get("jobs", [])]

        except Exception as e:
            logger.error(f"Failed to list remote jobs from {host}:{port}: {e}")
            return []


class FileBasedTrainerControlClient(TrainerControlClient):
    """File-based trainer control client for single-node scenarios."""

    def __init__(self):
        """Initialize file-based client."""
        self.control_dir = Path.home() / ".forgather" / "jobs"

    def send_command(
        self, job_id: str, command: str, data: Dict[str, Any] = None
    ) -> ControlResponse:
        """Send command via file system."""
        try:
            job_dir = self.control_dir / job_id
            if not job_dir.exists():
                return ControlResponse(success=False, message=f"Job {job_id} not found")

            command_file = (
                job_dir / "control" / f"{command}_{int(time.time()*1000)}.json"
            )
            command_file.parent.mkdir(exist_ok=True)

            command_data = {
                "command": command,
                "timestamp": time.time(),
                "data": data or {},
            }

            with open(command_file, "w") as f:
                json.dump(command_data, f, indent=2)

            return ControlResponse(
                success=True, message=f"Command {command} written to {command_file}"
            )

        except Exception as e:
            return ControlResponse(
                success=False, message=f"Failed to send command: {e}"
            )

    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get status from file system."""
        try:
            job_dir = self.control_dir / job_id
            status_file = job_dir / "status.json"

            if status_file.exists():
                with open(status_file) as f:
                    return json.load(f)
            else:
                return {
                    "job_id": job_id,
                    "status": "unknown",
                    "error": "No status file found",
                }

        except Exception as e:
            return {"job_id": job_id, "status": "error", "error": str(e)}

    def list_jobs(self) -> List[JobInfo]:
        """List jobs from file system."""
        jobs = []

        if not self.control_dir.exists():
            return jobs

        for job_dir in self.control_dir.iterdir():
            if not job_dir.is_dir():
                continue

            endpoint_file = job_dir / "endpoint.json"
            if endpoint_file.exists():
                try:
                    with open(endpoint_file) as f:
                        data = json.load(f)
                    jobs.append(JobInfo.from_dict(data))
                except Exception as e:
                    logger.warning(f"Could not read job info from {endpoint_file}: {e}")

        return jobs


def get_default_client() -> TrainerControlClient:
    """Get the default trainer control client based on available dependencies."""
    if REQUESTS_AVAILABLE:
        return HTTPTrainerControlClient()
    else:
        logger.warning("requests not available, falling back to file-based control")
        return FileBasedTrainerControlClient()


# Convenience functions for common operations


def list_jobs() -> List[JobInfo]:
    """List all discoverable training jobs."""
    client = get_default_client()
    return client.list_jobs()


def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get the status of a specific training job."""
    client = get_default_client()
    return client.get_status(job_id)


def graceful_stop(job_id: str) -> ControlResponse:
    """Send graceful stop command to a training job."""
    client = get_default_client()
    return client.graceful_stop(job_id)


def save_checkpoint(job_id: str) -> ControlResponse:
    """Request checkpoint save from a training job."""
    client = get_default_client()
    return client.save_checkpoint(job_id)


def save_and_stop(job_id: str) -> ControlResponse:
    """Request checkpoint save and graceful stop from a training job."""
    client = get_default_client()
    return client.save_and_stop(job_id)


def abort(job_id: str) -> ControlResponse:
    """Abort training job without saving."""
    client = get_default_client()
    return client.abort(job_id)
