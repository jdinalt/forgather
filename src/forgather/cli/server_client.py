"""HTTP/WebSocket client for the forgather-server API."""

import json
import os


class ServerUnreachable(Exception):
    pass


class ServerClient:
    def __init__(self, base_url=None, timeout=30.0):
        import requests

        base = (
            base_url
            or os.environ.get("FORGATHER_SERVER_URL")
            or "http://127.0.0.1:8765"
        )
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "forgather-cli"

    @classmethod
    def from_args(cls, args):
        return cls(getattr(args, "server", None))

    def _url(self, path):
        return f"{self.base}/api{path}"

    def _ws_url(self, path):
        url = self._url(path)
        if url.startswith("https://"):
            return "wss://" + url[len("https://") :]
        return "ws://" + url[len("http://") :]

    def _get(self, path, **kwargs):
        import requests

        try:
            r = self.session.get(self._url(path), timeout=self.timeout, **kwargs)
        except (requests.ConnectionError, requests.Timeout):
            raise ServerUnreachable(
                f"could not reach forgather-server at {self.base}; is it running? (start with: forgather server)"
            )
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RuntimeError(f"server: {detail}")
        return r

    def _post(self, path, body=None):
        import requests

        try:
            r = self.session.post(self._url(path), json=body, timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout):
            raise ServerUnreachable(
                f"could not reach forgather-server at {self.base}; is it running? (start with: forgather server)"
            )
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RuntimeError(f"server: {detail}")
        return r

    def _delete(self, path):
        import requests

        try:
            r = self.session.delete(self._url(path), timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout):
            raise ServerUnreachable(
                f"could not reach forgather-server at {self.base}; is it running? (start with: forgather server)"
            )
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RuntimeError(f"server: {detail}")
        return r

    # Queue

    def enqueue_job(
        self,
        *,
        project_dir,
        config,
        job_type,
        job_params,
        requested_gpus=0,
        priority=0,
        dynamic_args=None,
    ):
        body = {
            "project_dir": project_dir,
            "config": config,
            "dynamic_args": dynamic_args or {},
            "requested_gpus": requested_gpus,
            "priority": priority,
            "job_type": job_type,
            "job_params": job_params,
        }
        return self._post("/queue", body).json()

    def enqueue_training(
        self, project_dir, config, *, dynamic_args, priority, requested_gpus
    ):
        body = {
            "project_dir": project_dir,
            "config": config,
            "dynamic_args": dynamic_args,
            "requested_gpus": requested_gpus,
            "priority": priority,
            "job_type": "training",
            "job_params": {},
        }
        return self._post("/queue", body).json()

    def list_queue(self):
        return self._get("/queue").json()

    def cancel(self, queue_id):
        return self._delete(f"/queue/{queue_id}").json()

    def get_scheduler(self):
        return self._get("/queue/scheduler").json()

    def set_scheduler(self, enabled):
        return self._post("/queue/scheduler", {"enabled": enabled}).json()

    # Jobs

    def list_jobs(self, include_dead=False):
        params = {"include_dead_endpoints": "true" if include_dead else "false"}
        return self._get("/jobs", params=params).json()

    def job_status(self, job_id):
        import requests

        try:
            r = self.session.get(
                self._url(f"/jobs/{job_id}/status"), timeout=self.timeout
            )
        except (requests.ConnectionError, requests.Timeout):
            raise ServerUnreachable(
                f"could not reach forgather-server at {self.base}; is it running? (start with: forgather server)"
            )
        return r

    def job_control(self, job_id, action):
        return self._post(f"/jobs/{job_id}/control/{action}").json()

    def job_dump(self, job_id):
        return self._get(f"/jobs/{job_id}/tty").content

    def job_delete(self, job_id):
        return self._delete(f"/jobs/{job_id}").json()

    def cleanup_jobs(self):
        return self._post("/jobs/cleanup").json()

    def gc_jobs(self):
        return self._post("/jobs/gc").json()

    async def stream_tty(self, job_id, follow=True):
        import websockets
        import websockets.exceptions

        ws_url = (
            self._ws_url(f"/jobs/{job_id}/tty")
            + f"?follow={'true' if follow else 'false'}"
        )
        try:
            ws = await websockets.connect(ws_url)
        except OSError:
            raise ServerUnreachable(
                f"could not reach forgather-server at {self.base}; is it running? (start with: forgather server)"
            )
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    yield ("bytes", message)
                else:
                    try:
                        frame = json.loads(message)
                        if isinstance(frame, dict) and frame.get("type") == "error":
                            yield ("error", frame.get("detail", message))
                        else:
                            yield ("bytes", message.encode())
                    except Exception:
                        yield ("bytes", message.encode())
        except websockets.exceptions.ConnectionClosed:
            return
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    # GPUs

    def list_gpus(self):
        return self._get("/gpus").json()

    def gpu_policy_all(self):
        return self._get("/gpus/policy").json()

    def set_gpu_policy(self, idx, *, disabled=None, min_priority=None):
        body = {}
        if disabled is not None:
            body["disabled"] = disabled
        if min_priority is not None:
            body["min_priority"] = min_priority
        return self._post(f"/gpus/{idx}/policy", body).json()

    def kill_gpu_processes(self, idx):
        return self._post(f"/gpus/{idx}/kill", {"confirmed": True}).json()
