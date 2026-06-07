"""Dataset-metadata agent tools: list_dataset_servers + dataset_info.

All dataset-server discovery + HTTP is faked (monkeypatch); no network.
"""

from __future__ import annotations

import pytest

from forgather_server.agent import _dataset_servers, tools_jobs
from forgather_server.agent.registry import READ, ToolRegistry


def _entry(id="s1", base_url="http://h:8001", source="local", token=None, verify_tls=True):
    return {
        "id": id,
        "label": f"ds:{id}",
        "base_url": base_url,
        "source": source,
        "token": token,
        "verify_tls": verify_tls,
    }


class FakeClient:
    """Stands in for DatasetServerClient; per-instance scripted responses."""

    scripts: dict = {}  # base_url -> dict of method -> return/raise

    def __init__(self, url=None, token=None, insecure=False, timeout=30.0):
        self.url = url
        self._s = FakeClient.scripts.get(url, {})

    def _call(self, name):
        v = self._s.get(name)
        if isinstance(v, Exception):
            raise v
        if v is None:
            raise RuntimeError(f"no script for {name}")
        return v

    def health(self):
        return self._call("health")

    def list_hf_cache(self):
        return self._call("list_hf_cache")

    def list_local(self):
        return self._call("list_local")

    def load(self, load_args):
        return self._call("load")


@pytest.fixture(autouse=True)
def _patch_client(monkeypatch):
    monkeypatch.setattr(_dataset_servers, "DatasetServerClient", FakeClient)
    FakeClient.scripts = {}
    yield
    FakeClient.scripts = {}


def test_tools_registered():
    reg = ToolRegistry()
    tools_jobs.register_all(reg)
    by_name = {s.name: s for s in reg.specs()}
    assert {"list_dataset_servers", "dataset_info"} <= set(by_name)
    assert by_name["list_dataset_servers"].risk == READ
    assert by_name["dataset_info"].risk == READ


def test_list_dataset_servers_reports_reachability(monkeypatch):
    monkeypatch.setattr(
        _dataset_servers,
        "_discover",
        lambda: [_entry(id="up", base_url="http://up:1"), _entry(id="down", base_url="http://down:2")],
    )
    FakeClient.scripts = {
        "http://up:1": {"health": {"ok": True}},
        "http://down:2": {"health": ConnectionError("nope")},
    }
    out = tools_jobs._list_dataset_servers({})
    by_id = {s["id"]: s for s in out["servers"]}
    assert by_id["up"]["reachable"] is True
    assert by_id["down"]["reachable"] is False
    # Tokens never leak to the model.
    for s in out["servers"]:
        assert "token" not in s


def test_dataset_info_from_hf_cache(monkeypatch):
    monkeypatch.setattr(_dataset_servers, "_discover", lambda: [_entry(base_url="http://h:1")])
    FakeClient.scripts = {
        "http://h:1": {
            "health": {"ok": True},
            "list_hf_cache": {
                "datasets": [
                    {
                        "repo": "roneneldan/TinyStories",
                        "configs": [
                            {"config": "default", "splits": [
                                {"name": "train", "num_examples": 2000},
                                {"name": "validation", "num_examples": 100},
                            ], "features": ["text"]}
                        ],
                    }
                ]
            },
            "list_local": {"local": []},
        }
    }
    out = tools_jobs._dataset_info({"dataset": "roneneldan/TinyStories"})
    assert out["source"] == "hf_cache"
    assert {s["name"]: s["num_examples"] for s in out["splits"]} == {
        "train": 2000,
        "validation": 100,
    }
    assert out["features"] == ["text"]
    assert out["server"]["base_url"] == "http://h:1"


def test_dataset_info_features_fallback_load(monkeypatch):
    monkeypatch.setattr(_dataset_servers, "_discover", lambda: [_entry(base_url="http://h:1")])
    FakeClient.scripts = {
        "http://h:1": {
            "health": {"ok": True},
            # hf cache has splits but no features.
            "list_hf_cache": {
                "datasets": [
                    {"repo": "foo/bar", "configs": [
                        {"config": "default", "splits": [{"name": "train", "num_examples": 5}]}
                    ]}
                ]
            },
            "list_local": {"local": []},
            "load": {"handle": "h", "length": 5, "column_names": ["text", "label"]},
        }
    }
    out = tools_jobs._dataset_info({"dataset": "foo/bar"})
    # Features came from the load fallback; source stays where splits were found.
    assert out["features"] == ["text", "label"]
    assert out["source"] == "hf_cache"


def test_dataset_info_no_server(monkeypatch):
    monkeypatch.setattr(_dataset_servers, "_discover", lambda: [])
    with pytest.raises(ValueError, match="No dataset server is reachable"):
        tools_jobs._dataset_info({"dataset": "foo/bar"})


def test_dataset_info_picks_by_id(monkeypatch):
    monkeypatch.setattr(
        _dataset_servers,
        "_discover",
        lambda: [_entry(id="a", base_url="http://a:1"), _entry(id="b", base_url="http://b:2")],
    )
    FakeClient.scripts = {
        "http://a:1": {"health": {"ok": True}},
        "http://b:2": {
            "health": {"ok": True},
            "list_hf_cache": {"datasets": [
                {"repo": "d", "configs": [{"config": "c", "splits": [{"name": "train", "num_examples": 1}], "features": ["x"]}]}
            ]},
            "list_local": {"local": []},
        },
    }
    out = tools_jobs._dataset_info({"dataset": "d", "server_id": "b"})
    assert out["server"]["id"] == "b"


def test_dataset_info_requires_dataset():
    with pytest.raises(ValueError, match="dataset is required"):
        tools_jobs._dataset_info({})
