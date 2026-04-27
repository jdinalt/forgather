"""CRUD endpoints for the persistent project-discovery roots."""

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import search_roots as sr

router = APIRouter(tags=["search-roots"])


class SearchRootModel(BaseModel):
    path: str
    exists: bool


class AddSearchRootRequest(BaseModel):
    path: str
    # When true, ``mkdir -p`` the target before registering it. Used by
    # the New Workspace modal's "create new search root" flow so the
    # caller can supply ``parent + name`` and have the directory created
    # atomically with the registration.
    create: bool = False


@router.get("/search-roots", response_model=list[SearchRootModel])
def get_search_roots():
    return [SearchRootModel(path=r.path, exists=r.exists) for r in sr.list_roots()]


@router.post("/search-roots", response_model=SearchRootModel)
def post_search_root(req: AddSearchRootRequest):
    target = os.path.abspath(os.path.expanduser(req.path))
    if req.create and not os.path.exists(target):
        try:
            os.makedirs(target)
        except OSError as e:
            raise HTTPException(status_code=400, detail=f"mkdir failed: {e}")
    root = sr.add_root(target)
    return SearchRootModel(path=root.path, exists=root.exists)


@router.delete("/search-roots")
def delete_search_root(path: str):
    removed = sr.remove_root(path)
    if not removed:
        raise HTTPException(status_code=404, detail="Path not in search roots")
    return {"removed": path}
