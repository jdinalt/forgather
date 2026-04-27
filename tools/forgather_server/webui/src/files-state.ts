import { useCallback, useRef, useState } from "react";

import { api, SaveConflictError } from "./api";

/** One open file. Buffers are keyed by absolute path; the same buffer is
 *  shown in every split that has the file open, so editing in one place
 *  reflects in the other. */
export interface FileBuffer {
  path: string;
  loading: boolean;
  error: string | null;
  /** Last-saved content (or last-loaded content for read-only checks). */
  baseline: string;
  /** Current editor content; differs from baseline when dirty. */
  content: string;
  saving: boolean;
  /** mtime of the on-disk file at the moment we last loaded or saved
   *  it. Sent as ``expected_mtime`` on the next save; if the on-disk
   *  mtime is newer at that point, the server returns 409 and we
   *  prompt the user instead of clobbering. ``undefined`` means we
   *  haven't seen a mtime yet (load failed) — saves still work, the
   *  conflict check is just skipped. */
  baselineMtime?: number;
  /** Set when the most recent save attempt collided with an on-disk
   *  change. The editor renders a modal so the user can pick:
   *  Overwrite (force-save), Reload (discard local edits and re-fetch),
   *  or Cancel (dismiss; buffer stays dirty). */
  conflict?: { currentMtime: number };
}

/** A single editor split (column). Tabs are ordered; ``activePath`` may be
 *  null only when ``tabPaths`` is empty. */
export interface EditorSplit {
  id: string;
  tabPaths: string[];
  activePath: string | null;
}

export interface FilesState {
  buffers: Record<string, FileBuffer>;
  splits: EditorSplit[];
  activeSplitId: string;
}

export interface FilesApi {
  state: FilesState;
  /** Open a file in the active split. If already open in any split, focus
   *  the existing tab in that split instead of duplicating. */
  openFile: (path: string) => void;
  setContent: (path: string, content: string) => void;
  saveFile: (path: string) => Promise<void>;
  closeTab: (splitId: string, path: string) => void;
  closeOthers: (splitId: string, path: string) => void;
  closeAll: (splitId: string) => void;
  setActiveTab: (splitId: string, path: string) => void;
  setActiveSplit: (splitId: string) => void;
  splitVertical: (splitId: string) => void;
  /** Move ``path`` from ``fromSplitId`` to ``toSplitId`` (or reorder within a
   *  split). ``beforePath`` is the tab to insert before; null = end. */
  moveTab: (
    fromSplitId: string,
    toSplitId: string,
    path: string,
    beforePath: string | null,
  ) => void;
  isDirty: (path: string) => boolean;
  /** Drop ``path`` from every split *and* discard its buffer, without
   *  the dirty-prompt that ``closeTab`` runs. Used by external file
   *  operations (rename / move / delete from the Files tree) after the
   *  on-disk file has already changed — prompting the user about
   *  "unsaved changes" on a path that no longer exists would be
   *  confusing. The user is responsible for saving before invoking the
   *  operation; if they didn't, the buffer is silently discarded. */
  dropPath: (path: string) => void;
  /** Save without the optimistic-mtime check — used when the user
   *  has explicitly chosen "Overwrite with my version" in the
   *  conflict modal. */
  forceSaveFile: (path: string) => Promise<void>;
  /** Discard the buffer's local edits and re-fetch from disk. Used
   *  for the conflict modal's "Reload from disk" choice. */
  reloadFile: (path: string) => Promise<void>;
  /** Dismiss the conflict modal without an action — buffer stays
   *  dirty, baselineMtime stays at the old value. The user can
   *  retry save and will hit the same 409 unless they Overwrite or
   *  Reload first. */
  clearConflict: (path: string) => void;
}

let _counter = 0;
function genId() {
  _counter += 1;
  return `split-${Date.now().toString(36)}-${_counter}`;
}

function emptyState(): FilesState {
  const initial: EditorSplit = {
    id: genId(),
    tabPaths: [],
    activePath: null,
  };
  return {
    buffers: {},
    splits: [initial],
    activeSplitId: initial.id,
  };
}

export function useFilesState(): FilesApi {
  const [state, setState] = useState<FilesState>(emptyState);
  // Snapshot of latest state, used by async callbacks (load / save) to read
  // current values without needing to be re-bound on every state change.
  const stateRef = useRef(state);
  stateRef.current = state;

  const isDirty = useCallback(
    (path: string) => {
      const buf = state.buffers[path];
      return !!buf && !buf.loading && buf.content !== buf.baseline;
    },
    [state.buffers],
  );

  const setActiveSplit = useCallback((splitId: string) => {
    setState((s) => {
      if (s.activeSplitId === splitId) return s;
      if (!s.splits.some((sp) => sp.id === splitId)) return s;
      return { ...s, activeSplitId: splitId };
    });
  }, []);

  const setActiveTab = useCallback((splitId: string, path: string) => {
    setState((s) => ({
      ...s,
      activeSplitId: splitId,
      splits: s.splits.map((sp) =>
        sp.id === splitId && sp.tabPaths.includes(path)
          ? { ...sp, activePath: path }
          : sp,
      ),
    }));
  }, []);

  const openFile = useCallback((path: string) => {
    // Decide whether we'll need to fetch *before* calling setState, since
    // React 18 doesn't run the updater synchronously — checking a closure
    // flag mutated by the updater would always observe the stale value and
    // the fetch would never fire.
    const needsLoad = !stateRef.current.buffers[path];
    setState((s) => {
      // If the file is already open in some split, focus it there instead of
      // adding a duplicate tab. Prefer the active split, then any split.
      const active = s.splits.find((sp) => sp.id === s.activeSplitId);
      const owner =
        active && active.tabPaths.includes(path)
          ? active
          : s.splits.find((sp) => sp.tabPaths.includes(path));
      if (owner) {
        return {
          ...s,
          activeSplitId: owner.id,
          splits: s.splits.map((sp) =>
            sp.id === owner.id ? { ...sp, activePath: path } : sp,
          ),
        };
      }
      // New tab in the active split.
      const buf = s.buffers[path];
      const buffers = buf
        ? s.buffers
        : {
            ...s.buffers,
            [path]: {
              path,
              loading: true,
              error: null,
              baseline: "",
              content: "",
              saving: false,
            },
          };
      const splits = s.splits.map((sp) =>
        sp.id === s.activeSplitId
          ? { ...sp, tabPaths: [...sp.tabPaths, path], activePath: path }
          : sp,
      );
      return { ...s, buffers, splits };
    });

    if (needsLoad) {
      api
        .templateSourceWithMeta(path)
        .then(({ content, mtime }) => {
          setState((s) => {
            const buf = s.buffers[path];
            if (!buf) return s;
            return {
              ...s,
              buffers: {
                ...s.buffers,
                [path]: {
                  ...buf,
                  loading: false,
                  baseline: content,
                  content,
                  baselineMtime: mtime,
                  error: null,
                },
              },
            };
          });
        })
        .catch((err) => {
          setState((s) => {
            const buf = s.buffers[path];
            if (!buf) return s;
            return {
              ...s,
              buffers: {
                ...s.buffers,
                [path]: { ...buf, loading: false, error: String(err) },
              },
            };
          });
        });
    }
  }, []);

  const setContent = useCallback((path: string, content: string) => {
    setState((s) => {
      const buf = s.buffers[path];
      if (!buf) return s;
      return {
        ...s,
        buffers: { ...s.buffers, [path]: { ...buf, content } },
      };
    });
  }, []);

  /** Shared body of saveFile / forceSaveFile. ``force`` skips sending
   *  ``expected_mtime`` so the server doesn't run the conflict check. */
  const doSave = async (path: string, force: boolean): Promise<void> => {
    const cur = stateRef.current.buffers[path];
    if (!cur || cur.loading) return;
    if (cur.content === cur.baseline) return; // nothing to save
    setState((s) => {
      const buf = s.buffers[path];
      if (!buf) return s;
      return {
        ...s,
        buffers: {
          ...s.buffers,
          [path]: { ...buf, saving: true, error: null, conflict: undefined },
        },
      };
    });
    try {
      const sentContent = cur.content;
      const r = await api.putTemplateSource(
        path,
        sentContent,
        force ? undefined : cur.baselineMtime,
      );
      setState((s) => {
        const buf = s.buffers[path];
        if (!buf) return s;
        return {
          ...s,
          buffers: {
            ...s.buffers,
            [path]: {
              ...buf,
              saving: false,
              baseline: sentContent,
              baselineMtime: r.mtime,
              conflict: undefined,
            },
          },
        };
      });
    } catch (err) {
      if (err instanceof SaveConflictError) {
        // Don't surface the 409 as a fatal error in the buffer — stash
        // the conflict info so the UI can prompt; baseline + content
        // stay where they were so the user keeps their edits.
        const e = err;
        setState((s) => {
          const buf = s.buffers[path];
          if (!buf) return s;
          return {
            ...s,
            buffers: {
              ...s.buffers,
              [path]: {
                ...buf,
                saving: false,
                conflict: { currentMtime: e.currentMtime },
              },
            },
          };
        });
        return; // don't rethrow — the UI handles the prompt
      }
      setState((s) => {
        const buf = s.buffers[path];
        if (!buf) return s;
        return {
          ...s,
          buffers: {
            ...s.buffers,
            [path]: { ...buf, saving: false, error: String(err) },
          },
        };
      });
      throw err;
    }
  };

  const saveFile = useCallback(
    (path: string) => doSave(path, /*force=*/ false),
    [],
  );
  const forceSaveFile = useCallback(
    (path: string) => doSave(path, /*force=*/ true),
    [],
  );

  const reloadFile = useCallback(async (path: string) => {
    const cur = stateRef.current.buffers[path];
    if (!cur) return;
    setState((s) => {
      const buf = s.buffers[path];
      if (!buf) return s;
      return {
        ...s,
        buffers: {
          ...s.buffers,
          [path]: { ...buf, loading: true, error: null, conflict: undefined },
        },
      };
    });
    try {
      const { content, mtime } = await api.templateSourceWithMeta(path);
      setState((s) => {
        const buf = s.buffers[path];
        if (!buf) return s;
        return {
          ...s,
          buffers: {
            ...s.buffers,
            [path]: {
              ...buf,
              loading: false,
              baseline: content,
              content, // discard local edits — that's the user's choice here
              baselineMtime: mtime,
              error: null,
              conflict: undefined,
            },
          },
        };
      });
    } catch (err) {
      setState((s) => {
        const buf = s.buffers[path];
        if (!buf) return s;
        return {
          ...s,
          buffers: {
            ...s.buffers,
            [path]: { ...buf, loading: false, error: String(err) },
          },
        };
      });
    }
  }, []);

  const clearConflict = useCallback((path: string) => {
    setState((s) => {
      const buf = s.buffers[path];
      if (!buf) return s;
      return {
        ...s,
        buffers: { ...s.buffers, [path]: { ...buf, conflict: undefined } },
      };
    });
  }, []);

  // Helper that drops a tab from one split. If the file is no longer open
  // anywhere afterwards, we drop the buffer too so memory + the cross-split
  // shared-content invariant stay consistent.
  const dropTab = (s: FilesState, splitId: string, path: string): FilesState => {
    const splits = s.splits.map((sp) => {
      if (sp.id !== splitId) return sp;
      const tabPaths = sp.tabPaths.filter((p) => p !== path);
      let activePath = sp.activePath;
      if (activePath === path) {
        const idx = sp.tabPaths.indexOf(path);
        activePath = tabPaths[idx] ?? tabPaths[idx - 1] ?? tabPaths[0] ?? null;
      }
      return { ...sp, tabPaths, activePath };
    });
    const stillOpen = splits.some((sp) => sp.tabPaths.includes(path));
    let buffers = s.buffers;
    if (!stillOpen && buffers[path]) {
      buffers = { ...buffers };
      delete buffers[path];
    }
    return { ...s, splits, buffers };
  };

  const dropPath = useCallback((path: string) => {
    setState((s) => {
      let next = s;
      for (const sp of s.splits) {
        if (sp.tabPaths.includes(path)) {
          next = dropTab(next, sp.id, path);
        }
      }
      // Collapse any newly-emptied non-last split, matching closeTab's
      // tidy-up rule.
      while (next.splits.length > 1) {
        const empty = next.splits.find((sp) => sp.tabPaths.length === 0);
        if (!empty) break;
        const splits = next.splits.filter((sp) => sp.id !== empty.id);
        const activeSplitId =
          next.activeSplitId === empty.id ? splits[0].id : next.activeSplitId;
        next = { ...next, splits, activeSplitId };
      }
      return next;
    });
  }, []);

  const closeTab = useCallback((splitId: string, path: string) => {
    const cur = stateRef.current.buffers[path];
    if (cur && cur.content !== cur.baseline) {
      const ok = window.confirm(
        `${path}\n\nThis file has unsaved changes. Close anyway?`,
      );
      if (!ok) return;
    }
    setState((s) => {
      let next = dropTab(s, splitId, path);
      // If a split has zero tabs and there are other splits, drop the empty
      // one to keep the layout tidy. Always keep at least one split.
      if (next.splits.length > 1) {
        const empty = next.splits.find(
          (sp) => sp.id === splitId && sp.tabPaths.length === 0,
        );
        if (empty) {
          const splits = next.splits.filter((sp) => sp.id !== splitId);
          const activeSplitId =
            next.activeSplitId === splitId ? splits[0].id : next.activeSplitId;
          next = { ...next, splits, activeSplitId };
        }
      }
      return next;
    });
  }, []);

  const closeOthers = useCallback((splitId: string, path: string) => {
    // Only the named tab survives in this split. Buffers for the other paths
    // may stay alive if they're open in another split — dropTab handles that.
    setState((s) => {
      const target = s.splits.find((sp) => sp.id === splitId);
      if (!target) return s;
      const toClose = target.tabPaths.filter((p) => p !== path);
      // Confirm if any tab being closed is dirty.
      const dirtyClose = toClose.find((p) => {
        const b = s.buffers[p];
        return b && b.content !== b.baseline;
      });
      if (dirtyClose) {
        const ok = window.confirm(
          `Some tabs have unsaved changes (e.g. ${dirtyClose}). Close them anyway?`,
        );
        if (!ok) return s;
      }
      let next: FilesState = s;
      for (const p of toClose) next = dropTab(next, splitId, p);
      return next;
    });
  }, []);

  const closeAll = useCallback((splitId: string) => {
    setState((s) => {
      const target = s.splits.find((sp) => sp.id === splitId);
      if (!target) return s;
      const dirtyClose = target.tabPaths.find((p) => {
        const b = s.buffers[p];
        return b && b.content !== b.baseline;
      });
      if (dirtyClose) {
        const ok = window.confirm(
          `Some tabs have unsaved changes (e.g. ${dirtyClose}). Close them anyway?`,
        );
        if (!ok) return s;
      }
      let next: FilesState = s;
      for (const p of [...target.tabPaths]) next = dropTab(next, splitId, p);
      // Collapse newly-empty split if there's another to fall back on.
      if (next.splits.length > 1) {
        const splits = next.splits.filter((sp) => sp.id !== splitId);
        const activeSplitId =
          next.activeSplitId === splitId ? splits[0].id : next.activeSplitId;
        next = { ...next, splits, activeSplitId };
      }
      return next;
    });
  }, []);

  const splitVertical = useCallback((splitId: string) => {
    setState((s) => {
      const idx = s.splits.findIndex((sp) => sp.id === splitId);
      if (idx < 0) return s;
      const src = s.splits[idx];
      const newSplit: EditorSplit = {
        id: genId(),
        tabPaths: src.activePath ? [src.activePath] : [],
        activePath: src.activePath,
      };
      const splits = [...s.splits];
      splits.splice(idx + 1, 0, newSplit);
      return { ...s, splits, activeSplitId: newSplit.id };
    });
  }, []);

  const moveTab = useCallback(
    (
      fromSplitId: string,
      toSplitId: string,
      path: string,
      beforePath: string | null,
    ) => {
      setState((s) => {
        const from = s.splits.find((sp) => sp.id === fromSplitId);
        if (!from || !from.tabPaths.includes(path)) return s;

        // Within-split reorder.
        if (fromSplitId === toSplitId) {
          const without = from.tabPaths.filter((p) => p !== path);
          const insertAt =
            beforePath == null ? without.length : without.indexOf(beforePath);
          const at = insertAt < 0 ? without.length : insertAt;
          const reordered = [...without];
          reordered.splice(at, 0, path);
          return {
            ...s,
            splits: s.splits.map((sp) =>
              sp.id === fromSplitId
                ? { ...sp, tabPaths: reordered, activePath: path }
                : sp,
            ),
            activeSplitId: fromSplitId,
          };
        }

        // Cross-split move.
        let splits = s.splits.map((sp) => {
          if (sp.id === fromSplitId) {
            const tabPaths = sp.tabPaths.filter((p) => p !== path);
            let activePath = sp.activePath;
            if (activePath === path) {
              const idx = sp.tabPaths.indexOf(path);
              activePath =
                tabPaths[idx] ?? tabPaths[idx - 1] ?? tabPaths[0] ?? null;
            }
            return { ...sp, tabPaths, activePath };
          }
          if (sp.id === toSplitId) {
            // If already present in destination (rare — same path opened in
            // both splits), just focus it.
            if (sp.tabPaths.includes(path)) {
              return { ...sp, activePath: path };
            }
            const insertAt =
              beforePath == null
                ? sp.tabPaths.length
                : sp.tabPaths.indexOf(beforePath);
            const at = insertAt < 0 ? sp.tabPaths.length : insertAt;
            const tabPaths = [...sp.tabPaths];
            tabPaths.splice(at, 0, path);
            return { ...sp, tabPaths, activePath: path };
          }
          return sp;
        });
        // Drop the source split if it was emptied and there's a peer left.
        const src = splits.find((sp) => sp.id === fromSplitId);
        let activeSplitId = toSplitId;
        if (src && src.tabPaths.length === 0 && splits.length > 1) {
          splits = splits.filter((sp) => sp.id !== fromSplitId);
        }
        return { ...s, splits, activeSplitId };
      });
    },
    [],
  );

  return {
    state,
    openFile,
    setContent,
    saveFile,
    closeTab,
    closeOthers,
    closeAll,
    setActiveTab,
    setActiveSplit,
    splitVertical,
    moveTab,
    isDirty,
    dropPath,
    forceSaveFile,
    reloadFile,
    clearConflict,
  };
}
