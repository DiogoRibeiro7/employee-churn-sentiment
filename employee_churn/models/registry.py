"""Filesystem-based model registry for versioning and artifact management.

Stores trained model *bundles* (the dictionaries produced by
``scripts.train_model.train_model_bundle``) under a versioned directory layout
and maintains a JSON index of metadata so models can be listed, compared, and
reloaded without re-training:

    models/artifacts/
    ├── registry.json
    └── <model_name>/
        ├── v1.pkl
        └── v2.pkl

The registry is intentionally dependency-free (pickle + JSON) so it works in any
environment; swap in MLflow or a cloud store later behind the same interface.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_REGISTRY_ROOT = Path("models/artifacts")
_INDEX_FILENAME = "registry.json"
LATEST = "latest"


class ModelRegistry:
    """Version and persist model bundles on the local filesystem."""

    def __init__(self, root: str | Path = DEFAULT_REGISTRY_ROOT) -> None:
        """Create a registry rooted at ``root`` (created lazily on first write).

        Args:
            root: Directory under which model versions and the index live.
        """
        self.root = Path(root)

    @property
    def index_path(self) -> Path:
        """Path to the JSON registry index."""
        return self.root / _INDEX_FILENAME

    def _load_index(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self.index_path.is_file():
            return {}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save_index(self, index: Dict[str, List[Dict[str, Any]]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
        )

    def register(
        self,
        bundle: Dict[str, Any],
        name: str,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
    ) -> str:
        """Persist a model bundle as a new version and update the index.

        Args:
            bundle: The serializable model bundle to store.
            name: Logical model name (its directory under the registry root).
            metrics: Optional metric snapshot recorded in the index. Falls back
                to ``bundle["metrics"]`` when present.
            tags: Optional free-form metadata recorded in the index.
            created_at: Optional ISO timestamp; left to the caller so the
                registry stays deterministic and testable.

        Returns:
            The assigned version string (e.g. ``"v1"``).
        """
        index = self._load_index()
        versions = index.setdefault(name, [])
        version = f"v{len(versions) + 1}"

        model_dir = self.root / name
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = model_dir / f"{version}.pkl"
        with artifact_path.open("wb") as handle:
            pickle.dump(bundle, handle)

        entry = {
            "version": version,
            "path": str(artifact_path.relative_to(self.root)),
            "metrics": metrics if metrics is not None else bundle.get("metrics", {}),
            "tags": tags or {},
            "created_at": created_at,
        }
        versions.append(entry)
        self._save_index(index)
        return version

    def list_models(self) -> List[str]:
        """Return the registered model names."""
        return sorted(self._load_index())

    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """Return metadata entries for every version of ``name``.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        index = self._load_index()
        if name not in index:
            raise KeyError(f"unknown model '{name}'")
        return index[name]

    def latest_version(self, name: str) -> Optional[str]:
        """Return the most recent version string for ``name``, or ``None``."""
        versions = self._load_index().get(name, [])
        return versions[-1]["version"] if versions else None

    def _resolve_version(self, name: str, version: str) -> Dict[str, Any]:
        versions = self.list_versions(name)
        if version == LATEST:
            return versions[-1]
        for entry in versions:
            if entry["version"] == version:
                return entry
        raise KeyError(f"unknown version '{version}' for model '{name}'")

    def get_metadata(self, name: str, version: str = LATEST) -> Dict[str, Any]:
        """Return the index metadata for a model version."""
        return self._resolve_version(name, version)

    def load(self, name: str, version: str = LATEST) -> Dict[str, Any]:
        """Load and unpickle a stored model bundle.

        Args:
            name: Registered model name.
            version: Specific version string, or ``"latest"``.

        Returns:
            The deserialized model bundle.
        """
        entry = self._resolve_version(name, version)
        artifact_path = self.root / entry["path"]
        with artifact_path.open("rb") as handle:
            return pickle.load(handle)
