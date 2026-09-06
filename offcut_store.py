# SPDX-License-Identifier: GPL-3.0-only
"""SQLite persistence for Krea 2 boards, images, and enhancer connections."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# A LoRA stack entry with no strength of its own starts here. Useful strength is around 1.0 and
# most adapters degrade well before 2.0, so the shipped default sits just under that; a profile
# overrides it per LoRA.
DEFAULT_LORA_STRENGTH = 0.8

# Every effort name any provider exposes. Which subset a given model actually accepts is decided
# per model by the agent bridge's describer, not here; this is only the vocabulary a stored value
# has to belong to before it is worth asking about.
REASONING_EFFORTS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SecretStore:
    """Store connection API keys in a private local file."""

    def __init__(self, path: Path):
        self.path = path
        self.cache: dict[str, str] = {}
        self.persistent: set[str] = set()
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read connection secrets from {self.path}: {exc}") from exc
        if not isinstance(data, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in data.items()
        ):
            raise RuntimeError(f"Connection secrets file has an invalid format: {self.path}")
        self.path.parent.chmod(0o700)
        self.path.chmod(0o600)
        self.cache.update(data)
        self.persistent.update(data)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        if not self.cache:
            self.path.unlink(missing_ok=True)
            return
        descriptor, temporary_name = tempfile.mkstemp(prefix=".secrets-", dir=self.path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(self.cache, file, separators=(",", ":"), sort_keys=True)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def get(self, connection_id: str) -> str:
        return self.cache.get(connection_id, "")

    def set(self, connection_id: str, value: str, _label: str) -> bool:
        previous = self.cache.get(connection_id)
        self.cache[connection_id] = value
        try:
            self._write()
        except OSError as exc:
            if previous is None:
                self.cache.pop(connection_id, None)
            else:
                self.cache[connection_id] = previous
            raise RuntimeError(f"Could not persist connection API key: {exc}") from exc
        self.persistent.add(connection_id)
        return True

    def delete(self, connection_id: str) -> None:
        previous = self.cache.pop(connection_id, None)
        try:
            self._write()
        except OSError as exc:
            if previous is not None:
                self.cache[connection_id] = previous
            raise RuntimeError(f"Could not delete persisted connection API key: {exc}") from exc
        self.persistent.discard(connection_id)

    def storage(self, connection_id: str) -> str:
        if connection_id in self.persistent:
            return "local-file"
        if connection_id in self.cache:
            return "session"
        return "none"


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        secrets_path = Path(os.environ.get("OFFCUT_SECRETS", self.path.with_suffix(".secrets.json"))).expanduser()
        self.secrets = SecretStore(secrets_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self.lock, self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS boards (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    settings_revision INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY,
                    board_id TEXT NOT NULL REFERENCES boards(id) ON DELETE RESTRICT,
                    path TEXT NOT NULL UNIQUE,
                    raw_prompt TEXT NOT NULL DEFAULT '',
                    enhanced_prompt TEXT NOT NULL DEFAULT '',
                    final_prompt TEXT NOT NULL DEFAULT '',
                    preset TEXT NOT NULL DEFAULT 'turbo-int8',
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    steps INTEGER NOT NULL,
                    guidance REAL NOT NULL,
                    negative_prompt TEXT NOT NULL DEFAULT '',
                    loras_json TEXT NOT NULL DEFAULT '[]',
                    enhance INTEGER NOT NULL DEFAULT 0,
                    connection_id TEXT,
                    enhancer_model TEXT NOT NULL DEFAULT '',
                    favorite INTEGER NOT NULL DEFAULT 0,
                    position REAL NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    kind TEXT NOT NULL DEFAULT 'generated',
                    run_key TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS images_board_position ON images(board_id, position DESC);
                CREATE INDEX IF NOT EXISTS images_favorite ON images(favorite, created_at DESC);

                CREATE TABLE IF NOT EXISTS connections (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    protocol TEXT NOT NULL DEFAULT 'chat',
                    models_json TEXT NOT NULL DEFAULT '[]',
                    api_key_env TEXT NOT NULL DEFAULT '',
                    secret_persistent INTEGER NOT NULL DEFAULT 0,
                    default_model TEXT NOT NULL DEFAULT '',
                    default_reasoning TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    board_id TEXT NOT NULL REFERENCES boards(id) ON DELETE CASCADE,
                    title TEXT NOT NULL DEFAULT 'New chat',
                    connection_id TEXT REFERENCES connections(id) ON DELETE SET NULL,
                    model TEXT NOT NULL,
                    permission_mode TEXT NOT NULL CHECK(permission_mode IN ('draft', 'create', 'iterate')),
                    system_mode TEXT NOT NULL DEFAULT '',
                    notified_mode TEXT NOT NULL DEFAULT '',
                    reasoning_effort TEXT NOT NULL DEFAULT 'high',
                    workspace_revision INTEGER NOT NULL DEFAULT 0,
                    workspace_prompt TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    compacted_before INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chat_sessions_board_updated
                    ON chat_sessions(board_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS styles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    style_text TEXT NOT NULL,
                    reference_image_id TEXT REFERENCES images(id) ON DELETE SET NULL,
                    source TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS styles_name ON styles(name COLLATE NOCASE);

                CREATE TABLE IF NOT EXISTS lora_profiles (
                    name TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    default_strength REAL NOT NULL DEFAULT 0.8,  -- DEFAULT_LORA_STRENGTH
                    cover_image_id TEXT REFERENCES images(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    message_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS chat_messages_session_sequence
                    ON chat_messages(session_id, sequence);
                """
            )
            db.executescript("""
                CREATE TABLE IF NOT EXISTS chat_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    image_id TEXT REFERENCES images(id) ON DELETE SET NULL,
                    recipe_json TEXT NOT NULL,
                    change_note TEXT NOT NULL DEFAULT '',
                    observation TEXT NOT NULL DEFAULT '',
                    outcome TEXT NOT NULL DEFAULT 'unreviewed',
                    reused INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chat_attempts_session ON chat_attempts(session_id, id);
                CREATE TABLE IF NOT EXISTS chat_turns (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            style_columns = {row["name"] for row in db.execute("PRAGMA table_info(styles)")}
            if "kind" not in style_columns:
                db.execute("ALTER TABLE styles ADD COLUMN kind TEXT NOT NULL DEFAULT 'art'")
            board_columns = {row["name"] for row in db.execute("PRAGMA table_info(boards)").fetchall()}
            if "settings_revision" not in board_columns:
                db.execute("ALTER TABLE boards ADD COLUMN settings_revision INTEGER NOT NULL DEFAULT 0")
            connection_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(connections)").fetchall()
            }
            if "secret_persistent" not in connection_columns:
                db.execute("ALTER TABLE connections ADD COLUMN secret_persistent INTEGER NOT NULL DEFAULT 0")
            if "default_model" not in connection_columns:
                db.execute("ALTER TABLE connections ADD COLUMN default_model TEXT NOT NULL DEFAULT ''")
            if "default_reasoning" not in connection_columns:
                db.execute("ALTER TABLE connections ADD COLUMN default_reasoning TEXT NOT NULL DEFAULT ''")
            image_columns = {row["name"] for row in db.execute("PRAGMA table_info(images)").fetchall()}
            if "kind" not in image_columns:
                db.execute("ALTER TABLE images ADD COLUMN kind TEXT NOT NULL DEFAULT 'generated'")
            if "run_key" not in image_columns:
                db.execute("ALTER TABLE images ADD COLUMN run_key TEXT NOT NULL DEFAULT ''")
            # Created after the ALTER rather than in the script above, which runs before it and on
            # an existing database would name a column that is not there yet.
            db.execute("CREATE INDEX IF NOT EXISTS images_run_key ON images(board_id, kind, run_key)")
            chat_columns = {row["name"] for row in db.execute("PRAGMA table_info(chat_sessions)").fetchall()}
            if "creative_brief_json" not in chat_columns:
                db.execute("ALTER TABLE chat_sessions ADD COLUMN creative_brief_json TEXT NOT NULL DEFAULT '{}'")
            if "generation_limit" not in chat_columns:
                db.execute("ALTER TABLE chat_sessions ADD COLUMN generation_limit INTEGER NOT NULL DEFAULT 4")
            # Legacy modes remain readable in old messages, but all active chats use Create.
            db.execute("UPDATE chat_sessions SET permission_mode = 'create' WHERE permission_mode != 'create'")
            if "reasoning_effort" not in chat_columns:
                db.execute("ALTER TABLE chat_sessions ADD COLUMN reasoning_effort TEXT NOT NULL DEFAULT 'high'")
            if "workspace_revision" not in chat_columns:
                db.execute("ALTER TABLE chat_sessions ADD COLUMN workspace_revision INTEGER NOT NULL DEFAULT 0")
            if "workspace_prompt" not in chat_columns:
                db.execute("ALTER TABLE chat_sessions ADD COLUMN workspace_prompt TEXT NOT NULL DEFAULT ''")
            if "system_mode" not in chat_columns:
                db.execute("ALTER TABLE chat_sessions ADD COLUMN system_mode TEXT NOT NULL DEFAULT ''")
            if "notified_mode" not in chat_columns:
                db.execute("ALTER TABLE chat_sessions ADD COLUMN notified_mode TEXT NOT NULL DEFAULT ''")
            if db.execute("SELECT COUNT(*) FROM boards").fetchone()[0] == 0:
                now = utc_now()
                db.execute(
                    "INSERT INTO boards (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    ("inbox", "Inbox", "Imported images and quick studies", now, now),
                )
            if db.execute("SELECT COUNT(*) FROM connections").fetchone()[0] == 0:
                now = utc_now()
                db.execute(
                    """INSERT INTO connections
                       (id, name, base_url, protocol, models_json, api_key_env, secret_persistent, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                    (
                        "opencode-go",
                        "OpenCode Go",
                        "https://opencode.ai/zen/go/v1",
                        "opencode-go",
                        json.dumps(["glm-5.2"]),
                        "OPENCODE_GO_API_KEY",
                        now,
                        now,
                    ),
                )

    @staticmethod
    def _json(value: str, fallback: Any) -> Any:
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return fallback

    def list_boards(self) -> list[dict[str, Any]]:
        with self.lock, self._connect() as db:
            rows = db.execute(
                """SELECT b.*, COUNT(i.id) AS image_count,
                          COALESCE(SUM(i.favorite), 0) AS favorite_count,
                          (SELECT path FROM images cover WHERE cover.board_id = b.id
                           ORDER BY cover.position DESC, cover.created_at DESC LIMIT 1) AS cover_path
                          ,(SELECT id FROM images cover WHERE cover.board_id = b.id
                           ORDER BY cover.position DESC, cover.created_at DESC LIMIT 1) AS cover_id
                   FROM boards b LEFT JOIN images i ON i.board_id = b.id
                   GROUP BY b.id ORDER BY b.updated_at DESC"""
            ).fetchall()
            boards = [self._board(row) for row in rows]
            for board in boards:
                board["cover_ids"] = [
                    cover["id"]
                    for cover in db.execute(
                        "SELECT id FROM images WHERE board_id = ? ORDER BY position DESC, created_at DESC LIMIT 4",
                        (board["id"],),
                    ).fetchall()
                ]
        return boards

    def get_board(self, board_id: str) -> dict[str, Any]:
        with self.lock, self._connect() as db:
            row = db.execute(
                """SELECT b.*, COUNT(i.id) AS image_count,
                          COALESCE(SUM(i.favorite), 0) AS favorite_count,
                          (SELECT path FROM images cover WHERE cover.board_id = b.id
                           ORDER BY cover.position DESC, cover.created_at DESC LIMIT 1) AS cover_path
                          ,(SELECT id FROM images cover WHERE cover.board_id = b.id
                           ORDER BY cover.position DESC, cover.created_at DESC LIMIT 1) AS cover_id
                   FROM boards b LEFT JOIN images i ON i.board_id = b.id
                   WHERE b.id = ? GROUP BY b.id""",
                (board_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Board not found")
        board = self._board(row)
        with self.lock, self._connect() as db:
            board["cover_ids"] = [
                cover["id"]
                for cover in db.execute(
                    "SELECT id FROM images WHERE board_id = ? ORDER BY position DESC, created_at DESC LIMIT 4",
                    (board_id,),
                ).fetchall()
            ]
        return board

    def _board(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["settings"] = self._json(data.pop("settings_json"), {})
        return data

    def create_board(self, name: str, description: str = "") -> dict[str, Any]:
        if not isinstance(name, str) or not isinstance(description, str):
            raise ValueError("Board name and description must be strings")
        name = name.strip()
        if not 1 <= len(name) <= 100:
            raise ValueError("Board name must contain between 1 and 100 characters")
        if len(description) > 500:
            raise ValueError("Board description is limited to 500 characters")
        board_id = uuid.uuid4().hex
        now = utc_now()
        with self.lock, self._connect() as db:
            db.execute(
                "INSERT INTO boards (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (board_id, name, description.strip(), now, now),
            )
        return self.get_board(board_id)

    def update_board(self, board_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock, self._connect() as db:
            board = db.execute("SELECT * FROM boards WHERE id = ?", (board_id,)).fetchone()
            if board is None:
                raise ValueError("Board not found")
            name = payload.get("name", board["name"])
            description = payload.get("description", board["description"])
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
                raise ValueError("Board name must contain between 1 and 100 characters")
            if not isinstance(description, str) or len(description) > 500:
                raise ValueError("Board description is limited to 500 characters")
            expected_revision = payload.get("expected_settings_revision")
            if "expected_settings_revision" in payload and (
                type(expected_revision) is not int or expected_revision < 0
            ):
                raise ValueError("Expected settings revision must be a non-negative integer")
            now = utc_now()
            if "settings" in payload:
                settings = payload["settings"]
                if not isinstance(settings, dict):
                    raise ValueError("Board settings must be an object")
                try:
                    settings_json = json.dumps(settings)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Board settings must be JSON serializable") from exc
                parameters: list[Any] = [name.strip(), description.strip(), settings_json, now, board_id]
                revision_clause = ""
                if expected_revision is not None:
                    revision_clause = " AND settings_revision = ?"
                    parameters.append(expected_revision)
                cursor = db.execute(
                    """UPDATE boards SET name = ?, description = ?, settings_json = ?,
                              settings_revision = settings_revision + 1, updated_at = ?
                       WHERE id = ?""" + revision_clause,
                    parameters,
                )
                if cursor.rowcount != 1:
                    raise ValueError("Board settings changed; reload the board and try again")
            else:
                db.execute(
                    "UPDATE boards SET name = ?, description = ?, updated_at = ? WHERE id = ?",
                    (name.strip(), description.strip(), now, board_id),
                )
        return self.get_board(board_id)

    def delete_board(self, board_id: str) -> None:
        if board_id == "inbox":
            raise ValueError("The Inbox board cannot be deleted")
        self.get_board(board_id)
        with self.lock, self._connect() as db:
            next_position = db.execute(
                "SELECT COALESCE(MAX(position), 0) FROM images WHERE board_id = 'inbox'"
            ).fetchone()[0]
            rows = db.execute("SELECT id FROM images WHERE board_id = ? ORDER BY position", (board_id,)).fetchall()
            for offset, row in enumerate(rows, 1):
                db.execute(
                    "UPDATE images SET board_id = 'inbox', position = ? WHERE id = ?",
                    (next_position + offset, row["id"]),
                )
            db.execute("DELETE FROM boards WHERE id = ?", (board_id,))
            db.execute("UPDATE boards SET updated_at = ? WHERE id = 'inbox'", (utc_now(),))

    def import_existing(self, output_root: Path) -> int:
        if not output_root.is_dir():
            return 0
        from PIL import Image

        imported = 0
        for path in sorted(output_root.rglob("*.png"), key=lambda item: item.stat().st_mtime):
            resolved = str(path.resolve())
            with self.lock, self._connect() as db:
                if db.execute("SELECT 1 FROM images WHERE path = ?", (resolved,)).fetchone():
                    continue
            metadata: dict[str, Any] = {}
            try:
                with Image.open(path) as image:
                    metadata = self._json(image.info.get("krea2", "{}"), {})
                    width, height = image.size
            except OSError:
                continue
            result = {
                **metadata,
                "output_path": resolved,
                "width": metadata.get("width", width),
                "height": metadata.get("height", height),
                "seed": metadata.get("seed", 0),
                "steps": metadata.get("steps", 8),
                "guidance": metadata.get("guidance", 0.0),
                "preset": metadata.get("preset", "turbo-int8"),
                "loras": metadata.get("loras", []),
            }
            request = {
                key: metadata[key]
                for key in (
                    "raw_prompt",
                    "preset",
                    "width",
                    "height",
                    "seed",
                    "steps",
                    "guidance",
                    "negative_prompt",
                    "styles",
                    "loras",
                    "enhance",
                    "triggers",
                )
                if key in metadata
            }
            request["prompt"] = request.pop("raw_prompt", "")
            self.record_image(
                "inbox",
                result,
                request,
                enhanced_prompt=metadata.get("enhanced_prompt", ""),
            )
            imported += 1
        return imported

    def record_image(
        self,
        board_id: str,
        result: dict[str, Any],
        request: dict[str, Any],
        enhanced_prompt: str = "",
        connection_id: str | None = None,
        enhancer_model: str = "",
        kind: str = "generated",
        run_key: str = "",
    ) -> dict[str, Any]:
        image_id = uuid.uuid4().hex
        path = str(Path(result["output_path"]).resolve())
        now = utc_now()
        with self.lock, self._connect() as db:
            if db.execute("SELECT 1 FROM boards WHERE id = ?", (board_id,)).fetchone() is None:
                raise ValueError("Board not found")
            position = db.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 FROM images WHERE board_id = ?", (board_id,)
            ).fetchone()[0]
            db.execute(
                """INSERT OR IGNORE INTO images
                   (id, board_id, path, raw_prompt, enhanced_prompt, final_prompt, preset,
                    width, height, seed, steps, guidance, negative_prompt, loras_json,
                    enhance, connection_id, enhancer_model, position, metadata_json, kind,
                    run_key, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    image_id,
                    board_id,
                    path,
                    result.get("raw_prompt", request.get("prompt", "")),
                    enhanced_prompt or result.get("enhanced_prompt", ""),
                    result.get("final_prompt", ""),
                    result.get("preset", "turbo-int8"),
                    int(result.get("width", 1024)),
                    int(result.get("height", 1024)),
                    int(result.get("seed", 0)),
                    int(result.get("steps", 8)),
                    float(result.get("guidance", 0.0)),
                    request.get("negative_prompt", ""),
                    json.dumps(request.get("loras", result.get("loras", []))),
                    int(bool(request.get("enhance", False))),
                    connection_id,
                    enhancer_model,
                    position,
                    json.dumps(result),
                    kind,
                    run_key,
                    now,
                ),
            )
            row = db.execute("SELECT id FROM images WHERE path = ?", (path,)).fetchone()
            image_id = row["id"]
            board_settings = {
                key: request[key]
                for key in (
                    "prompt",
                    "preset",
                    "width",
                    "height",
                    "steps",
                    "guidance",
                    "negative_prompt",
                    "seed",
                    "styles",
                    "loras",
                    "enhance",
                    "connection_id",
                    "enhancer_model",
                    "triggers",
                )
                if key in request
            }
            if kind == "generated":
                db.execute(
                    """UPDATE boards SET settings_json = ?, settings_revision = settings_revision + 1,
                              updated_at = ? WHERE id = ?""",
                    (json.dumps(board_settings), now, board_id),
                )
        return self.get_image(image_id)

    def record_reference_image(
        self,
        board_id: str,
        path: Path,
        width: int,
        height: int,
        label: str = "",
    ) -> dict[str, Any]:
        image_id = uuid.uuid4().hex
        resolved = str(path.resolve())
        now = utc_now()
        with self.lock, self._connect() as db:
            if db.execute("SELECT 1 FROM boards WHERE id = ?", (board_id,)).fetchone() is None:
                raise ValueError("Board not found")
            position = db.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 FROM images WHERE board_id = ?", (board_id,)
            ).fetchone()[0]
            db.execute(
                """INSERT INTO images
                   (id, board_id, path, raw_prompt, enhanced_prompt, final_prompt, preset,
                    width, height, seed, steps, guidance, negative_prompt, loras_json,
                    enhance, connection_id, enhancer_model, position, metadata_json, kind, created_at)
                   VALUES (?, ?, ?, ?, '', '', 'reference', ?, ?, 0, 0, 0, '', '[]',
                           0, NULL, '', ?, ?, 'reference', ?)""",
                (
                    image_id,
                    board_id,
                    resolved,
                    label.strip()[:500],
                    int(width),
                    int(height),
                    position,
                    json.dumps({"reference": True, "label": label.strip()[:500]}),
                    now,
                ),
            )
            db.execute("UPDATE boards SET updated_at = ? WHERE id = ?", (now, board_id))
        return self.get_image(image_id)

    def list_images(
        self,
        board_id: str | None = None,
        query: str = "",
        favorite: bool = False,
        limit: int = 200,
        kinds: tuple[str, ...] = ("generated", "reference"),
    ) -> list[dict[str, Any]]:
        # A cover belongs to a library entry, not to a board. It is a real image row so that
        # cover_image_id and reference_image_id have something to point at, but every listing that
        # draws a board, the gallery or the rail is asking about the user's own work, so covers are
        # left out unless a caller names that kind.
        clauses: list[str] = []
        parameters: list[Any] = []
        if kinds:
            clauses.append(f"i.kind IN ({', '.join('?' * len(kinds))})")
            parameters.extend(kinds)
        if board_id:
            clauses.append("i.board_id = ?")
            parameters.append(board_id)
        if query.strip():
            clauses.append("(i.raw_prompt LIKE ? OR i.enhanced_prompt LIKE ? OR i.final_prompt LIKE ?)")
            needle = f"%{query.strip()}%"
            parameters.extend((needle, needle, needle))
        if favorite:
            clauses.append("i.favorite = 1")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        maximum = 5000 if board_id else 500
        parameters.append(max(1, min(limit, maximum)))
        order = "i.position DESC, i.created_at DESC" if board_id else "i.favorite DESC, i.created_at DESC"
        with self.lock, self._connect() as db:
            rows = db.execute(
                f"""SELECT i.*, b.name AS board_name FROM images i
                     JOIN boards b ON b.id = i.board_id{where}
                     ORDER BY {order} LIMIT ?""",
                parameters,
            ).fetchall()
        return [self._image(row) for row in rows]

    def find_run(self, board_id: str, run_key: str, kind: str = "generated") -> dict[str, Any] | None:
        """Return the first image on this board that was sampled from byte-identical inputs.

        The empty key never matches. It is what rows written before this column existed carry, and
        what a caller passes for a run it does not want served from an earlier frame.
        """
        if not run_key:
            return None
        with self.lock, self._connect() as db:
            row = db.execute(
                """SELECT i.*, b.name AS board_name FROM images i JOIN boards b ON b.id = i.board_id
                   WHERE i.board_id = ? AND i.kind = ? AND i.run_key = ?
                   ORDER BY i.created_at, i.id LIMIT 1""",
                (board_id, kind, run_key),
            ).fetchone()
        return self._image(row) if row is not None else None

    def get_image(self, image_id: str) -> dict[str, Any]:
        with self.lock, self._connect() as db:
            row = db.execute(
                "SELECT i.*, b.name AS board_name FROM images i JOIN boards b ON b.id = i.board_id WHERE i.id = ?",
                (image_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Image not found")
        return self._image(row)

    def _image(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["loras"] = self._json(data.pop("loras_json"), [])
        data["metadata"] = self._json(data.pop("metadata_json"), {})
        data["favorite"] = bool(data["favorite"])
        data["enhance"] = bool(data["enhance"])
        return data

    def update_image(self, image_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock, self._connect() as db:
            row = db.execute("SELECT board_id, favorite, position FROM images WHERE id = ?", (image_id,)).fetchone()
            if row is None:
                raise ValueError("Image not found")
            old_board_id = row["board_id"]
            board_id = payload.get("board_id", old_board_id)
            if not isinstance(board_id, str) or db.execute(
                "SELECT 1 FROM boards WHERE id = ?", (board_id,)
            ).fetchone() is None:
                raise ValueError("Board not found")
            assignments: list[str] = []
            parameters: list[Any] = []
            if "favorite" in payload:
                favorite = payload["favorite"]
                if not isinstance(favorite, bool):
                    raise ValueError("favorite must be a boolean")
                assignments.append("favorite = ?")
                parameters.append(int(favorite))
            if board_id != old_board_id:
                position = db.execute(
                    "SELECT COALESCE(MAX(position), 0) + 1 FROM images WHERE board_id = ?", (board_id,)
                ).fetchone()[0]
                assignments.extend(("board_id = ?", "position = ?"))
                parameters.extend((board_id, position))
            if assignments:
                parameters.append(image_id)
                db.execute(f"UPDATE images SET {', '.join(assignments)} WHERE id = ?", parameters)
            now = utc_now()
            db.execute("UPDATE boards SET updated_at = ? WHERE id IN (?, ?)", (now, board_id, old_board_id))
        return self.get_image(image_id)

    def delete_image(self, image_id: str) -> dict[str, Any]:
        """Drop one image row and hand the record back so the caller can remove its files."""
        image = self.get_image(image_id)
        with self.lock, self._connect() as db:
            db.execute("DELETE FROM images WHERE id = ?", (image_id,))
            db.execute("UPDATE boards SET updated_at = ? WHERE id = ?", (utc_now(), image["board_id"]))
        return image

    def reorder_images(self, board_id: str, image_ids: list[str]) -> None:
        self.get_board(board_id)
        with self.lock, self._connect() as db:
            existing = {
                row["id"]
                for row in db.execute("SELECT id FROM images WHERE board_id = ?", (board_id,)).fetchall()
            }
            if len(image_ids) != len(set(image_ids)) or set(image_ids) != existing:
                raise ValueError("Reorder list must contain every image in the board exactly once")
            top = len(image_ids)
            for index, image_id in enumerate(image_ids):
                db.execute("UPDATE images SET position = ? WHERE id = ?", (top - index, image_id))
            db.execute("UPDATE boards SET updated_at = ? WHERE id = ?", (utc_now(), board_id))

    @staticmethod
    def _validate_permission_mode(permission_mode: Any) -> str:
        if permission_mode not in ("draft", "create", "iterate"):
            raise ValueError("Permission mode must be draft, create, or iterate")
        return "create"  # Accept old clients while retiring the old modes.

    @staticmethod
    def _message_preview(message: dict[str, Any]) -> str:
        content = message.get("content", message.get("text", ""))
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    value = block.get("text", block.get("content", ""))
                    if isinstance(value, str):
                        parts.append(value)
            text = " ".join(parts)
        else:
            text = ""
        return " ".join(text.split())[:200]

    def _chat(self, row: sqlite3.Row, db: sqlite3.Connection) -> dict[str, Any]:
        data = dict(row)
        data["permission_mode"] = "create"
        data["creative_brief"] = self._json(data.pop("creative_brief_json"), {})
        message_rows = db.execute(
            "SELECT message_json FROM chat_messages WHERE session_id = ? ORDER BY sequence",
            (data["id"],),
        ).fetchall()
        messages = [self._json(message["message_json"], {}) for message in message_rows]
        data["message_count"] = len(messages)
        visible_messages = [
            message
            for message in messages
            if isinstance(message, dict) and message.get("role") in ("user", "assistant")
        ]
        data["latest_preview"] = self._message_preview(visible_messages[-1]) if visible_messages else ""
        total_usage: dict[str, int | float] = {}
        total_cost: int | float = 0
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            usage = message.get("usage")
            if isinstance(usage, dict):
                for key, value in usage.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if key in ("cost", "cost_usd", "total_cost"):
                            total_cost += value
                        else:
                            total_usage[key] = total_usage.get(key, 0) + value
                    elif key == "cost" and isinstance(value, dict):
                        cost_total = value.get("total")
                        if isinstance(cost_total, (int, float)) and not isinstance(cost_total, bool):
                            total_cost += cost_total
            cost = message.get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                total_cost += cost
        data["total_usage"] = total_usage
        data["total_cost"] = total_cost
        return data

    def create_chat(
        self,
        board_id: str,
        connection_id: str | None,
        model: str,
        permission_mode: str,
        reasoning_effort: str = "high",
    ) -> dict[str, Any]:
        if not isinstance(board_id, str) or not board_id:
            raise ValueError("Board ID is required")
        if connection_id is not None and (not isinstance(connection_id, str) or not connection_id):
            raise ValueError("Connection ID must be a non-empty string or null")
        if not isinstance(model, str) or not 1 <= len(model.strip()) <= 200:
            raise ValueError("Chat model must contain between 1 and 200 characters")
        permission_mode = self._validate_permission_mode(permission_mode)
        if reasoning_effort not in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
            raise ValueError("Unsupported reasoning effort")
        chat_id = uuid.uuid4().hex
        now = utc_now()
        with self.lock, self._connect() as db:
            board = db.execute(
                "SELECT settings_revision, settings_json FROM boards WHERE id = ?", (board_id,)
            ).fetchone()
            if board is None:
                raise ValueError("Board not found")
            if connection_id is not None and db.execute(
                "SELECT 1 FROM connections WHERE id = ?", (connection_id,)
            ).fetchone() is None:
                raise ValueError("Connection not found")
            db.execute(
                """INSERT INTO chat_sessions
                   (id, board_id, connection_id, model, permission_mode, system_mode,
                    notified_mode, reasoning_effort,
                    workspace_revision, workspace_prompt, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chat_id,
                    board_id,
                    connection_id,
                    model.strip(),
                    permission_mode,
                    permission_mode,
                    permission_mode,
                    reasoning_effort,
                    board["settings_revision"],
                    str(self._json(board["settings_json"], {}).get("prompt", "")),
                    now,
                    now,
                ),
            )
        return self.get_chat(chat_id)

    def list_chats(self, board_id: str) -> list[dict[str, Any]]:
        if not isinstance(board_id, str) or not board_id:
            raise ValueError("Board ID is required")
        with self.lock, self._connect() as db:
            if db.execute("SELECT 1 FROM boards WHERE id = ?", (board_id,)).fetchone() is None:
                raise ValueError("Board not found")
            rows = db.execute(
                """SELECT * FROM chat_sessions WHERE board_id = ?
                   ORDER BY updated_at DESC, created_at DESC""",
                (board_id,),
            ).fetchall()
            return [self._chat(row, db) for row in rows]

    def get_chat(self, chat_id: str) -> dict[str, Any]:
        if not isinstance(chat_id, str) or not chat_id:
            raise ValueError("Chat ID is required")
        with self.lock, self._connect() as db:
            row = db.execute("SELECT * FROM chat_sessions WHERE id = ?", (chat_id,)).fetchone()
            if row is None:
                raise ValueError("Chat not found")
            return self._chat(row, db)

    def update_chat(self, chat_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(chat_id, str) or not chat_id:
            raise ValueError("Chat ID is required")
        if not isinstance(payload, dict):
            raise ValueError("Chat update must be an object")
        assignments: list[str] = []
        parameters: list[Any] = []
        with self.lock, self._connect() as db:
            if db.execute("SELECT 1 FROM chat_sessions WHERE id = ?", (chat_id,)).fetchone() is None:
                raise ValueError("Chat not found")
            if "generation_limit" in payload:
                limit = payload["generation_limit"]
                if type(limit) is not int or not 1 <= limit <= 50:
                    raise ValueError("Generation limit must be between 1 and 50")
                assignments.append("generation_limit = ?")
                parameters.append(limit)
            if "creative_brief" in payload:
                brief = payload["creative_brief"]
                if not isinstance(brief, dict) or len(json.dumps(brief)) > 16000:
                    raise ValueError("Creative brief must be an object under 16,000 characters")
                assignments.append("creative_brief_json = ?")
                parameters.append(json.dumps(brief))
            if "title" in payload:
                title = payload["title"]
                if not isinstance(title, str) or not 1 <= len(title.strip()) <= 200:
                    raise ValueError("Chat title must contain between 1 and 200 characters")
                assignments.append("title = ?")
                parameters.append(title.strip())
            if "connection_id" in payload:
                connection_id = payload["connection_id"]
                if connection_id is not None and (not isinstance(connection_id, str) or not connection_id):
                    raise ValueError("Connection ID must be a non-empty string or null")
                if connection_id is not None and db.execute(
                    "SELECT 1 FROM connections WHERE id = ?", (connection_id,)
                ).fetchone() is None:
                    raise ValueError("Connection not found")
                assignments.append("connection_id = ?")
                parameters.append(connection_id)
            if "model" in payload:
                model = payload["model"]
                if not isinstance(model, str) or not 1 <= len(model.strip()) <= 200:
                    raise ValueError("Chat model must contain between 1 and 200 characters")
                assignments.append("model = ?")
                parameters.append(model.strip())
            if "permission_mode" in payload:
                assignments.append("permission_mode = ?")
                parameters.append(self._validate_permission_mode(payload["permission_mode"]))
            if "system_mode" in payload:
                if db.execute(
                    "SELECT system_mode FROM chat_sessions WHERE id = ?", (chat_id,)
                ).fetchone()["system_mode"]:
                    raise ValueError("system_mode is fixed once a chat has recorded one")
                assignments.append("system_mode = ?")
                parameters.append(self._validate_permission_mode(payload["system_mode"]))
            if "notified_mode" in payload:
                assignments.append("notified_mode = ?")
                parameters.append(self._validate_permission_mode(payload["notified_mode"]))
            if "reasoning_effort" in payload:
                reasoning_effort = payload["reasoning_effort"]
                if reasoning_effort not in REASONING_EFFORTS:
                    raise ValueError("Unsupported reasoning effort")
                assignments.append("reasoning_effort = ?")
                parameters.append(reasoning_effort)
            if "workspace_revision" in payload:
                workspace_revision = payload["workspace_revision"]
                if type(workspace_revision) is not int or workspace_revision < 0:
                    raise ValueError("workspace_revision must be a non-negative integer")
                assignments.append("workspace_revision = ?")
                parameters.append(workspace_revision)
            if "workspace_prompt" in payload:
                workspace_prompt = payload["workspace_prompt"]
                if not isinstance(workspace_prompt, str) or len(workspace_prompt) > 20_000:
                    raise ValueError("workspace_prompt must be a string no longer than 20,000 characters")
                assignments.append("workspace_prompt = ?")
                parameters.append(workspace_prompt)
            if "summary" in payload:
                summary = payload["summary"]
                if not isinstance(summary, str) or len(summary) > 100_000:
                    raise ValueError("Chat summary must be a string no longer than 100,000 characters")
                assignments.append("summary = ?")
                parameters.append(summary)
            if "compacted_before" in payload:
                compacted_before = payload["compacted_before"]
                if type(compacted_before) is not int or compacted_before < 0:
                    raise ValueError("compacted_before must be a non-negative integer")
                assignments.append("compacted_before = ?")
                parameters.append(compacted_before)
            if assignments:
                assignments.append("updated_at = ?")
                parameters.extend((utc_now(), chat_id))
                db.execute(f"UPDATE chat_sessions SET {', '.join(assignments)} WHERE id = ?", parameters)
        return self.get_chat(chat_id)

    def delete_chat(self, chat_id: str) -> None:
        if not isinstance(chat_id, str) or not chat_id:
            raise ValueError("Chat ID is required")
        with self.lock, self._connect() as db:
            if db.execute("DELETE FROM chat_sessions WHERE id = ?", (chat_id,)).rowcount != 1:
                raise ValueError("Chat not found")

    def append_chat_messages(
        self, chat_id: str, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not isinstance(chat_id, str) or not chat_id:
            raise ValueError("Chat ID is required")
        if not isinstance(messages, list) or len(messages) > 1000:
            raise ValueError("Chat messages must be an array of at most 1,000 objects")
        encoded: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each chat message must be an object")
            try:
                value = json.dumps(message, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise ValueError("Chat messages must be JSON serializable") from exc
            if len(value) > 2_000_000:
                raise ValueError("A chat message cannot exceed 2,000,000 characters")
            encoded.append(value)
        inserted_ids: list[str] = []
        with self.lock, self._connect() as db:
            if db.execute("SELECT 1 FROM chat_sessions WHERE id = ?", (chat_id,)).fetchone() is None:
                raise ValueError("Chat not found")
            sequence = db.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM chat_messages WHERE session_id = ?", (chat_id,)
            ).fetchone()[0]
            now = utc_now()
            for value in encoded:
                sequence += 1
                message_id = uuid.uuid4().hex
                db.execute(
                    """INSERT INTO chat_messages (id, session_id, sequence, message_json, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (message_id, chat_id, sequence, value, now),
                )
                inserted_ids.append(message_id)
            if encoded:
                db.execute("UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now, chat_id))
            if not inserted_ids:
                return []
            placeholders = ",".join("?" for _ in inserted_ids)
            rows = db.execute(
                f"SELECT * FROM chat_messages WHERE id IN ({placeholders}) ORDER BY sequence", inserted_ids
            ).fetchall()
            return [self._chat_message(row) for row in rows]

    def list_chat_messages(
        self, chat_id: str, include_compacted: bool = False
    ) -> list[dict[str, Any]]:
        if not isinstance(chat_id, str) or not chat_id:
            raise ValueError("Chat ID is required")
        if not isinstance(include_compacted, bool):
            raise ValueError("include_compacted must be a boolean")
        with self.lock, self._connect() as db:
            chat = db.execute(
                "SELECT compacted_before FROM chat_sessions WHERE id = ?", (chat_id,)
            ).fetchone()
            if chat is None:
                raise ValueError("Chat not found")
            parameters: list[Any] = [chat_id]
            clause = ""
            if not include_compacted:
                clause = " AND sequence > ?"
                parameters.append(chat["compacted_before"])
            rows = db.execute(
                f"SELECT * FROM chat_messages WHERE session_id = ?{clause} ORDER BY sequence",
                parameters,
            ).fetchall()
            return [self._chat_message(row) for row in rows]

    # The sequence a running agent turn started after, so the chat detail endpoint can hold
    # that turn's own messages back for the event replay instead of rendering them twice.
    def chat_message_watermark(self, chat_id: str) -> int:
        if not isinstance(chat_id, str) or not chat_id:
            raise ValueError("Chat ID is required")
        with self.lock, self._connect() as db:
            if db.execute("SELECT 1 FROM chat_sessions WHERE id = ?", (chat_id,)).fetchone() is None:
                raise ValueError("Chat not found")
            row = db.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM chat_messages WHERE session_id = ?", (chat_id,)
            ).fetchone()
        return int(row[0])

    def _chat_message(self, row: sqlite3.Row) -> dict[str, Any]:
        data = self._json(row["message_json"], {})
        if not isinstance(data, dict):
            data = {"message": data}
        return {
            **data,
            "id": row["id"],
            "sequence": row["sequence"],
            "created_at": row["created_at"],
        }

    def list_connections(self) -> list[dict[str, Any]]:
        with self.lock, self._connect() as db:
            rows = db.execute("SELECT * FROM connections ORDER BY name COLLATE NOCASE").fetchall()
        return [self._connection(row) for row in rows]

    def get_connection(self, connection_id: str) -> dict[str, Any]:
        with self.lock, self._connect() as db:
            row = db.execute("SELECT * FROM connections WHERE id = ?", (connection_id,)).fetchone()
        if row is None:
            raise ValueError("Connection not found")
        return self._connection(row)

    def _connection(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["models"] = self._json(data.pop("models_json"), [])
        data["has_api_key"] = bool(self.get_connection_key(data))
        env_name = data.get("api_key_env", "")
        secret_storage = self.secrets.storage(data["id"])
        if secret_storage != "none":
            data["key_storage"] = secret_storage
        elif env_name:
            import os

            data["key_storage"] = "environment" if os.environ.get(env_name, "").strip() else "none"
        else:
            data["key_storage"] = "none"
        data.pop("secret_persistent", None)
        return data

    def get_connection_key(self, connection: dict[str, Any]) -> str:
        value = self.secrets.cache.get(connection["id"], "")
        if value:
            return value
        if connection.get("secret_persistent"):
            value = self.secrets.get(connection["id"])
            if value:
                return value
        env_name = connection.get("api_key_env", "")
        if env_name:
            import os

            return os.environ.get(env_name, "").strip()
        return ""

    def save_connection(self, payload: dict[str, Any]) -> dict[str, Any]:
        connection_id = payload.get("id") or uuid.uuid4().hex
        existing = None
        try:
            existing = self.get_connection(connection_id)
        except ValueError:
            pass
        name = payload.get("name", existing["name"] if existing else "")
        base_url = payload.get("base_url", existing["base_url"] if existing else "")
        protocol = payload.get("protocol", existing["protocol"] if existing else "chat")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
            raise ValueError("Connection name must contain between 1 and 100 characters")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Connection base URL is required")
        if protocol not in ("chat", "responses", "anthropic", "opencode-go"):
            raise ValueError("Unsupported connection protocol")
        connection_changed = bool(
            existing
            and (
                base_url.strip().rstrip("/") != existing["base_url"]
                or protocol != existing["protocol"]
            )
        )
        models = payload.get("models", [] if connection_changed else (existing["models"] if existing else []))
        if not isinstance(models, list) or any(not isinstance(model, str) for model in models):
            raise ValueError("Connection models must be a string array")
        default_model = payload.get("default_model", "" if connection_changed else (existing["default_model"] if existing else ""))
        if not isinstance(default_model, str):
            raise ValueError("Connection default model must be a string")
        default_model = default_model.strip()
        # A default naming a model this profile cannot reach is worse than no default at all: it
        # would be silently discarded at every chat start. Refuse it while the caller can still
        # see why, but let a re-pointed endpoint drop its stale one without an error.
        if default_model and default_model not in models:
            raise ValueError("The default model is not in this connection's synced models")
        default_reasoning = payload.get("default_reasoning", "" if connection_changed else (existing["default_reasoning"] if existing else ""))
        if not isinstance(default_reasoning, str):
            raise ValueError("Connection default reasoning must be a string")
        default_reasoning = default_reasoning.strip().lower()
        if default_reasoning and default_reasoning not in REASONING_EFFORTS:
            raise ValueError("Unsupported reasoning effort")
        now = utc_now()
        created_at = existing["created_at"] if existing else now
        api_key_env = existing.get("api_key_env", "") if existing else ""
        with self.lock, self._connect() as db:
            db.execute(
                """INSERT INTO connections
                   (id, name, base_url, protocol, models_json, api_key_env, secret_persistent,
                    default_model, default_reasoning, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET name=excluded.name, base_url=excluded.base_url,
                   protocol=excluded.protocol, models_json=excluded.models_json,
                   default_model=excluded.default_model, default_reasoning=excluded.default_reasoning,
                   updated_at=excluded.updated_at""",
                (
                    connection_id,
                    name.strip(),
                    base_url.strip().rstrip("/"),
                    protocol,
                    json.dumps(sorted(set(models))),
                    api_key_env,
                    0,
                    default_model,
                    default_reasoning,
                    created_at,
                    now,
                ),
            )
        api_key = payload.get("api_key")
        if api_key is not None:
            if not isinstance(api_key, str) or len(api_key) > 20_000:
                raise ValueError("API key must be a string no longer than 20,000 characters")
            if api_key.strip():
                persistent = self.secrets.set(connection_id, api_key.strip(), name.strip())
                with self.lock, self._connect() as db:
                    db.execute(
                        "UPDATE connections SET secret_persistent = ? WHERE id = ?",
                        (int(persistent), connection_id),
                    )
        if payload.get("clear_api_key") is True:
            self.secrets.delete(connection_id)
            with self.lock, self._connect() as db:
                db.execute("UPDATE connections SET secret_persistent = 0 WHERE id = ?", (connection_id,))
        return self.get_connection(connection_id)

    def update_connection_models(self, connection_id: str, models: list[str]) -> dict[str, Any]:
        connection = self.get_connection(connection_id)
        # A sync that retires the default model leaves the profile pointing at nothing the
        # provider still serves, so the pointer goes with it rather than failing a chat later.
        default_model = connection["default_model"] if connection["default_model"] in models else ""
        with self.lock, self._connect() as db:
            db.execute(
                "UPDATE connections SET models_json = ?, default_model = ?, updated_at = ? WHERE id = ?",
                (json.dumps(sorted(set(models))), default_model, utc_now(), connection_id),
            )
        return self.get_connection(connection_id)

    def delete_connection(self, connection_id: str) -> None:
        self.get_connection(connection_id)
        with self.lock, self._connect() as db:
            db.execute("UPDATE images SET connection_id = NULL WHERE connection_id = ?", (connection_id,))
            db.execute("DELETE FROM connections WHERE id = ?", (connection_id,))
        self.secrets.delete(connection_id)

    # Styles are global rather than board-scoped: a saved look is meant to be reachable from a
    # board that did not exist when it was captured. Which styles are *active* is per board, and
    # lives in that board's settings alongside its LoRA stack.
    @staticmethod
    def _validate_style(name: Any, description: Any, style_text: Any) -> tuple[str, str, str]:
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError("Style name must contain between 1 and 80 characters")
        if not isinstance(description, str) or len(description) > 500:
            raise ValueError("Style description is limited to 500 characters")
        if not isinstance(style_text, str) or not 1 <= len(style_text.strip()) <= 2_000:
            raise ValueError("Style text must contain between 1 and 2,000 characters")
        return name.strip(), description.strip(), " ".join(style_text.split())

    def list_styles(self) -> list[dict[str, Any]]:
        with self.lock, self._connect() as db:
            rows = db.execute("SELECT * FROM styles ORDER BY name COLLATE NOCASE").fetchall()
        return [dict(row) for row in rows]

    def get_style(self, style_id: str) -> dict[str, Any]:
        with self.lock, self._connect() as db:
            row = db.execute("SELECT * FROM styles WHERE id = ?", (style_id,)).fetchone()
        if row is None:
            raise ValueError("Style not found")
        return dict(row)

    def create_style(
        self,
        name: str,
        description: str,
        style_text: str,
        reference_image_id: str | None = None,
        source: str = "user",
        kind: str = "art",
    ) -> dict[str, Any]:
        name, description, style_text = self._validate_style(name, description, style_text)
        if kind not in ("art", "scene"):
            raise ValueError("Style kind must be art or scene")
        if source not in ("user", "agent"):
            raise ValueError("Style source must be user or agent")
        style_id = uuid.uuid4().hex
        now = utc_now()
        with self.lock, self._connect() as db:
            if reference_image_id is not None:
                if db.execute("SELECT 1 FROM images WHERE id = ?", (reference_image_id,)).fetchone() is None:
                    raise ValueError("Reference image not found")
            if db.execute("SELECT 1 FROM styles WHERE name = ? COLLATE NOCASE", (name,)).fetchone():
                raise ValueError(f"A style named {name} already exists")
            db.execute(
                """INSERT INTO styles
                   (id, name, description, style_text, reference_image_id, source, created_at, updated_at, kind)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (style_id, name, description, style_text, reference_image_id, source, now, now, kind),
            )
        return self.get_style(style_id)

    def update_style(self, style_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.get_style(style_id)
        name, description, style_text = self._validate_style(
            payload.get("name", current["name"]),
            payload.get("description", current["description"]),
            payload.get("style_text", current["style_text"]),
        )
        kind = payload.get("kind", current["kind"])
        if kind not in ("art", "scene"):
            raise ValueError("Style kind must be art or scene")
        reference_image_id = payload.get("reference_image_id", current["reference_image_id"])
        with self.lock, self._connect() as db:
            if reference_image_id is not None:
                if db.execute("SELECT 1 FROM images WHERE id = ?", (reference_image_id,)).fetchone() is None:
                    raise ValueError("Reference image not found")
            clash = db.execute(
                "SELECT 1 FROM styles WHERE name = ? COLLATE NOCASE AND id != ?", (name, style_id)
            ).fetchone()
            if clash:
                raise ValueError(f"A style named {name} already exists")
            db.execute(
                """UPDATE styles SET name = ?, description = ?, style_text = ?,
                   reference_image_id = ?, updated_at = ?, kind = ? WHERE id = ?""",
                (name, description, style_text, reference_image_id, utc_now(), kind, style_id),
            )
        return self.get_style(style_id)

    def delete_style(self, style_id: str) -> None:
        self.get_style(style_id)
        with self.lock, self._connect() as db:
            db.execute("DELETE FROM styles WHERE id = ?", (style_id,))

    # A LoRA is a file on disk, not a row: the filesystem stays the source of truth for which ones
    # exist, and discover_loras finds them whether or not anything here knows their name. A profile
    # only decorates one — a cover frame, a friendlier label, a default strength, the user's own
    # notes — so a missing row is the normal case and reads as an entry with every field empty.
    # Nothing is written until the user edits something, and deleting a LoRA file just strands a
    # row that no listing will ever join against again.
    @staticmethod
    def _validate_lora_profile(
        display_name: Any, notes: Any, default_strength: Any
    ) -> tuple[str, str, float]:
        if not isinstance(display_name, str) or len(display_name) > 80:
            raise ValueError("LoRA display name is limited to 80 characters")
        if not isinstance(notes, str) or len(notes) > 2_000:
            raise ValueError("LoRA notes are limited to 2,000 characters")
        try:
            strength = float(default_strength)
        except (TypeError, ValueError) as exc:
            raise ValueError("LoRA default strength must be a number") from exc
        # The real bound is offcut_cli.LORA_STRENGTH_LIMIT and the caller enforces it, so that the
        # limit keeps living in one place. This only rejects what SQLite cannot store meaningfully.
        if not math.isfinite(strength):
            raise ValueError("LoRA default strength must be a finite number")
        return display_name.strip(), notes.strip(), strength

    def list_lora_profiles(self) -> list[dict[str, Any]]:
        with self.lock, self._connect() as db:
            rows = db.execute("SELECT * FROM lora_profiles ORDER BY name COLLATE NOCASE").fetchall()
        return [dict(row) for row in rows]

    def get_lora_profile(self, name: str) -> dict[str, Any] | None:
        with self.lock, self._connect() as db:
            row = db.execute("SELECT * FROM lora_profiles WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def save_lora_profile(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 200:
            raise ValueError("LoRA name must contain between 1 and 200 characters")
        name = name.strip()
        current = self.get_lora_profile(name) or {
            "display_name": "",
            "notes": "",
            "default_strength": DEFAULT_LORA_STRENGTH,
            "cover_image_id": None,
        }
        display_name, notes, default_strength = self._validate_lora_profile(
            payload.get("display_name", current["display_name"]),
            payload.get("notes", current["notes"]),
            payload.get("default_strength", current["default_strength"]),
        )
        cover_image_id = payload.get("cover_image_id", current["cover_image_id"]) or None
        now = utc_now()
        with self.lock, self._connect() as db:
            if cover_image_id is not None:
                if db.execute("SELECT 1 FROM images WHERE id = ?", (cover_image_id,)).fetchone() is None:
                    raise ValueError("Cover image not found")
            db.execute(
                """INSERT INTO lora_profiles
                   (name, display_name, notes, default_strength, cover_image_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                       display_name = excluded.display_name,
                       notes = excluded.notes,
                       default_strength = excluded.default_strength,
                       cover_image_id = excluded.cover_image_id,
                       updated_at = excluded.updated_at""",
                (name, display_name, notes, default_strength, cover_image_id, now, now),
            )
        return self.get_lora_profile(name)


    def record_chat_attempt(self, chat_id, image_id, recipe, change_note="", reused=False):
        with self.lock, self._connect() as db:
            cursor = db.execute(
                "INSERT INTO chat_attempts (session_id, image_id, recipe_json, change_note, reused, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (chat_id, image_id, json.dumps(recipe), change_note, int(reused), utc_now()),
            )
            return cursor.lastrowid

    def list_chat_attempts(self, chat_id, limit=12):
        with self.lock, self._connect() as db:
            rows = db.execute("SELECT * FROM chat_attempts WHERE session_id = ? ORDER BY id DESC LIMIT ?", (chat_id, limit)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["recipe"] = self._json(item.pop("recipe_json"), {})
            item["reused"] = bool(item["reused"])
            result.append(item)
        return result

    def review_chat_attempt(self, chat_id, attempt_id, observation, outcome):
        if not isinstance(observation, str) or not 1 <= len(observation) <= 2000:
            raise ValueError("Observation must contain between 1 and 2,000 characters")
        if outcome not in ("improved", "regressed", "mixed", "unclear"):
            raise ValueError("Outcome must be improved, regressed, mixed, or unclear")
        with self.lock, self._connect() as db:
            if db.execute("UPDATE chat_attempts SET observation = ?, outcome = ? WHERE session_id = ? AND id = ?",
                          (observation, outcome, chat_id, attempt_id)).rowcount != 1:
                raise ValueError("Attempt not found in this chat")

    def save_chat_turn(self, turn_id, chat_id, manifest):
        with self.lock, self._connect() as db:
            db.execute("INSERT INTO chat_turns (id, session_id, manifest_json, created_at) VALUES (?, ?, ?, ?) "
                       "ON CONFLICT(id) DO UPDATE SET manifest_json = excluded.manifest_json",
                       (turn_id, chat_id, json.dumps(manifest), utc_now()))

    def list_chat_turns(self, chat_id, limit=20):
        with self.lock, self._connect() as db:
            rows = db.execute("SELECT id, manifest_json, created_at FROM chat_turns WHERE session_id = ? ORDER BY created_at DESC LIMIT ?", (chat_id, limit)).fetchall()
        return [{"id": row["id"], "created_at": row["created_at"], **self._json(row["manifest_json"], {})} for row in rows]
