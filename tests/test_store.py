import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from offcut_store import Store


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp_dir.name) / "offcut.sqlite3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_board_image_lifecycle(self):
        board = self.store.create_board("Logo Study", "Marks for the app")
        image = self.store.record_image(
            board["id"],
            {
                "output_path": str(Path(self.temp_dir.name) / "image.png"),
                "raw_prompt": "a sharp fox mark",
                "enhanced_prompt": "a sharp geometric fox mark",
                "final_prompt": "ink style, a sharp geometric fox mark",
                "preset": "raw-int8-turbo-lora",
                "width": 1024,
                "height": 1024,
                "seed": 42,
                "steps": 8,
                "guidance": 0.0,
                "loras": [],
            },
            {
                "prompt": "a sharp fox mark",
                "loras": [{"name": "offcut_test", "strength": 0.8}],
                "enhance": True,
            },
            enhanced_prompt="a sharp geometric fox mark",
            connection_id="connection",
            enhancer_model="model",
        )
        self.assertEqual(image["board_id"], board["id"])
        self.assertEqual(self.store.get_board(board["id"])["cover_ids"], [image["id"]])
        self.assertEqual(self.store.list_images(query="geometric")[0]["id"], image["id"])
        favorite = self.store.update_image(image["id"], {"favorite": True})
        self.assertTrue(favorite["favorite"])
        moved = self.store.update_image(image["id"], {"board_id": "inbox"})
        self.assertEqual(moved["board_id"], "inbox")

    def test_board_settings_revision_guards_draft_updates(self):
        board = self.store.create_board("Draft")
        self.assertEqual(board["settings_revision"], 0)

        renamed = self.store.update_board(board["id"], {"name": "Renamed draft"})
        self.assertEqual(renamed["settings_revision"], 0)
        first = self.store.update_board(
            board["id"],
            {"settings": {"prompt": "first"}, "expected_settings_revision": 0},
        )
        self.assertEqual(first["settings_revision"], 1)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.store.update_board(
                board["id"],
                {"settings": {"prompt": "stale"}, "expected_settings_revision": 0},
            )
        self.assertEqual(self.store.get_board(board["id"])["settings"], {"prompt": "first"})

        unguarded = self.store.update_board(board["id"], {"settings": {"prompt": "second"}})
        self.assertEqual(unguarded["settings_revision"], 2)

    def test_record_image_updates_settings_revision(self):
        board = self.store.create_board("Generated")
        self.store.record_image(
            board["id"],
            {
                "output_path": str(Path(self.temp_dir.name) / "revision.png"),
                "width": 512,
                "height": 512,
                "seed": 7,
                "steps": 8,
                "guidance": 0,
            },
            {"prompt": "persist me", "width": 512},
        )
        updated = self.store.get_board(board["id"])
        self.assertEqual(updated["settings_revision"], 1)
        self.assertEqual(updated["settings"], {"prompt": "persist me", "width": 512})

    def test_existing_boards_schema_gets_settings_revision_once(self):
        path = Path(self.temp_dir.name) / "legacy.sqlite3"
        with sqlite3.connect(path) as db:
            db.execute(
                """CREATE TABLE boards (
                       id TEXT PRIMARY KEY, name TEXT NOT NULL,
                       description TEXT NOT NULL DEFAULT '',
                       settings_json TEXT NOT NULL DEFAULT '{}',
                       created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                   )"""
            )
            db.execute(
                "INSERT INTO boards VALUES ('legacy', 'Legacy', '', '{}', 'now', 'now')"
            )

        migrated = Store(path)
        self.assertEqual(migrated.get_board("legacy")["settings_revision"], 0)
        reopened = Store(path)
        with sqlite3.connect(path) as db:
            columns = [row[1] for row in db.execute("PRAGMA table_info(boards)")]
        self.assertEqual(columns.count("settings_revision"), 1)
        self.assertEqual(reopened.get_board("legacy")["settings_revision"], 0)

    def test_existing_image_and_chat_schemas_get_new_context_columns_once(self):
        path = Path(self.temp_dir.name) / "legacy-context.sqlite3"
        Store(path)
        with sqlite3.connect(path) as db:
            db.execute("ALTER TABLE images RENAME TO old_images")
            db.execute(
                """CREATE TABLE images AS SELECT id, board_id, path, raw_prompt, enhanced_prompt,
                          final_prompt, preset, width, height, seed, steps, guidance, negative_prompt,
                          loras_json, enhance, connection_id, enhancer_model, favorite, position,
                          metadata_json, created_at FROM old_images"""
            )
            db.execute("DROP TABLE old_images")
            db.execute("ALTER TABLE chat_sessions RENAME TO old_chat_sessions")
            db.execute(
                """CREATE TABLE chat_sessions AS SELECT id, board_id, title, connection_id, model,
                          permission_mode, summary, compacted_before, created_at, updated_at
                   FROM old_chat_sessions"""
            )
            db.execute("DROP TABLE old_chat_sessions")

        Store(path)
        Store(path)
        with sqlite3.connect(path) as db:
            image_columns = [row[1] for row in db.execute("PRAGMA table_info(images)")]
            chat_columns = [row[1] for row in db.execute("PRAGMA table_info(chat_sessions)")]
        self.assertEqual(image_columns.count("kind"), 1)
        self.assertEqual(image_columns.count("run_key"), 1)
        self.assertEqual(chat_columns.count("reasoning_effort"), 1)
        self.assertEqual(chat_columns.count("workspace_revision"), 1)
        self.assertEqual(chat_columns.count("workspace_prompt"), 1)

    def record_run(self, board_id, name, run_key, kind="generated"):
        return self.store.record_image(
            board_id,
            {
                "output_path": str(Path(self.temp_dir.name) / f"{name}.png"),
                "raw_prompt": "a sharp fox mark",
                "final_prompt": "a sharp fox mark",
                "preset": "turbo-int8",
                "width": 1024,
                "height": 1024,
                "seed": 42,
                "steps": 8,
                "guidance": 0.0,
            },
            {"prompt": "a sharp fox mark"},
            kind=kind,
            run_key=run_key,
        )

    def test_find_run_matches_an_earlier_frame_from_the_same_inputs(self):
        board = self.store.create_board("Studies")
        image = self.record_run(board["id"], "first", "key-a")
        self.record_run(board["id"], "second", "key-b")

        found = self.store.find_run(board["id"], "key-a")
        self.assertEqual(found["id"], image["id"])
        self.assertIsNone(self.store.find_run(board["id"], "key-c"))

    def test_find_run_returns_the_oldest_frame_when_several_share_a_key(self):
        board = self.store.create_board("Studies")
        # Rows predating the key, or written while it was bypassed, can collide once it exists.
        first = self.record_run(board["id"], "first", "key-a")
        self.record_run(board["id"], "second", "key-a")

        self.assertEqual(self.store.find_run(board["id"], "key-a")["id"], first["id"])

    def test_find_run_never_matches_the_empty_key(self):
        board = self.store.create_board("Studies")
        # What every row written before the column existed carries.
        self.record_run(board["id"], "legacy", "")

        self.assertIsNone(self.store.find_run(board["id"], ""))

    def test_find_run_is_scoped_to_one_board_and_one_kind(self):
        board = self.store.create_board("Studies")
        other = self.store.create_board("Covers")
        self.record_run(board["id"], "generated", "key-a")
        self.record_run(board["id"], "cover", "key-a", kind="cover")

        self.assertIsNone(self.store.find_run(other["id"], "key-a"))
        self.assertEqual(self.store.find_run(board["id"], "key-a", "cover")["kind"], "cover")

    def test_deleting_board_moves_images_to_inbox(self):
        board = self.store.create_board("Temporary")
        image = self.store.record_image(
            board["id"],
            {
                "output_path": str(Path(self.temp_dir.name) / "move.png"),
                "width": 512,
                "height": 512,
                "seed": 1,
                "steps": 8,
                "guidance": 0,
            },
            {"prompt": "test"},
        )
        self.store.delete_board(board["id"])
        self.assertEqual(self.store.get_image(image["id"])["board_id"], "inbox")

    def test_deleting_an_image_leaves_the_board_and_its_neighbours(self):
        board = self.store.create_board("Temporary")
        kept, removed = (
            self.store.record_image(
                board["id"],
                {
                    "output_path": str(Path(self.temp_dir.name) / f"{name}.png"),
                    "width": 512,
                    "height": 512,
                    "seed": seed,
                    "steps": 8,
                    "guidance": 0,
                },
                {"prompt": "test"},
            )
            for name, seed in (("kept", 1), ("removed", 2))
        )
        record = self.store.delete_image(removed["id"])
        self.assertEqual(record["path"], str(Path(self.temp_dir.name) / "removed.png"))
        with self.assertRaises(ValueError):
            self.store.get_image(removed["id"])
        self.assertEqual([image["id"] for image in self.store.list_images(board["id"])], [kept["id"]])
        self.assertEqual(self.store.get_board(board["id"])["image_count"], 1)

    def test_deleting_a_missing_image_is_refused(self):
        with self.assertRaises(ValueError):
            self.store.delete_image("nothing")

    def test_partial_reorder_is_rejected(self):
        board = self.store.create_board("Ordering")
        images = []
        for seed in (1, 2):
            images.append(
                self.store.record_image(
                    board["id"],
                    {
                        "output_path": str(Path(self.temp_dir.name) / f"{seed}.png"),
                        "width": 512,
                        "height": 512,
                        "seed": seed,
                        "steps": 8,
                        "guidance": 0,
                    },
                    {"prompt": "test"},
                )
            )
        with self.assertRaises(ValueError):
            self.store.reorder_images(board["id"], [images[0]["id"]])

    def test_connection_secret_is_not_in_database_record(self):
        with patch.object(self.store.secrets, "set") as secret_set, patch.object(
            self.store.secrets, "get", return_value="stored-secret"
        ):
            connection = self.store.save_connection(
                {
                    "name": "Local",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "protocol": "chat",
                    "api_key": "stored-secret",
                }
            )
        secret_set.assert_called_once()
        self.assertTrue(connection["has_api_key"])
        self.assertNotIn("api_key", connection)

    def test_connection_secret_survives_store_restart(self):
        self.store.save_connection({"id": "opencode-go", "api_key": "stored-secret"})

        reopened = Store(self.store.path)
        connection = reopened.get_connection("opencode-go")

        self.assertEqual(reopened.get_connection_key(connection), "stored-secret")
        self.assertEqual(connection["key_storage"], "local-file")
        self.assertEqual(reopened.secrets.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(reopened.secrets.path.parent.stat().st_mode & 0o777, 0o700)

    def test_connection_endpoint_change_clears_stale_models(self):
        with patch.object(self.store.secrets, "get", return_value=""):
            connection = self.store.save_connection(
                {
                    "name": "Provider",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "protocol": "chat",
                    "models": ["old-model"],
                }
            )
            updated = self.store.save_connection(
                {"id": connection["id"], "base_url": "http://127.0.0.1:9000/v1"}
            )
        self.assertEqual(updated["models"], [])

    def test_connection_defaults_round_trip_and_reject_unreachable_model(self):
        with patch.object(self.store.secrets, "get", return_value=""):
            connection = self.store.save_connection(
                {
                    "name": "Provider",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "protocol": "chat",
                    "models": ["fast-model", "smart-model"],
                    "default_model": "smart-model",
                    "default_reasoning": "HIGH",
                }
            )
            self.assertEqual(connection["default_model"], "smart-model")
            self.assertEqual(connection["default_reasoning"], "high")

            with self.assertRaises(ValueError):
                self.store.save_connection({"id": connection["id"], "default_model": "absent-model"})
            with self.assertRaises(ValueError):
                self.store.save_connection({"id": connection["id"], "default_reasoning": "ludicrous"})

            kept = self.store.save_connection({"id": connection["id"], "name": "Renamed"})
        self.assertEqual(kept["default_model"], "smart-model")
        self.assertEqual(kept["default_reasoning"], "high")

    def test_sync_drops_a_default_model_the_provider_retired(self):
        with patch.object(self.store.secrets, "get", return_value=""):
            connection = self.store.save_connection(
                {
                    "name": "Provider",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "protocol": "chat",
                    "models": ["fast-model", "smart-model"],
                    "default_model": "smart-model",
                    "default_reasoning": "high",
                }
            )
            resynced = self.store.update_connection_models(connection["id"], ["fast-model"])
        self.assertEqual(resynced["default_model"], "")
        # The effort preference is not model-specific, so it outlives the model it was set beside.
        self.assertEqual(resynced["default_reasoning"], "high")

    def test_chat_lifecycle_messages_compaction_and_summaries(self):
        board = self.store.create_board("Agent board")
        chat = self.store.create_chat(board["id"], "opencode-go", "glm-5.2", "draft")
        self.assertEqual(chat["title"], "New chat")
        self.assertEqual(chat["message_count"], 0)
        self.assertEqual(chat["latest_preview"], "")

        first_batch = self.store.append_chat_messages(
            chat["id"],
            [
                {"role": "user", "content": "Make a geometric fox"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I prepared the first draft."}],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "total_tokens": 30,
                        "cost": 0.025,
                    },
                },
            ],
        )
        second_batch = self.store.append_chat_messages(
            chat["id"],
            [
                {
                    "role": "assistant",
                    "content": "Here is the revision.",
                    "usage": {"total_tokens": 5, "cost": {"total": 0.01}},
                },
                {"role": "toolResult", "content": "internal result"},
            ],
        )
        self.assertEqual([message["sequence"] for message in first_batch + second_batch], [1, 2, 3, 4])
        messages = self.store.list_chat_messages(chat["id"])
        self.assertEqual(
            [message["role"] for message in messages],
            ["user", "assistant", "assistant", "toolResult"],
        )
        self.assertTrue(all({"id", "sequence", "created_at"} <= message.keys() for message in messages))

        updated = self.store.update_chat(
            chat["id"],
            {
                "title": "Fox exploration",
                "connection_id": None,
                "model": "new-model",
                "permission_mode": "iterate",
                "summary": "A fox mark is being refined.",
                "compacted_before": 2,
            },
        )
        self.assertEqual(updated["title"], "Fox exploration")
        self.assertIsNone(updated["connection_id"])
        self.assertEqual(updated["permission_mode"], "create")
        self.assertEqual(updated["summary"], "A fox mark is being refined.")
        self.assertEqual(updated["message_count"], 4)
        self.assertEqual(updated["latest_preview"], "Here is the revision.")
        self.assertEqual(updated["total_usage"]["total_tokens"], 35)
        self.assertEqual(updated["total_cost"], 0.035)
        self.assertEqual(
            [message["sequence"] for message in self.store.list_chat_messages(chat["id"])],
            [3, 4],
        )
        self.assertEqual(
            [message["sequence"] for message in self.store.list_chat_messages(chat["id"], True)],
            [1, 2, 3, 4],
        )
        # The watermark a running agent turn captures so the transcript can hold its own
        # messages back for the event replay.
        self.assertEqual(self.store.chat_message_watermark(chat["id"]), 4)
        self.assertEqual(self.store.list_chats(board["id"])[0]["id"], chat["id"])

        self.store.delete_chat(chat["id"])
        with self.assertRaisesRegex(ValueError, "not found"):
            self.store.get_chat(chat["id"])
        with sqlite3.connect(self.store.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0], 0)

    def test_chats_are_board_local_and_cascade_with_board(self):
        first_board = self.store.create_board("First")
        second_board = self.store.create_board("Second")
        first_chat = self.store.create_chat(first_board["id"], None, "model", "create")
        second_chat = self.store.create_chat(second_board["id"], None, "model", "draft")
        self.store.append_chat_messages(first_chat["id"], [{"role": "user", "content": "private"}])

        self.assertEqual([chat["id"] for chat in self.store.list_chats(first_board["id"])], [first_chat["id"]])
        self.assertEqual([chat["id"] for chat in self.store.list_chats(second_board["id"])], [second_chat["id"]])
        self.store.delete_board(first_board["id"])
        with self.assertRaisesRegex(ValueError, "not found"):
            self.store.get_chat(first_chat["id"])
        self.assertEqual(self.store.get_chat(second_chat["id"])["board_id"], second_board["id"])
        with sqlite3.connect(self.store.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0], 0)

    def test_deleting_connection_nulls_chat_reference(self):
        connection = self.store.save_connection(
            {
                "name": "Disposable",
                "base_url": "http://127.0.0.1:9000/v1",
                "protocol": "chat",
            }
        )
        board = self.store.create_board("Connection board")
        chat = self.store.create_chat(board["id"], connection["id"], "model", "draft")

        self.store.delete_connection(connection["id"])

        self.assertIsNone(self.store.get_chat(chat["id"])["connection_id"])

    def test_chat_validation_rejects_invalid_values(self):
        board = self.store.create_board("Validation")
        with self.assertRaises(ValueError):
            self.store.create_chat(board["id"], None, "model", "unrestricted")
        chat = self.store.create_chat(board["id"], None, "model", "draft")
        with self.assertRaises(ValueError):
            self.store.append_chat_messages(chat["id"], ["not an object"])
        with self.assertRaises(ValueError):
            self.store.update_chat(chat["id"], {"compacted_before": -1})


if __name__ == "__main__":
    unittest.main()


class StyleStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp_dir.name) / "offcut.sqlite3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_image(self, board_id, name="reference.png"):
        return self.store.record_image(
            board_id,
            {
                "output_path": str(Path(self.temp_dir.name) / name),
                "raw_prompt": "a harbour at dawn",
                "enhanced_prompt": "",
                "final_prompt": "a harbour at dawn",
                "preset": "turbo-int8",
                "width": 1024,
                "height": 1024,
                "seed": 7,
                "steps": 8,
                "guidance": 0.0,
                "loras": [],
            },
            {"prompt": "a harbour at dawn"},
        )

    def test_style_lifecycle(self):
        board = self.store.create_board("Styles")
        image = self.make_image(board["id"])
        style = self.store.create_style(
            "Deco Travel Poster",
            "For flat, graphic, poster-like scenes",
            "1920s travel-poster lithography,   stippled ink texture,\n peach and cobalt palette",
            image["id"],
        )
        # Whitespace is collapsed because the text is prefixed onto a prompt, where a stray
        # newline would land in the middle of the final string.
        self.assertEqual(
            style["style_text"],
            "1920s travel-poster lithography, stippled ink texture, peach and cobalt palette",
        )
        self.assertEqual(style["source"], "user")
        self.assertEqual(style["reference_image_id"], image["id"])
        self.assertEqual([item["id"] for item in self.store.list_styles()], [style["id"]])

        renamed = self.store.update_style(style["id"], {"name": "Deco Poster"})
        self.assertEqual(renamed["name"], "Deco Poster")
        self.assertEqual(renamed["style_text"], style["style_text"])

        self.store.delete_style(style["id"])
        self.assertEqual(self.store.list_styles(), [])
        with self.assertRaises(ValueError):
            self.store.get_style(style["id"])

    def test_style_names_are_unique_case_insensitively(self):
        self.store.create_style("Muted Risograph", "", "duotone risograph print, misregistered ink")
        with self.assertRaises(ValueError):
            self.store.create_style("muted risograph", "", "something else entirely")

    def test_style_rejects_missing_reference_and_empty_text(self):
        with self.assertRaises(ValueError):
            self.store.create_style("Ghost", "", "a style", "no-such-image")
        with self.assertRaises(ValueError):
            self.store.create_style("Blank", "", "   ")

    def test_lora_profile_decorates_a_file_without_owning_it(self):
        # A LoRA with no profile is the normal case, not a missing row to repair.
        self.assertIsNone(self.store.get_lora_profile("offcut_style"))
        self.assertEqual(self.store.list_lora_profiles(), [])

        board = self.store.create_board("Loras")
        image = self.make_image(board["id"])
        profile = self.store.save_lora_profile(
            "offcut_style",
            {"display_name": "Poster Maximalist", "default_strength": 0.9, "cover_image_id": image["id"]},
        )
        self.assertEqual(profile["display_name"], "Poster Maximalist")
        self.assertEqual(profile["default_strength"], 0.9)
        self.assertEqual(profile["cover_image_id"], image["id"])
        self.assertEqual(profile["notes"], "")

        # A patch touching one field leaves the rest of the profile alone.
        noted = self.store.save_lora_profile("offcut_style", {"notes": "Ease off below 0.8"})
        self.assertEqual(noted["notes"], "Ease off below 0.8")
        self.assertEqual(noted["display_name"], "Poster Maximalist")
        self.assertEqual(noted["default_strength"], 0.9)
        self.assertEqual(noted["created_at"], profile["created_at"])
        self.assertEqual([item["name"] for item in self.store.list_lora_profiles()], ["offcut_style"])

    def test_lora_profile_rejects_a_missing_cover_and_an_unusable_strength(self):
        with self.assertRaises(ValueError):
            self.store.save_lora_profile("offcut_style", {"cover_image_id": "no-such-image"})
        with self.assertRaises(ValueError):
            self.store.save_lora_profile("offcut_style", {"default_strength": float("inf")})
        with self.assertRaises(ValueError):
            self.store.save_lora_profile("offcut_style", {"display_name": "x" * 81})
        # None of the refusals may leave a half-written row behind.
        self.assertEqual(self.store.list_lora_profiles(), [])

    def test_a_removed_cover_image_leaves_the_lora_profile_intact(self):
        board = self.store.create_board("Loras")
        image = self.make_image(board["id"])
        self.store.save_lora_profile("offcut_style", {"display_name": "Grainy", "cover_image_id": image["id"]})
        self.store.delete_image(image["id"])
        reloaded = self.store.get_lora_profile("offcut_style")
        self.assertIsNone(reloaded["cover_image_id"])
        self.assertEqual(reloaded["display_name"], "Grainy")

    def test_a_removed_reference_image_leaves_the_style_intact(self):
        # A style is the text, and losing its reference must never lose the style.
        board = self.store.create_board("Styles")
        image = self.make_image(board["id"])
        style = self.store.create_style("Grainy Film", "", "35mm film grain, halation, muted kodachrome", image["id"])
        self.store.delete_image(image["id"])
        reloaded = self.store.get_style(style["id"])
        self.assertIsNone(reloaded["reference_image_id"])
        self.assertEqual(reloaded["style_text"], "35mm film grain, halation, muted kodachrome")


class CoverImageKindTests(unittest.TestCase):
    """A cover is a real image row so cover_image_id has something to point at, but it belongs to
    a library entry rather than to a board: it stays out of every listing and never rewrites the
    settings of whatever board happened to receive it."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp_dir.name) / "offcut.sqlite3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _record(self, name, kind, prompt):
        return self.store.record_image(
            "inbox",
            {
                "output_path": str(Path(self.temp_dir.name) / name),
                "raw_prompt": prompt,
                "preset": "turbo-int8",
                "width": 1024,
                "height": 1024,
                "seed": 7,
                "steps": 8,
                "guidance": 0.0,
            },
            {"prompt": prompt, "width": 1024, "height": 1024},
            kind=kind,
        )

    def test_covers_are_left_out_of_listings_unless_asked_for(self):
        self._record("frame.png", "generated", "a working frame")
        cover = self._record("cover.png", "cover", "the cover recipe")
        listed = self.store.list_images()
        self.assertEqual([image["raw_prompt"] for image in listed], ["a working frame"])
        self.assertEqual([image["raw_prompt"] for image in self.store.list_images(board_id="inbox")], ["a working frame"])
        covers = self.store.list_images(kinds=("cover",))
        self.assertEqual([image["id"] for image in covers], [cover["id"]])
        # It is still addressable by id, which is what a cover_image_id join needs.
        self.assertEqual(self.store.get_image(cover["id"])["kind"], "cover")

    def test_a_cover_does_not_rewrite_the_board_it_landed_on(self):
        self._record("frame.png", "generated", "a working frame")
        before = self.store.get_board("inbox")
        self._record("cover.png", "cover", "the cover recipe")
        after = self.store.get_board("inbox")
        self.assertEqual(after["settings"]["prompt"], "a working frame")
        self.assertEqual(after["settings_revision"], before["settings_revision"])
