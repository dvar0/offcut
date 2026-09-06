"""User-authored LoRA metadata stays local and reaches only its intended consumers."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import offcut_cli as cli
import offcut_server as server
from offcut_store import Store


class LoraMetadataTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = self.root / "settings.json"
        self.loras = self.root / "loras"
        self.loras.mkdir()
        self.file = self.loras / "my_ink_style.safetensors"
        self.file.touch()
        self.config.write_text(json.dumps({"lora_dirs": [str(self.loras)]}))
        self.store = Store(self.root / "test.sqlite3")
        self.enterContext(patch.dict(os.environ, {"OFFCUT_CONFIG": str(self.config)}))
        self.enterContext(patch.object(server, "STORE", self.store))

    def test_arbitrary_filename_is_listed_without_creating_a_profile(self):
        managed = self.loras / Path(cli.DOWNLOADS["turbo-lora"].relative_path).name
        managed.touch()
        (self.loras / "not-a-file.safetensors").mkdir()
        listed = server.public_loras()
        self.assertEqual([item["name"] for item in listed], [self.file.stem])
        self.assertEqual(listed[0]["trigger"], "")
        self.assertEqual(listed[0]["prompting_notes"], "")
        self.assertEqual(self.store.list_lora_profiles(), [])
        self.assertNotIn("lora_metadata", json.loads(self.config.read_text()))

    def test_editor_save_survives_reload_and_is_shared_with_cli(self):
        result = server.update_lora_profile(self.file.stem, {
            "display_name": "Ink", "default_strength": 0.65, "notes": "Library-only reminder",
            "trigger": " ink strokes ", "summary": "Ink illustration", "prompting_notes": "Try a short scene sentence.",
        })
        self.assertEqual(result["trigger"], "ink strokes")
        saved = cli.load_settings()
        self.assertEqual(cli.resolve_lora(saved, "my_ink_style:0.65")[2], "ink strokes")
        reloaded = server.public_loras()[0]
        self.assertEqual(reloaded["prompting_notes"], "Try a short scene sentence.")
        self.assertEqual(reloaded["default_strength"], 0.65)
        self.assertNotIn("prompting_notes", self.store.get_lora_profile(self.file.stem))
        self.assertNotIn("Library-only reminder", json.dumps(saved["lora_metadata"]))

    def test_partial_and_cover_updates_keep_metadata_and_other_adapters(self):
        server.update_lora_profile(self.file.stem, {"trigger": "ink strokes", "prompting_notes": "Use short captions."})
        second = self.loras / "other.safetensors"
        second.touch()
        server.update_lora_profile(second.stem, {"trigger": "chalk"})
        server.update_lora_profile(self.file.stem, {"cover_image_id": None, "display_name": "Ink"})
        server.update_lora_profile(self.file.stem, {"summary": "Black ink"})
        entries = cli.load_settings()["lora_metadata"]
        self.assertEqual(entries[self.file.stem]["trigger"], "ink strokes")
        self.assertEqual(entries[self.file.stem]["prompting_notes"], "Use short captions.")
        self.assertEqual(entries[second.stem]["trigger"], "chalk")

    def test_clearing_notes_and_trigger_removes_them_from_consumers(self):
        server.update_lora_profile(self.file.stem, {"trigger": "ink", "summary": "Ink", "prompting_notes": "Use short captions."})
        server.update_lora_profile(self.file.stem, {"trigger": "", "summary": "", "prompting_notes": ""})
        settings = cli.load_settings()
        self.assertIsNone(cli.resolve_lora(settings, "my_ink_style")[2])
        self.assertEqual(server.enhancer_lora_notes([self.file.stem], settings), [])
        self.assertEqual(server.public_loras()[0]["prompting_notes"], "")

    def test_invalid_edits_change_neither_store(self):
        before = self.config.read_bytes()
        for payload in ({"trigger": None}, {"prompting_notes": "x" * 12001}, {"summary": "x" * 301}, {"default_strength": True}, {"notes": "x" * 2001}, {"cover_image_id": "missing"}):
            with self.subTest(payload_keys=list(payload)):
                with self.assertRaises(ValueError):
                    server.update_lora_profile(self.file.stem, {"display_name": "Should not save", **payload})
                self.assertEqual(self.config.read_bytes(), before)
                self.assertEqual(self.store.list_lora_profiles(), [])
        with self.assertRaises(ValueError):
            server.update_lora_profile("missing", {"trigger": "ink"})

    def test_list_is_brief_and_inspection_gets_user_notes_not_library_notes(self):
        server.update_lora_profile(self.file.stem, {
            "summary": "Ink illustration", "prompting_notes": "Try short captions.",
            "notes": "Library-only reminder", "default_strength": 0.65,
        })
        board = self.store.create_board("Test")
        context = {"chat": {"permission_mode": "create", "board_id": board["id"]},
                   "settings": {"loras": [{"name": self.file.stem, "strength": 0.75}]}, "tool_lock": threading.Lock()}
        listing = server.execute_chat_tool(context, "list_loras", {})
        self.assertEqual(listing["loras"][0]["summary"], "Ink illustration")
        self.assertTrue(listing["loras"][0]["has_prompting_notes"])
        self.assertNotIn("Try short captions.", json.dumps(listing))
        inspection = server.execute_chat_tool(context, "inspect_lora", {"name": self.file.stem})
        self.assertEqual(inspection["prompting_notes"], "Try short captions.")
        self.assertEqual(inspection["default_strength"], 0.65)
        self.assertEqual(inspection["active_strength"], 0.75)
        self.assertNotIn("Library-only reminder", json.dumps(inspection))
        self.assertIn("User-provided", inspection["notes_source"])
        self.assertNotIn("Library-only reminder", json.dumps(server.enhancer_lora_notes([self.file.stem])))

    def test_cli_refuses_invalid_metadata_before_saving(self):
        from types import SimpleNamespace
        before = self.config.read_bytes()
        with self.assertRaises(ValueError):
            cli.run_settings(SimpleNamespace(settings_command="set", key="lora_metadata", value='{"my_ink_style": {"trigger": 42}}'), cli.load_settings())
        self.assertEqual(self.config.read_bytes(), before)
