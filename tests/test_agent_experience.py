"""Regressions from real multi-turn image work; no provider or inference calls."""
import base64
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import offcut_server as server
from offcut_store import Store


class AgentExperienceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "audit.sqlite3")
        for target, attribute, kwargs in (
            (server, "STORE", {"new": self.store}),
            (server, "validate_chat_connection", {"return_value": ({"name": "test", "base_url": "http://localhost/v1", "protocol": "chat"}, "vision-model")}),
            (server, "describe_chat_model", {"return_value": {"vision": True, "supportedThinkingLevels": ["high"], "recommendedThinkingLevel": "high"}}),
            (self.store, "get_connection_key", {"return_value": "secret-not-for-manifests"}),
            (server, "discover_loras", {"return_value": []}),
            (server.STATE, "generate", {"side_effect": AssertionError("Mock inference explicitly")}),
        ):
            mock = patch.object(target, attribute, **kwargs)
            mock.start()
            self.addCleanup(mock.stop)
        self.board = self.store.create_board("Work")
        self.chat = self.store.create_chat(self.board["id"], None, "vision-model", "create")

    def frame(self, board=None, seed=9007199254740993, prompt="a fox", reference=False):
        path = self.root / f"image-{len(list(self.root.glob('*.png')))}.png"
        image = Image.new("RGB", (100, 80), "red")
        image.paste("blue", (50, 0, 100, 80))
        image.save(path)
        board_id = (board or self.board)["id"]
        if reference:
            return self.store.record_reference_image(board_id, path, 100, 80, prompt)
        return self.store.record_image(board_id, {
            "output_path": str(path), "raw_prompt": prompt, "final_prompt": prompt,
            "preset": "turbo-int8", "width": 1024, "height": 1024, "seed": seed,
            "steps": 8, "guidance": 0.0, "loras": [],
        }, {"prompt": prompt, "seed": str(seed), "styles": []})

    def turn(self, attachments=(), selected=None, message="Refine the composition"):
        return server.prepare_chat_turn(self.chat["id"], {
            "message": message, "workspace_selected_image_id": selected,
            "selected_image_id": None, "attachment_ids": list(attachments),
            "workspace": {"board_id": self.board["id"]},
            "workspace_revision": self.store.get_board(self.board["id"])["settings_revision"],
        })

    def attach_history(self, images):
        self.store.append_chat_messages(self.chat["id"], [{"role": "user", "content": [
            {"type": "text", "text": "Use these references"},
            *[{"type": "imageRef", "imageId": image["id"]} for image in images],
        ]}])

    def test_old_reference_can_be_reopened_without_automatic_replay(self):
        image = self.frame(reference=True)
        self.attach_history([image])
        context = self.turn(selected=image["id"])
        self.assertEqual(context["images"], {})
        self.assertNotIn("imageRef", json.dumps(context["messages"]))
        result = server.execute_chat_tool(context, "get_selected_image", {})
        self.assertEqual(result["pixels_supplied"], [image["id"]])
        self.assertIn(image["id"], result["__krea2_images"])
        repeat = server.execute_chat_tool(context, "inspect_image", {"image_id": image["id"]})
        self.assertEqual(repeat["pixels_supplied"], [image["id"]])
        self.assertNotIn("__krea2_images", repeat)  # Already resident in this turn's bridge.

    def test_cross_board_reference_access_survives_compaction(self):
        other_board = self.store.create_board("Elsewhere")
        local, foreign, stranger = self.frame(reference=True), self.frame(other_board, reference=True), self.frame(other_board, reference=True)
        self.attach_history([foreign])
        self.store.update_chat(self.chat["id"], {"compacted_before": 1, "summary": "Use the reference"})
        context = self.turn()
        result = server.execute_chat_tool(context, "compare_images", {"image_ids": [local["id"], foreign["id"]]})
        self.assertEqual(result["pixels_supplied"], [local["id"], foreign["id"]])
        with self.assertRaises(ValueError):
            server.execute_chat_tool(context, "inspect_image", {"image_id": stranger["id"]})

    def test_crop_supplies_the_requested_detail_and_keeps_full_preview(self):
        image = self.frame(reference=True)
        context = self.turn([image["id"]])
        full = context["images"][image["id"]]
        result = server.execute_chat_tool(context, "inspect_image", {"image_id": image["id"], "crop": {"x": .5, "y": 0, "width": .5, "height": 1}})
        preview = result["__krea2_images"][result["preview_id"]]
        with Image.open(io.BytesIO(base64.b64decode(preview["data"]))) as crop:
            self.assertEqual(crop.size, (50, 80))
            self.assertGreater(crop.getpixel((20, 20))[2], 240)
        self.assertEqual(context["images"][image["id"]], full)
        with self.assertRaises(ValueError):
            server.execute_chat_tool(context, "inspect_image", {"image_id": image["id"], "crop": {"x": .9, "y": 0, "width": .5, "height": 1}})

    def test_missing_pixels_and_nonvision_are_explicit(self):
        image = self.frame(reference=True)
        Path(image["path"]).unlink()
        context = self.turn()
        result = server.execute_chat_tool(context, "inspect_image", {"image_id": image["id"]})
        self.assertEqual(result["pixels_supplied"], [])
        self.assertIn(image["id"], result["preview_errors"])
        context["vision"] = False
        self.assertEqual(server.execute_chat_tool(context, "inspect_image", {"image_id": image["id"]})["pixels_supplied"], [])

    def test_exact_recipe_restore_bakes_historical_style_without_double_prefix(self):
        image = self.frame(prompt="an archived ink style, a fox")
        context = self.turn()
        result = server.execute_chat_tool(context, "restore_image_settings", {"image_id": image["id"]})
        self.assertEqual(result["settings"]["seed"], "9007199254740993")
        self.assertEqual(result["settings"]["prompt"], image["final_prompt"])
        self.assertEqual(result["settings"]["styles"], [])
        self.assertFalse(result["settings"]["enhance"])
        self.assertFalse(result["generation_performed"])
        self.assertEqual(server.curated_image(image)["seed"], "9007199254740993")

    def test_seed_policy_and_active_styles_survive_recording(self):
        for requested in (None, "9007199254740993"):
            image = self.frame()
            request = {"prompt": "fox", "seed": requested, "styles": ["style-id"], "steps": None, "guidance": None}
            self.store.record_image(self.board["id"], {**image, "output_path": image["path"]}, request)
            settings = self.store.get_board(self.board["id"])["settings"]
            self.assertEqual(settings["seed"], requested)
            self.assertEqual(settings["styles"], ["style-id"])
            self.assertIsNone(settings["steps"])
            self.assertIsNone(settings["guidance"])

    def test_creative_brief_preserves_approval_and_reference_purpose(self):
        first, second = self.frame(), self.frame()
        context = self.turn()
        server.execute_chat_tool(context, "update_creative_brief", {"goal": "Landscape poster", "must_keep": ["two hills", "lake on the left"],
            "approved_image_id": first["id"], "best_image_id": first["id"], "references": [{"image_id": first["id"], "purpose": "layout"}]})
        server.execute_chat_tool(context, "update_creative_brief", {"best_image_id": second["id"], "next_change": "smaller cabin"})
        self.store.update_chat(self.chat["id"], {"summary": "Compacted", "compacted_before": 99})
        later = self.turn()
        brief = later["chat"]["creative_brief"]
        self.assertEqual(brief["approved_image_id"], first["id"])
        self.assertEqual(brief["best_image_id"], second["id"])
        self.assertEqual(brief["must_keep"], ["two hills", "lake on the left"])
        self.assertIn("smaller cabin", later["agent_prompt"])
        self.assertIn(first["id"], later["known_images"])

    def test_brief_rejects_unknown_images_and_oversized_notes(self):
        context = self.turn()
        for patch_value in ({"goal": "x" * 2001}, {"must_keep": ["x"] * 17}, {"best_image_id": "missing"}, {"unknown": "field"}):
            with self.assertRaises(ValueError):
                server.execute_chat_tool(context, "update_creative_brief", patch_value)

    def test_generation_allowance_attempt_history_and_explicit_fresh_seed(self):
        source = self.frame()
        context = self.turn()
        context["settings"].update(seed="9007199254740993", enhance=True)
        context["generation_limit"] = 2
        sent = []
        def generate(payload):
            sent.append(payload.copy())
            self.store.record_image(self.board["id"], {**source, "output_path": source["path"]}, payload)
            return {"image": source, "reused": len(sent) == 2}
        with patch.object(server.STATE, "generate", side_effect=generate):
            first = server.execute_chat_tool(context, "generate_image", {"change_note": "smaller cabin"})
            second = server.execute_chat_tool(context, "generate_image", {"fresh_seed": True})
            with self.assertRaisesRegex(ValueError, "allowance is exhausted"):
                server.execute_chat_tool(context, "generate_image", {})
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0]["seed"], "9007199254740993")
        self.assertFalse(sent[0]["enhance"])
        self.assertIsNone(sent[1]["seed"])
        self.assertTrue(first["generation_performed"])
        self.assertFalse(second["generation_performed"])
        self.assertTrue(second["reused"])
        self.assertEqual(second["generations_remaining"], 0)
        server.execute_chat_tool(context, "review_attempt", {"attempt_id": first["attempt_id"], "observation": "Castle remains central", "outcome": "regressed"})
        attempts = self.store.list_chat_attempts(self.chat["id"])
        self.assertEqual(attempts[1]["observation"], "Castle remains central")
        self.assertEqual(attempts[1]["recipe"]["seed"], "9007199254740993")

    def test_visual_review_requires_image_evidence_and_never_implies_approval(self):
        image = self.frame()
        attempt = self.store.record_chat_attempt(self.chat["id"], image["id"], {})
        context = self.turn()
        with self.assertRaisesRegex(ValueError, "Inspect"):
            server.execute_chat_tool(context, "review_attempt", {"attempt_id": attempt, "observation": "looks good", "outcome": "improved"})
        server.execute_chat_tool(context, "review_attempt", {"attempt_id": attempt, "observation": "pixels unavailable", "outcome": "unclear"})
        self.assertNotIn("approved_image_id", self.store.get_chat(self.chat["id"])["creative_brief"])

    def test_prompt_only_edit_does_not_generate_and_comparison_survives_reload(self):
        first, second = self.frame(), self.frame()
        context = self.turn()
        edit = server.execute_chat_tool(context, "edit_prompt", {"old_text": "fox", "new_text": "cat"})
        self.assertFalse(edit["generation_performed"])
        result = server.execute_chat_tool(context, "compare_images", {"image_ids": [first["id"], second["id"]]})
        transcript = [{"role": "assistant", "content": [{"type": "toolCall", "id": "comparison", "name": "compare_images", "arguments": {}}]},
                      {"role": "toolResult", "toolCallId": "comparison", "content": [{"type": "text", "text": "{}"}], "details": result}]
        public = server.public_chat_messages(transcript)
        self.assertEqual(public[0]["tools"][0]["comparison_image_ids"], [first["id"], second["id"]])

    def test_auto_settings_and_aspect_ratio_preserve_intent(self):
        result = server.validate_chat_setting_patch({"steps": None, "guidance": None, "seed": None}, {"preset": "turbo-int8", "seed": "17", "steps": 15})
        self.assertIsNone(result["steps"])
        self.assertIsNone(result["guidance"])
        self.assertIsNone(result["seed"])
        width, height = server.aspect_ratio_dimensions("16:9", 1440, 1440)
        self.assertLess(abs(width * height / (1440 * 1440) - 1), .02)
        self.assertLess(abs(width / height - 16 / 9), .03)
        with self.assertRaises(ValueError):
            server.aspect_ratio_dimensions("1:0")

    def test_scene_recipes_round_trip_without_losing_the_setting(self):
        scene = self.store.create_style("Vortex profile", "Falling beside the subject", "camera alongside the subject falling through a violet vortex", kind="scene")
        self.assertEqual(scene["kind"], "scene")
        edited = self.store.update_style(scene["id"], {"description": "A closer crop"})
        self.assertEqual(edited["kind"], "scene")
        self.assertIn("vortex", edited["style_text"])
        context = self.turn()
        listed = server.execute_chat_tool(context, "list_styles", {})["styles"]
        self.assertEqual(listed[0]["kind"], "scene")
        body = server.execute_chat_tool(context, "load_skill", {"name": "save-scene"})
        self.assertIn("instructions", body)

    def test_legacy_modes_migrate_to_create_without_rewriting_user_messages(self):
        self.attach_history([])
        with self.store._connect() as db:
            db.execute("UPDATE chat_sessions SET permission_mode='draft', system_mode='draft' WHERE id=?", (self.chat["id"],))
        reopened = Store(self.root / "audit.sqlite3")
        self.assertEqual(reopened.get_chat(self.chat["id"])["permission_mode"], "create")
        self.assertEqual(reopened.list_chat_messages(self.chat["id"])[0]["content"][0]["text"], "Use these references")
        self.assertIn("generate_image", server.chat_tool_names("draft"))

    def test_manifest_carries_versions_and_image_delivery_without_credentials(self):
        image = self.frame(reference=True)
        context = self.turn([image["id"]])
        self.store.save_chat_turn(context["turn_id"], self.chat["id"], context["manifest"])
        saved = self.store.list_chat_turns(self.chat["id"])[0]
        self.assertEqual(saved["initial_image_ids"], [image["id"]])
        self.assertEqual(len(saved["tools_hash"]), 64)
        self.assertNotIn("secret-not-for-manifests", json.dumps(saved))
        self.assertNotIn("base64", json.dumps(saved))

    def test_stop_targets_only_its_own_generation(self):
        state = server.AppState()
        own, other = threading.Event(), threading.Event()
        state.external_cancel = own
        state.set_progress("sampling", .5, "Working", active=True, generation_id="owned")
        with patch.object(state, "runtime") as runtime:
            self.assertFalse(state.request_cancel(expected_event=other)["cancelled"])
            runtime.request_interrupt.assert_not_called()
            self.assertTrue(state.request_cancel(expected_event=own)["cancelled"])
            runtime.request_interrupt.assert_called_once()
        state.set_progress("complete", 1, "Done", active=False)
        self.assertFalse(state.request_cancel(expected_event=own)["cancelled"])

    def test_stop_before_generation_starts_never_loads_models(self):
        state = server.AppState()
        stopped = threading.Event()
        stopped.set()
        with patch.object(server.offcut_cli, "load_settings", side_effect=AssertionError("Should not reach settings")):
            with self.assertRaises(server.offcut_cli.GenerationCancelled):
                state.generate({"prompt": "fox"}, cancel_event=stopped)
        self.assertFalse(state.busy)
        self.assertIsNone(state.external_cancel)
        self.assertFalse(state.cancel_requested)

    def test_comparison_is_present_in_both_stream_and_persisted_tool(self):
        ids = [self.frame()["id"], self.frame()["id"]]
        event = server.normalized_bridge_event({"type": "tool_execution_end", "toolCallId": "call",
            "toolName": "compare_images", "result": {"details": {"comparison_image_ids": ids}}})
        self.assertEqual(event["comparison_image_ids"], ids)

    def test_historical_prompt_receipt_is_compact_without_mutating_transcript(self):
        original = {"role": "toolResult", "toolName": "edit_prompt", "content": [{"type": "text", "text": json.dumps({
            "prompt": "a fox", "settings": {"prompt": "a fox", "seed": "42"},
            "prompt_change": {"before": "fox", "after": "a fox"}})}], "details": {"prompt_change": "UI only"}}
        copy = json.dumps(original)
        replay = server.pi_context_message(original)
        result = json.loads(replay["content"][0]["text"])
        self.assertFalse(result["generation_performed"])
        self.assertNotIn("prompt", result["settings"])
        self.assertNotIn("details", replay)
        self.assertEqual(json.dumps(original), copy)
