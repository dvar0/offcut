import json
import io
import os
import queue
import re
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.client import HTTPConnection
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import offcut_cli
import offcut_server
import offcut_store
from offcut_store import Store


class ValidationTests(unittest.TestCase):
    def test_integer_parser_rejects_fraction(self):
        with self.assertRaises(ValueError):
            offcut_server.parse_integer(1024.5, "width")

    def test_endpoint_requires_https_or_loopback(self):
        self.assertEqual(
            offcut_server.validate_endpoint("http://127.0.0.1:8000/v1/chat/completions"),
            "http://127.0.0.1:8000/v1/chat/completions",
        )
        with self.assertRaises(ValueError):
            offcut_server.validate_endpoint("http://example.com/v1/chat/completions")

    def test_invalid_settings_are_not_persisted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "settings.json"
            with patch.dict(os.environ, {"OFFCUT_CONFIG": str(path)}):
                with self.assertRaises(ValueError):
                    offcut_server.update_settings({"api_key_env": ""})
                self.assertFalse(path.exists())

    def test_generation_bounds(self):
        with self.assertRaises(ValueError):
            offcut_cli.validate_dimensions(4096, 4096)
        with self.assertRaises(ValueError):
            offcut_cli.validate_generation_values("prompt", 0, 1_000_000, 0.0)

    def test_reasoning_blocks_are_removed(self):
        response = {
            "output": [
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "private chain"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "a clean fox logo"}]},
            ]
        }
        self.assertEqual(offcut_server.extract_enhanced_text(response), "a clean fox logo")

    def test_string_reasoning_tags_and_duplicate_shapes_are_removed(self):
        response = {
            "output_text": "<think>private chain</think> a concise fox icon",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "duplicate"}]}],
        }
        self.assertEqual(offcut_server.extract_enhanced_text(response), "a concise fox icon")

    def test_anthropic_thinking_blocks_are_removed(self):
        response = {
            "content": [
                {"type": "thinking", "thinking": "private chain"},
                {"type": "text", "text": "a paper fox under moonlight"},
            ]
        }
        self.assertEqual(offcut_server.extract_enhanced_text(response), "a paper fox under moonlight")

    def test_opencode_protocol_selection_asks_the_catalog_first(self):
        connection = {"protocol": "opencode-go", "base_url": "https://opencode.ai/zen/go/v1"}
        # The vendor prefix would route the first two to Anthropic and the third to completions.
        self.assertEqual(offcut_server.connection_protocol(connection, "qwen3.8-max"), "chat")
        self.assertEqual(offcut_server.connection_protocol(connection, "minimax-m2.7"), "chat")
        self.assertEqual(
            offcut_server.connection_protocol(connection, "muse-spark-1.2-contributor"), "responses"
        )
        self.assertEqual(offcut_server.connection_protocol(connection, "qwen3.8-flash"), "anthropic")
        self.assertEqual(offcut_server.connection_protocol(connection, "glm-5.3-flash"), "chat")

    def test_opencode_protocol_falls_back_to_the_name_for_unknown_models(self):
        connection = {"protocol": "opencode-go", "base_url": "https://opencode.ai/zen/go/v1"}
        self.assertEqual(offcut_server.connection_protocol(connection, "qwen3-coder"), "anthropic")
        self.assertEqual(offcut_server.connection_protocol(connection, "gpt-luna"), "responses")
        self.assertEqual(offcut_server.connection_protocol(connection, "glm-5.2-unreleased"), "chat")

    def test_replayed_image_placeholder_names_who_produced_the_image(self):
        # "was attached" alone read as the user having attached a tool's own output, and a model
        # answered about a reference picture nobody had sent.
        user_turn = offcut_server.pi_context_message(
            {"role": "user", "content": [{"type": "imageRef", "imageId": "img-1"}]}
        )
        self.assertIn("The user attached image img-1", user_turn["content"][0]["text"])
        tool_turn = offcut_server.pi_context_message(
            {"role": "toolResult", "content": [{"type": "imageRef", "imageId": "img-2"}]}
        )
        self.assertIn("returned by a tool", tool_turn["content"][0]["text"])
        self.assertNotIn("user", tool_turn["content"][0]["text"])

    def test_public_chat_message_exposes_reasoning_but_strips_private_signatures(self):
        messages = [
            {
                "id": "db-message",
                "sequence": 2,
                "created_at": "now",
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Displayable provider reasoning",
                        "thinkingSignature": "private-thinking-signature",
                    },
                    {"type": "text", "text": "Visible answer"},
                    {
                        "type": "toolCall",
                        "id": "tool-1",
                        "name": "set_prompt",
                        "arguments": {"prompt": "new"},
                        "thoughtSignature": "private-signature",
                    },
                ],
                "usage": {
                    "input": 12,
                    "output": 4,
                    "cacheRead": 3,
                    "cacheWrite": 2,
                    "cost": {"total": 0.25},
                },
                "path": "/private/file",
                "data": "base64-private",
            },
            {
                "role": "toolResult",
                "toolCallId": "tool-1",
                "toolName": "set_prompt",
                "content": [{"type": "text", "text": "Updated"}],
                "isError": False,
            },
        ]
        public = offcut_server.public_chat_messages(messages)
        self.assertEqual(len(public), 1)
        self.assertEqual(public[0]["content"], "Visible answer")
        self.assertEqual(public[0]["tools"][0]["detail"], "Updated")
        self.assertEqual(public[0]["usage"]["cost"], 0.25)
        self.assertEqual(public[0]["reasoning"], "Displayable provider reasoning")
        encoded = json.dumps(public)
        self.assertNotIn("private-signature", encoded)
        self.assertNotIn("base64", encoded)

    def test_glm_flash_capabilities_come_from_pi_catalog(self):
        capabilities = offcut_server.describe_chat_model(
            {
                "protocol": "opencode-go",
                "base_url": "https://opencode.ai/zen/go/v1",
            },
            "glm-5.3-flash",
        )
        self.assertTrue(capabilities["vision"])
        self.assertTrue(capabilities["reasoning"])
        self.assertEqual(capabilities["supportedThinkingLevels"], ["low", "high", "max"])
        self.assertEqual(capabilities["recommendedThinkingLevel"], "high")

    def test_reasoning_stream_events_are_public(self):
        self.assertEqual(
            offcut_server.normalized_bridge_event(
                {
                    "type": "message_update",
                    "assistantMessageEvent": {
                        "type": "thinking_delta",
                        "delta": "Considering composition",
                        "contentIndex": 0,
                    },
                }
            ),
            {
                "type": "thinking_delta",
                "delta": "Considering composition",
                "content_index": 0,
            },
        )

    def test_tool_permissions_aspect_ratio_and_nonvision_metadata(self):
        self.assertIn("edit_prompt", offcut_server.chat_tool_names("draft"))
        self.assertIn("update_style", offcut_server.chat_tool_names("draft"))
        self.assertIn("generate_image", offcut_server.chat_tool_names("draft"))
        self.assertIn("generate_image", offcut_server.chat_tool_names("create"))
        width, height = offcut_server.aspect_ratio_dimensions("16:9")
        offcut_cli.validate_dimensions(width, height)
        self.assertLess(abs(width / height - 16 / 9), 0.03)
        self.assertFalse(offcut_server.model_supports_vision("plain-text-model"))
        self.assertFalse(offcut_server.image_has_usable_context({"metadata": {}}))
        self.assertTrue(offcut_server.image_has_usable_context({"raw_prompt": "a fox"}))

    def test_model_discovery_and_reasoning_free_enhancement(self):
        class MockProvider(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                body = json.dumps({"data": [{"id": "mock-vision"}, {"id": "mock-fast"}]}).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                self.server.last_request = request
                body = json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "reasoning_content": "private chain",
                                    "content": [{"type": "text", "text": "a precise geometric fox mark"}],
                                }
                            }
                        ]
                    }
                ).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), MockProvider)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = {
            "id": "mock",
            "name": "Mock",
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "protocol": "chat",
        }
        try:
            with patch.object(offcut_server.STORE, "get_connection_key", return_value="test-key"):
                self.assertEqual(
                    offcut_server.discover_connection_models(connection),
                    ["mock-fast", "mock-vision"],
                )
                result = offcut_server.enhance_with_connection(
                    "a fox icon",
                    ["muted sketch style"],
                    connection,
                    "mock-fast",
                )
            self.assertEqual(result, "a precise geometric fox mark")
            user_message = server.last_request["messages"][1]["content"]
            self.assertIn("muted sketch style", user_message)
            self.assertIn("app will prepend", user_message)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


class LocalBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = offcut_server.OffcutHTTPServer(("127.0.0.1", 0), offcut_server.RequestHandler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_same_origin_health_request(self):
        with urllib.request.urlopen(f"{self.base_url}/api/health") as response:
            self.assertEqual(json.load(response)["ok"], True)
            self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))

    def test_cross_origin_request_is_rejected(self):
        request = urllib.request.Request(
            f"{self.base_url}/api/health",
            headers={"Origin": "https://attacker.example"},
        )
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(request)
        self.assertEqual(context.exception.code, 403)
        self.assertIsNone(context.exception.headers.get("Access-Control-Allow-Origin"))

    def test_unbound_hosts_and_malformed_origins_are_rejected(self):
        for headers in (
            {"Host": "192.168.1.50"},
            {"Host": "attacker.example"},
            {"Host": "127.0.0.1:bad-port"},
            {"Host": "user@127.0.0.1"},
            {"Origin": "http://[broken"},
            {"Origin": "null"},
            {"Origin": self.base_url.replace("http:", "https:")},
        ):
            with self.subTest(headers=headers):
                request = urllib.request.Request(f"{self.base_url}/api/health", headers=headers)
                with self.assertRaises(urllib.error.HTTPError) as context:
                    urllib.request.urlopen(request)
                self.assertEqual(context.exception.code, 403)

    def test_settings_and_legacy_connections_serve_app(self):
        for path in ("/settings", "/connections"):
            with urllib.request.urlopen(f"{self.base_url}{path}") as response:
                body = response.read().decode()
                self.assertEqual(response.headers.get_content_type(), "text/html")
                self.assertIn("<title>Offcut</title>", body)
                self.assertIn('data-page="settings"', body)

    def test_theme_catalog_is_served(self):
        with urllib.request.urlopen(f"{self.base_url}/themes.js") as response:
            body = response.read().decode()
            self.assertIn("DEFAULT_APPEARANCE", body)
            self.assertIn("offcut-dark", body)
        with urllib.request.urlopen(f"{self.base_url}/themes.css") as response:
            self.assertIn("[data-theme=\"oracle\"]", response.read().decode())

    def test_post_requires_json_content_type(self):
        request = urllib.request.Request(
            f"{self.base_url}/api/unload",
            data=b"{}",
            method="POST",
            headers={"Content-Type": "text/plain"},
        )
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(request)
        self.assertEqual(context.exception.code, 400)


class LanBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = offcut_server.OffcutHTTPServer(("0.0.0.0", 0), offcut_server.RequestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        # A second loopback IP exercises the wildcard listener without needing a LAN adapter.
        cls.origin = f"http://127.0.0.2:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, path="/api/health", headers=None, method="GET", body=None):
        connection = HTTPConnection("127.0.0.2", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_interface_ip_serves_app_and_same_origin_api(self):
        status, _, body = self.request("/create")
        self.assertEqual(status, 200)
        self.assertIn(b"<title>Offcut</title>", body)
        status, headers, body = self.request(headers={"Origin": self.origin})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["ok"])
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_same_origin_post_reaches_handler(self):
        with patch.object(offcut_server.STATE, "close") as close:
            status, _, body = self.request(
                "/api/unload", method="POST", body=b"{}",
                headers={"Origin": self.origin, "Content-Type": "application/json"},
            )
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["ok"])
        close.assert_called_once()

    def test_server_hostname_is_accepted(self):
        host = f"{self.server.server_name}:{self.server.server_port}"
        status, _, _ = self.request(headers={"Host": host, "Origin": f"http://{host}"})
        self.assertEqual(status, 200)

    def test_cross_origin_and_foreign_hosts_stay_rejected(self):
        for headers in (
            {"Origin": "https://attacker.example"},
            {"Origin": "http://127.0.0.2:1"},
            {"Host": "attacker.example", "Origin": "http://attacker.example"},
            {"Host": "192.168.1.50"},
        ):
            with self.subTest(headers=headers):
                status, response_headers, _ = self.request(headers=headers)
                self.assertEqual(status, 403)
                self.assertNotIn("Access-Control-Allow-Origin", response_headers)


class ChatServerTests(unittest.TestCase):
    def setUp(self):
        inference = patch.object(offcut_server.STATE, "generate", side_effect=AssertionError("Chat tests must mock inference explicitly"))
        inference.start()
        self.addCleanup(inference.stop)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp_dir.name) / "chat.sqlite3")
        self.store.save_connection(
            {
                "id": "chat-provider",
                "name": "Chat Provider",
                "base_url": "http://127.0.0.1:9000/v1",
                "protocol": "chat",
                "models": ["plain-model", "vision-model"],
                "api_key": "super-secret-key",
            }
        )
        self.board = self.store.create_board("Chat board")
        self.store_patch = patch.object(offcut_server, "STORE", self.store)
        self.store_patch.start()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), offcut_server.RequestHandler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.store_patch.stop()
        self.temp_dir.cleanup()

    def request(self, path, payload=None, origin=None):
        headers = {}
        data = None
        method = "GET"
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
            method = "POST"
        if origin:
            headers["Origin"] = origin
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        with urllib.request.urlopen(request) as response:
            return json.load(response)

    def create_chat(self, model="plain-model", mode="draft"):
        return self.request(
            "/api/chats",
            {
                "board_id": self.board["id"],
                "connection_id": "chat-provider",
                "model": model,
                "permission_mode": mode,
            },
        )["chat"]

    def describe_levels(self, levels_by_model):
        def describe(_connection, model):
            levels = levels_by_model[model]
            return {
                "api": None,
                "vision": True,
                "reasoning": True,
                "supportedThinkingLevels": levels,
                "recommendedThinkingLevel": levels[-1],
            }

        return patch.object(offcut_server, "describe_chat_model", side_effect=describe)

    def test_new_chat_takes_the_profile_default_model_and_effort(self):
        self.store.save_connection(
            {"id": "chat-provider", "default_model": "vision-model", "default_reasoning": "medium"}
        )
        levels = {"vision-model": ["off", "low", "medium", "high"], "plain-model": ["low", "high", "max"]}
        with self.describe_levels(levels):
            chat = self.request(
                "/api/chats",
                {"board_id": self.board["id"], "connection_id": "chat-provider", "permission_mode": "create"},
            )["chat"]
            self.assertEqual(chat["model"], "vision-model")
            self.assertEqual(chat["reasoning_effort"], "medium")
            # plain-model offers no "medium", so the profile preference gives way to whatever the
            # describer recommends for the model actually being switched to.
            switched = self.request(f"/api/chats/{chat['id']}", {"model": "plain-model"})["chat"]
        self.assertEqual(switched["model"], "plain-model")
        self.assertEqual(switched["reasoning_effort"], "max")

    def test_new_chat_without_a_profile_default_uses_the_first_synced_model(self):
        with self.describe_levels({"plain-model": ["low", "high"]}):
            chat = self.request(
                "/api/chats",
                {"board_id": self.board["id"], "connection_id": "chat-provider", "permission_mode": "create"},
            )["chat"]
        self.assertEqual(chat["model"], "plain-model")
        self.assertEqual(chat["reasoning_effort"], "high")

    def test_chat_crud_and_local_boundary(self):
        chat = self.create_chat()
        listed = self.request(f"/api/chats?board_id={self.board['id']}")
        self.assertEqual([item["id"] for item in listed["chats"]], [chat["id"]])
        detail = self.request(f"/api/chats/{chat['id']}")
        self.assertEqual(detail["chat"]["board_id"], self.board["id"])
        self.assertEqual(detail["messages"], [])
        updated = self.request(f"/api/chats/{chat['id']}", {"title": "A useful title"})
        self.assertEqual(updated["chat"]["title"], "A useful title")
        with self.assertRaises(urllib.error.HTTPError) as invalid_model:
            self.request(f"/api/chats/{chat['id']}", {"model": "not-synced"})
        self.assertEqual(invalid_model.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as cross_origin:
            self.request(
                f"/api/chats?board_id={self.board['id']}",
                origin="https://attacker.example",
            )
        self.assertEqual(cross_origin.exception.code, 403)
        self.assertEqual(self.request(f"/api/chats/{chat['id']}/delete", {}), {"ok": True})

    def test_draft_prompt_mutation_is_revision_guarded(self):
        chat = self.create_chat()
        context = {
            "chat": chat,
            "settings": {"prompt": "a red fox", "width": 1024, "height": 1024},
            "revision": self.board["settings_revision"],
            "selected_image_id": None,
            "tool_lock": threading.Lock(),
        }
        patches = []
        result = offcut_server.execute_chat_tool(
            context,
            "edit_prompt",
            {"old_text": "red", "new_text": "silver"},
            emit=patches.append,
        )
        self.assertEqual(result["prompt"], "a silver fox")
        self.assertEqual(result["prompt_change"]["removed"], "red")
        self.assertEqual(result["prompt_change"]["added"], "silver")
        self.assertEqual(self.store.get_board(self.board["id"])["settings"]["prompt"], "a silver fox")
        self.assertEqual(patches[0]["type"], "workspace_patch")
        self.assertEqual(patches[0]["prompt_change"]["before"], "a red fox")
        self.assertIn("generate_image", offcut_server.chat_tool_names(context["chat"]["permission_mode"]))
        with self.assertRaisesRegex(ValueError, "exactly once"):
            offcut_server.execute_chat_tool(
                context,
                "edit_prompt",
                {"old_text": "missing", "new_text": "replacement"},
            )
        context["chat"] = self.store.update_chat(chat["id"], {"permission_mode": "create"})
        settings_result = offcut_server.execute_chat_tool(
            context,
            "update_generation_settings",
            # raw-int8 because guidance is only a dial on the undistilled route.
            {"preset": "raw-int8", "width": 1152, "height": 896, "steps": 12, "guidance": 2.5},
        )
        self.assertEqual(settings_result["settings"]["width"], 1152)
        self.assertEqual(settings_result["settings"]["steps"], 12)

    def make_image(self, board_id, prompt="a harbour at dawn"):
        path = Path(self.temp_dir.name) / f"{prompt.replace(' ', '-')}-{board_id}.png"
        path.write_bytes(b"not-a-real-png")
        return self.store.record_image(
            board_id,
            {
                "output_path": str(path),
                "raw_prompt": prompt,
                "enhanced_prompt": "",
                "final_prompt": prompt,
                "preset": "turbo-int8",
                "width": 1024,
                "height": 1024,
                "seed": 7,
                "steps": 8,
                "guidance": 0.0,
                "loras": [],
            },
            {"prompt": prompt},
        )

    def tool_context(self, chat, settings=None):
        return {
            "chat": chat,
            "settings": settings or {"prompt": "a red fox", "preset": "turbo-int8"},
            "revision": self.store.get_board(self.board["id"])["settings_revision"],
            "workspace_selected_image_id": None,
            "tool_lock": threading.Lock(),
            "vision": False,
            "known_images": {},
            "images": {},
            "seen_image_ids": set(),
        }

    def test_style_api_round_trip(self):
        image = self.make_image(self.board["id"])
        created = self.request(
            "/api/styles",
            {
                "name": "Deco Travel Poster",
                "description": "Flat graphic poster work",
                "style_text": "1920s travel-poster lithography, stippled ink texture, peach and cobalt palette",
                "reference_image_id": image["id"],
            },
        )
        self.assertEqual(created["reference_url"], f"/api/image-files/{image['id']}")
        listed = self.request("/api/styles")["styles"]
        self.assertEqual([item["id"] for item in listed], [created["id"]])
        renamed = self.request(f"/api/styles/{created['id']}", {"name": "Deco Poster"})
        self.assertEqual(renamed["name"], "Deco Poster")
        self.assertEqual(self.request(f"/api/styles/{created['id']}/delete", {}), {"ok": True})
        self.assertEqual(self.request("/api/styles")["styles"], [])

    def test_skills_endpoint_lists_names_without_bodies(self):
        skills = self.request("/api/skills")["skills"]
        self.assertIn("save-style", [skill["name"] for skill in skills])
        self.assertTrue(all(set(skill) == {"name", "description"} for skill in skills))

    def test_style_tools_read_and_save_but_never_activate(self):
        chat = self.create_chat()
        image = self.make_image(self.board["id"])
        context = self.tool_context(chat)
        saved = offcut_server.execute_chat_tool(
            context,
            "save_style",
            {
                "name": "Muted Risograph",
                "description": "Two-ink print work",
                "style_text": "duotone risograph print, misregistered ink, coarse paper grain",
                "reference_image_id": image["id"],
            },
        )
        self.assertEqual(saved["name"], "Muted Risograph")

        listed = offcut_server.execute_chat_tool(context, "list_styles", {})
        self.assertEqual([item["name"] for item in listed["styles"]], ["Muted Risograph"])
        # Saving must never turn a style on: that toggle belongs to the user.
        self.assertFalse(listed["styles"][0]["active"])

        inspected = offcut_server.execute_chat_tool(context, "inspect_style", {"style_id": saved["id"]})
        self.assertEqual(inspected["style_text"], "duotone risograph print, misregistered ink, coarse paper grain")
        self.assertEqual(inspected["reference"]["id"], image["id"])

        # styles is reported in workspace state but is not a patchable setting.
        with self.assertRaisesRegex(ValueError, "Unsupported generation setting"):
            offcut_server.validate_chat_setting_patch({"styles": [saved["id"]]}, context["settings"])

    def test_update_style_edits_in_place_without_duplicating(self):
        chat = self.create_chat()
        context = self.tool_context(chat)
        saved = offcut_server.execute_chat_tool(
            context,
            "save_style",
            {
                "name": "Muted Risograph",
                "description": "Two-ink print work",
                "style_text": "duotone risograph print, misregistered ink, coarse paper grain",
            },
        )
        updated = offcut_server.execute_chat_tool(
            context,
            "update_style",
            {
                "style_id": saved["id"],
                "description": "Two-ink print work with flat spot color",
                "style_text": "duotone risograph print, misregistered ink, coarse paper grain, flat spot color",
            },
        )
        # Omitted fields keep their value and the library still holds exactly one entry.
        self.assertEqual(updated["name"], "Muted Risograph")
        listed = offcut_server.execute_chat_tool(context, "list_styles", {})
        self.assertEqual([item["id"] for item in listed["styles"]], [saved["id"]])
        inspected = offcut_server.execute_chat_tool(context, "inspect_style", {"style_id": saved["id"]})
        self.assertEqual(
            inspected["style_text"],
            "duotone risograph print, misregistered ink, coarse paper grain, flat spot color",
        )

        # Editing is a library write like saving, so draft mode carries it too.
        draft = self.tool_context(self.store.update_chat(chat["id"], {"permission_mode": "draft"}))
        offcut_server.execute_chat_tool(draft, "update_style", {"style_id": saved["id"], "name": "Flat Risograph"})

        with self.assertRaisesRegex(ValueError, "at least one"):
            offcut_server.execute_chat_tool(context, "update_style", {"style_id": saved["id"]})
        with self.assertRaisesRegex(ValueError, "Style not found"):
            offcut_server.execute_chat_tool(context, "update_style", {"style_id": "missing", "name": "Ghost"})

    def test_load_skill_returns_instructions_in_every_mode(self):
        chat = self.create_chat(mode="draft")
        context = self.tool_context(chat)
        result = offcut_server.execute_chat_tool(context, "load_skill", {"name": "save-style"})
        self.assertEqual(result["name"], "save-style")
        self.assertIn("style", result["instructions"].lower())
        with self.assertRaises(ValueError):
            offcut_server.execute_chat_tool(context, "load_skill", {"name": "no-such-skill"})

    def test_an_attachment_may_come_from_another_board_but_a_selection_may_not(self):
        chat = self.create_chat()
        other_board = self.store.create_board("Elsewhere")
        foreign = self.make_image(other_board["id"], "a lone fisherman")
        base = {
            "message": "look at this",
            "workspace": self.store.get_board(self.board["id"])["settings"],
            "workspace_revision": self.store.get_board(self.board["id"])["settings_revision"],
        }
        context = offcut_server.prepare_chat_turn(chat["id"], {**base, "attachment_ids": [foreign["id"]]})
        self.assertIn(foreign["id"], context["attachment_ids"])
        with self.assertRaisesRegex(ValueError, "selected image must belong"):
            offcut_server.prepare_chat_turn(chat["id"], {**base, "selected_image_id": foreign["id"]})

    def test_compare_images_allows_an_image_already_attached_from_another_board(self):
        chat = self.create_chat()
        other_board = self.store.create_board("Elsewhere")
        local = self.make_image(self.board["id"], "a harbour")
        foreign = self.make_image(other_board["id"], "a fisherman")
        stranger = self.make_image(other_board["id"], "a gull")
        context = self.tool_context(chat)
        context["known_images"] = {foreign["id"]: foreign}
        result = offcut_server.execute_chat_tool(
            context, "compare_images", {"image_ids": [local["id"], foreign["id"]]}
        )
        self.assertEqual(len(result["images"]), 2)
        with self.assertRaisesRegex(ValueError, "belong to the chat's board"):
            offcut_server.execute_chat_tool(
                context, "compare_images", {"image_ids": [local["id"], stranger["id"]]}
            )

    def test_a_slash_skill_message_reaches_the_model_as_the_skill_body(self):
        chat = self.create_chat()
        board = self.store.get_board(self.board["id"])
        context = offcut_server.prepare_chat_turn(
            chat["id"],
            {
                "message": "/save-style call it Deco Poster",
                "workspace": board["settings"],
                "workspace_revision": board["settings_revision"],
            },
        )
        self.assertEqual(context["skill_request"], "save-style")
        self.assertIn("<skill name=\"save-style\">", context["agent_prompt"])
        self.assertIn("call it Deco Poster", context["agent_prompt"])
        # The transcript keeps the short command; only the model sees the expansion.
        self.assertEqual(context["message"], "/save-style call it Deco Poster")

    def test_active_styles_become_prompt_triggers_and_survive_deletion(self):
        first = self.store.create_style("Deco Poster", "", "1920s travel-poster lithography, stippled ink texture")
        second = self.store.create_style("Grainy Film", "", "35mm film grain, halation, muted kodachrome")
        self.store.delete_style(second["id"])

        # A style deleted while still active on some board stops applying rather than breaking
        # that board's next generation.
        triggers = offcut_server.resolve_style_triggers([first["id"], second["id"]])
        self.assertEqual(triggers, ["1920s travel-poster lithography, stippled ink texture"])

        prefixed = offcut_cli.prefix_prompt("a lone fisherman mending nets", triggers)
        self.assertEqual(
            prefixed, "1920s travel-poster lithography, stippled ink texture, a lone fisherman mending nets"
        )
        # Idempotent, so a style the user or the agent already wrote into the prompt is not doubled.
        self.assertEqual(offcut_cli.prefix_prompt(prefixed, triggers), prefixed)

        with self.assertRaisesRegex(ValueError, "at most eight"):
            offcut_server.resolve_style_triggers([first["id"]] * 9)
        with self.assertRaisesRegex(ValueError, "must be a style ID"):
            offcut_server.resolve_style_triggers([{"id": first["id"]}])

    def test_a_generated_image_rides_on_the_tool_call_into_the_transcript(self):
        # The toolResult message that carries the reference is dropped from the public
        # transcript, so without this lift the generated image is visible only while the turn
        # streams and disappears on the next reload.
        image = self.make_image(self.board["id"], "a smooth grey pebble")
        messages = [
            {
                "id": "m1",
                "sequence": 0,
                "role": "assistant",
                "content": [{"type": "toolCall", "id": "call-1", "name": "generate_image", "arguments": {}}],
            },
            {
                "role": "toolResult",
                "toolCallId": "call-1",
                "content": [{"type": "text", "text": json.dumps({"image_id": image["id"], "image": {"id": image["id"]}})}],
            },
        ]
        public = offcut_server.public_chat_messages(messages)
        tools = [block["tool"] for block in public[0]["blocks"] if block["type"] == "tool"]
        self.assertEqual(tools[0]["image_id"], image["id"])
        self.assertEqual(tools[0]["status"], "complete")

    def test_a_tool_call_without_an_image_gets_no_image_id(self):
        messages = [
            {
                "id": "m1",
                "sequence": 0,
                "role": "assistant",
                "content": [{"type": "toolCall", "id": "call-1", "name": "set_prompt", "arguments": {}}],
            },
            {
                "role": "toolResult",
                "toolCallId": "call-1",
                "content": [{"type": "text", "text": json.dumps({"prompt": "a silver fox"})}],
            },
        ]
        public = offcut_server.public_chat_messages(messages)
        tool = [block["tool"] for block in public[0]["blocks"] if block["type"] == "tool"][0]
        self.assertNotIn("image_id", tool)

    def test_mock_bridge_turn_streams_persists_and_auto_titles(self):
        image_path = Path(self.temp_dir.name) / "prompt-reference.png"
        image = self.store.record_reference_image(
            self.board["id"], image_path, 24, 16, "copper fox source prompt"
        )
        chat = self.create_chat()
        user_message = {
            "role": "user",
            "content": "Design a fox mark",
            "timestamp": 1,
        }
        assistant_message = {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "provider analysis",
                    "thinkingSignature": "private-signature",
                },
                {"type": "text", "text": "I prepared the prompt direction."},
            ],
            "usage": {
                "input": 10,
                "output": 5,
                "cacheRead": 0,
                "cacheWrite": 0,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0.01},
            },
            "timestamp": 2,
        }
        lines = [
            {"type": "event", "event": {"type": "message_end", "message": user_message}},
            {"type": "event", "event": {"type": "message_start", "message": assistant_message}},
            {"type": "event", "event": {"type": "message_end", "message": assistant_message}},
            {"type": "done"},
        ]

        class FakeProcess:
            def __init__(self):
                self.stdin = io.StringIO()
                self.stdout = io.StringIO("".join(json.dumps(line) + "\n" for line in lines))
                self.returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = 0

            def wait(self, timeout=None):
                self.returncode = 0
                return 0

            def kill(self):
                self.returncode = -9

        payload = {
            "message": "Design a fox mark",
            "selected_image_id": image["id"],
            "workspace_selected_image_id": image["id"],
            "attachment_ids": [],
            "workspace": {"board_id": self.board["id"]},
            "workspace_revision": 0,
        }
        with patch.object(offcut_server.subprocess, "Popen", return_value=FakeProcess()):
            request = urllib.request.Request(
                f"{self.base_url}/api/chats/{chat['id']}/turn",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                self.assertEqual(response.headers.get_content_type(), "application/x-ndjson")
                body = response.read().decode()
        self.assertIn('"type":"message_end"', body)
        self.assertIn('"type":"done"', body)
        self.assertIn("provider analysis", body)
        self.assertNotIn("private-signature", body)
        self.assertNotIn("super-secret-key", body)
        stored = self.store.list_chat_messages(chat["id"], include_compacted=True)
        self.assertEqual([message["role"] for message in stored], ["user", "assistant"])
        self.assertEqual(stored[0]["image_context"][0]["raw_prompt"], "copper fox source prompt")
        self.assertEqual(stored[1]["content"][0]["thinkingSignature"], "private-signature")
        self.assertEqual(self.store.get_chat(chat["id"])["title"], "Design a fox mark")

    def test_historical_image_references_do_not_reinflate_pixels(self):
        from PIL import Image

        image_path = Path(self.temp_dir.name) / "reference.png"
        Image.new("RGB", (12, 8), "red").save(image_path)
        image = self.store.record_reference_image(self.board["id"], image_path, 12, 8, "red reference")
        chat = self.create_chat(model="vision-model")
        self.store.append_chat_messages(
            chat["id"],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Use this reference"},
                        {"type": "imageRef", "imageId": image["id"]},
                    ],
                    "image_context": [offcut_server.curated_image(image)],
                },
                {"role": "assistant", "content": "I inspected it."},
            ],
        )
        payload = {
            "message": "Now tighten the prompt",
            "selected_image_id": None,
            "workspace_selected_image_id": image["id"],
            "attachment_ids": [],
            "workspace": {"board_id": self.board["id"]},
            "workspace_revision": 0,
        }
        context = offcut_server.prepare_chat_turn(chat["id"], payload)
        self.assertEqual(context["images"], {})
        self.assertEqual(context["prompt_image_ids"], [])
        replay = json.dumps(context["messages"])
        self.assertNotIn("imageRef", replay)
        self.assertIn("pixels are not replayed", replay)
        self.assertIn("red reference", replay)
        selected = offcut_server.execute_chat_tool(context, "get_selected_image", {})
        self.assertEqual(selected["image"]["raw_prompt"], "red reference")
        self.assertIn(image["id"], selected["__krea2_images"])
        self.assertEqual(selected["pixels_supplied"], [image["id"]])
        repeated = offcut_server.prepare_chat_turn(chat["id"], payload)
        self.assertEqual(context["messages"], repeated["messages"])

    def test_workspace_prompt_change_is_added_as_revisioned_event(self):
        chat = self.create_chat()
        board = self.store.update_board(
            self.board["id"],
            {
                "settings": {"prompt": "angular silver fox"},
                "expected_settings_revision": 0,
            },
        )
        context = offcut_server.prepare_chat_turn(
            chat["id"],
            {
                "message": "Continue",
                "selected_image_id": None,
                "attachment_ids": [],
                "workspace": {"board_id": self.board["id"]},
                "workspace_revision": board["settings_revision"],
            },
        )
        self.assertEqual(context["workspace_change"]["actor"], "user")
        self.assertEqual(context["workspace_change"]["added"], "angular silver fox")
        self.assertEqual(context["workspace_change"]["revision"], 1)
        self.assertIn("krea2_workspace_change", context["agent_prompt"])
        # settings_revision is an internal concurrency token that no tool accepts back. Handed it,
        # models narrate it at the user ("the prompt is set at revision 7"), which is meaningless
        # because the counter tracks board autosaves, not prompt versions.
        self.assertNotIn("revision", context["agent_prompt"])

    def test_mode_switch_announces_itself_without_rewriting_the_system_prompt(self):
        chat = self.create_chat(mode="create")

        def turn(message):
            return offcut_server.prepare_chat_turn(
                chat["id"],
                {
                    "message": message,
                    "selected_image_id": None,
                    "attachment_ids": [],
                    "workspace": {"board_id": self.board["id"]},
                    "workspace_revision": self.store.get_board(self.board["id"])["settings_revision"],
                },
            )

        opening = turn("Start")
        self.assertIn(offcut_server.CHAT_MODE_RULES["create"], opening["system_prompt"])
        self.assertIsNone(opening["mode_change"])
        # Older clients and saved chats cannot reintroduce retired restrictions.
        self.store.update_chat(chat["id"], {"permission_mode": "draft"})
        switched = turn("Edit only, don't generate")
        self.assertEqual(switched["system_prompt"], opening["system_prompt"])
        self.assertEqual(switched["chat"]["permission_mode"], "create")
        self.assertIsNone(switched["mode_change"])
        self.assertIn("Edit only, don't generate", switched["agent_prompt"])
        replayed = offcut_server.pi_context_message({"role": "user", "content": "Don't generate yet",
            "mode_change": {"from": "create", "to": "draft"}})
        replay = " ".join(block.get("text", "") for block in replayed["content"])
        self.assertIn("Don't generate yet", replay)
        self.assertNotIn("cannot change generation settings", replay)

    def test_workspace_tool_results_hide_the_revision_from_the_model(self):
        chat = self.create_chat(mode="iterate")
        # get_workspace_state reloads settings from the board, so the board has to agree with the
        # context or the edit_prompt below has nothing to match.
        self.store.update_board(self.board["id"], {"settings": {"prompt": "a red fox"}})
        context = {
            "chat": chat,
            "settings": {"prompt": "a red fox", "width": 1024, "height": 1024},
            "revision": self.board["settings_revision"],
            "selected_image_id": None,
            "workspace_selected_image_id": None,
            "tool_lock": threading.Lock(),
        }
        patches: list[dict] = []
        for tool, args in (
            ("get_workspace_state", {}),
            ("edit_prompt", {"old_text": "red", "new_text": "silver"}),
            ("update_generation_settings", {"steps": 8}),
            ("set_aspect_ratio", {"aspect_ratio": "16:9"}),
        ):
            result = offcut_server.execute_chat_tool(context, tool, args, emit=patches.append)
            self.assertNotIn("revision", result, tool)
            self.assertNotIn("revision", result.get("prompt_change", {}), tool)
        # The browser still needs it: it is the expected_settings_revision for its next write.
        self.assertTrue(patches)
        self.assertTrue(all(isinstance(patch.get("revision"), int) for patch in patches))

    def test_prompt_change_label_names_the_edit_not_a_number(self):
        self.assertEqual(offcut_server.prompt_change_label("set_prompt"), "Prompt rewritten")
        self.assertEqual(offcut_server.prompt_change_label("edit_prompt"), "Prompt edited")

    def test_reference_image_upload_is_board_local(self):
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (16, 10), "blue").save(buffer, format="PNG")
        request = urllib.request.Request(
            f"{self.base_url}/api/reference-images?board_id={self.board['id']}&label=moodboard.png",
            data=buffer.getvalue(),
            headers={"Content-Type": "image/png"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            uploaded = json.load(response)
        self.assertEqual(uploaded["board_id"], self.board["id"])
        self.assertEqual(uploaded["kind"], "reference")
        self.assertEqual((uploaded["width"], uploaded["height"]), (16, 10))
        self.assertEqual(self.store.get_image(uploaded["id"])["raw_prompt"], "moodboard.png")

    def test_compact_keeps_summary_visible_and_out_of_future_context(self):
        chat = self.create_chat()
        summary = "Keep the angular fox silhouette and muted red paper texture."
        lines = [
            {"type": "event", "event": {"type": "message_end", "message": {"role": "user", "content": "/compact"}}},
            {"type": "event", "event": {"type": "message_end", "message": {"role": "assistant", "content": summary}}},
            {"type": "done"},
        ]

        class FakeProcess:
            def __init__(self):
                self.stdin = io.StringIO()
                self.stdout = io.StringIO("".join(json.dumps(line) + "\n" for line in lines))
                self.returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = 0

            def wait(self, timeout=None):
                self.returncode = 0
                return 0

            def kill(self):
                self.returncode = -9

        payload = {
            "message": "/compact",
            "selected_image_id": None,
            "attachment_ids": [],
            "workspace": {"board_id": self.board["id"]},
            "workspace_revision": 0,
        }
        with patch.object(offcut_server.subprocess, "Popen", return_value=FakeProcess()):
            request = urllib.request.Request(
                f"{self.base_url}/api/chats/{chat['id']}/turn",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                response.read()

        visible = self.store.list_chat_messages(chat["id"])
        self.assertEqual([(message["role"], message["content"]) for message in visible], [("assistant", summary)])
        context = offcut_server.prepare_chat_turn(chat["id"], {**payload, "message": "Continue"})
        self.assertEqual(context["messages"], [])
        self.assertIn(summary, context["system_prompt"])

    def test_bridge_process_arguments_and_errors_do_not_expose_secret(self):
        class FakeProcess:
            stdin = None

            def poll(self):
                return 0

        chat = self.create_chat()
        self.store.append_chat_messages(chat["id"], [{"role": "user", "content": "hello"}])
        with patch.object(offcut_server.subprocess, "Popen", return_value=FakeProcess()) as popen:
            run = offcut_server.start_chat_bridge(chat["id"])
            try:
                command = popen.call_args.args[0]
                self.assertNotIn("super-secret-key", json.dumps(command))
                # The watermark the run captured is what the detail endpoint uses to hold the
                # turn's own messages back for the event replay.
                self.assertEqual(run.watermark, 1)
            finally:
                offcut_server.finish_chat_bridge(run)
        self.assertEqual(
            offcut_server.redact_bridge_error("provider rejected super-secret-key", "super-secret-key"),
            "provider rejected [redacted]",
        )

    def fake_running_run(self, chat_id, watermark=0):
        class FakeProcess:
            stdin = None

            def poll(self):
                return 0

        run = offcut_server.ActiveChatRun(chat_id, FakeProcess(), watermark)
        with offcut_server.ACTIVE_CHAT_LOCK:
            offcut_server.ACTIVE_CHAT_RUNS[chat_id] = run
        return run

    def release_run(self, run):
        with offcut_server.ACTIVE_CHAT_LOCK:
            offcut_server.ACTIVE_CHAT_RUNS.pop(run.chat_id, None)
        run.close_subscribers()

    def test_a_running_turn_hides_its_messages_until_released(self):
        chat = self.create_chat()
        self.store.append_chat_messages(
            chat["id"],
            [
                {"role": "user", "content": "a red fox"},
                {"role": "assistant", "content": "here is the fox"},
            ],
        )
        run = self.fake_running_run(chat["id"], watermark=2)
        try:
            # A message the turn persisted after it started: the transcript holds it back
            # because the buffered event replay owns it while the run is live.
            self.store.append_chat_messages(chat["id"], [{"role": "assistant", "content": "still working"}])
            detail = self.request(f"/api/chats/{chat['id']}")
            self.assertTrue(detail["chat"]["running"])
            self.assertEqual(
                [message["content"] for message in detail["messages"]],
                ["a red fox", "here is the fox"],
            )
            listed = self.request(f"/api/chats?board_id={self.board['id']}")
            self.assertTrue(listed["chats"][0]["running"])
        finally:
            self.release_run(run)
        detail = self.request(f"/api/chats/{chat['id']}")
        self.assertFalse(detail["chat"]["running"])
        self.assertEqual(
            [message["content"] for message in detail["messages"]],
            ["a red fox", "here is the fox", "still working"],
        )

    def test_events_endpoint_replays_the_turn_and_requires_a_live_run(self):
        chat = self.create_chat()
        with self.assertRaises(urllib.error.HTTPError) as refused:
            self.request(f"/api/chats/{chat['id']}/events")
        self.assertEqual(refused.exception.code, 409)

        run = self.fake_running_run(chat["id"])
        run.publish({"type": "message_start", "role": "assistant"})
        run.publish({"type": "text_delta", "delta": "working"})
        received = []

        def attach():
            request = urllib.request.Request(f"{self.base_url}/api/chats/{chat['id']}/events")
            with urllib.request.urlopen(request) as response:
                for line in response.read().decode().splitlines():
                    if line.strip():
                        received.append(json.loads(line))

        reader = threading.Thread(target=attach, daemon=True)
        reader.start()
        # The run must not be released before the handler has actually subscribed, or the
        # endpoint answers 409 and the reader dies on the HTTP error.
        deadline = time.monotonic() + 5
        while True:
            with run.fanout_lock:
                attached = bool(run.subscribers)
            if attached or time.monotonic() > deadline:
                break
            time.sleep(0.01)
        run.publish({"type": "done", "revision": 1})
        self.release_run(run)
        reader.join(timeout=5)
        self.assertFalse(reader.is_alive())
        # The whole turn replays from its first buffered event, then the stream ends on its
        # own once the run closes, without waiting for another event.
        self.assertEqual([event["type"] for event in received], ["message_start", "text_delta", "done"])
        self.assertEqual(received[1]["delta"], "working")

    def test_run_subscribers_see_each_event_once_and_end_on_close(self):
        class FakeProcess:
            stdin = None

            def poll(self):
                return 0

        run = offcut_server.ActiveChatRun("no-http", FakeProcess(), 0)
        run.publish({"type": "a"})
        early = run.subscribe()
        run.publish({"type": "b"})
        late = run.subscribe()
        run.close_subscribers()
        self.assertEqual([early.get()["type"], early.get()["type"], early.get()], ["a", "b", None])
        self.assertEqual([late.get()["type"], late.get()["type"], late.get()], ["a", "b", None])
        run.unsubscribe(late)
        run.publish({"type": "c"})
        with self.assertRaises(queue.Empty):
            late.get_nowait()


class LoraProfileTests(unittest.TestCase):
    """A LoRA is a file on disk; a profile only decorates one. discover_loras stays the pure
    listing the agent tools and the settings validator read, and the join happens above it."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp_dir.name) / "offcut.sqlite3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_a_lora_without_a_profile_still_carries_the_browser_fields(self):
        listed = offcut_server.public_lora({"name": "offcut_style", "trigger": "ink style"}, None)
        self.assertEqual(listed["display_name"], "")
        self.assertEqual(listed["notes"], "")
        self.assertEqual(listed["default_strength"], offcut_store.DEFAULT_LORA_STRENGTH)
        self.assertIsNone(listed["cover_url"])
        self.assertEqual(listed["trigger"], "ink style")

    def test_a_profile_supplies_a_cover_url_and_leaves_the_file_facts_alone(self):
        profile = {
            "name": "offcut_style",
            "display_name": "Poster Maximalist",
            "notes": "Tall posters only",
            "default_strength": 0.9,
            "cover_image_id": "abc123",
        }
        listed = offcut_server.public_lora({"name": "offcut_style", "trigger": "ink style"}, profile)
        self.assertEqual(listed["display_name"], "Poster Maximalist")
        self.assertEqual(listed["notes"], "Tall posters only")
        self.assertEqual(listed["default_strength"], 0.9)
        self.assertEqual(listed["cover_url"], "/api/image-files/abc123")
        self.assertEqual(listed["trigger"], "ink style")

    def test_the_listing_joins_profiles_onto_what_is_on_disk(self):
        self.store.save_lora_profile("offcut_style", {"display_name": "Poster Maximalist"})
        # A profile whose file is gone joins against nothing and drops out of the listing.
        self.store.save_lora_profile("offcut_deleted", {"display_name": "Stranded"})
        with patch.object(offcut_server, "STORE", self.store), patch.object(
            offcut_server, "discover_loras", return_value=[{"name": "offcut_style", "trigger": ""}]
        ), patch.object(offcut_cli, "load_settings", return_value={}):
            listed = offcut_server.public_loras()
        self.assertEqual([item["name"] for item in listed], ["offcut_style"])
        self.assertEqual(listed[0]["display_name"], "Poster Maximalist")


class ImageDeleteTests(unittest.TestCase):
    """A deleted frame has to lose its file too: startup re-indexes every PNG under the output
    directory, so a row deleted on its own is back in Inbox on the next launch."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store = Store(self.root / "offcut.sqlite3")
        self.board = self.store.create_board("Frames")
        self.store_patch = patch.object(offcut_server, "STORE", self.store)
        self.store_patch.start()
        self.addCleanup(self.store_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)

    def record(self, name):
        path = self.root / name
        path.write_bytes(b"not really a png")
        return self.store.record_image(
            self.board["id"],
            {"output_path": str(path), "width": 512, "height": 512, "seed": 1, "steps": 8, "guidance": 0},
            {"prompt": "test"},
        )

    def test_delete_removes_the_row_the_frame_and_its_thumbnails(self):
        image = self.record("frame.png")
        thumbnails = self.root / "thumbnails"
        thumbnails.mkdir()
        cached = thumbnails / f"{image['id']}-w256.webp"
        cached.write_bytes(b"thumb")
        other = thumbnails / "someone-else-w256.webp"
        other.write_bytes(b"thumb")

        offcut_server.delete_image_record(image["id"])

        self.assertFalse((self.root / "frame.png").exists())
        self.assertFalse(cached.exists())
        self.assertTrue(other.exists())
        self.assertEqual(self.store.list_images(self.board["id"]), [])

    def test_delete_still_drops_the_row_when_the_file_is_already_gone(self):
        image = self.record("frame.png")
        (self.root / "frame.png").unlink()
        offcut_server.delete_image_record(image["id"])
        with self.assertRaises(ValueError):
            self.store.get_image(image["id"])

    def test_deleting_a_missing_image_is_refused(self):
        with self.assertRaises(ValueError):
            offcut_server.delete_image_record("nothing")


class LoraGuidanceTests(unittest.TestCase):
    def test_enhancer_notes_skip_loras_without_user_notes(self):
        self.assertEqual(offcut_server.enhancer_lora_notes(["my_ink_style"], {}), [])
        self.assertEqual(offcut_server.enhancer_lora_notes([], {}), [])

    def test_web_enhancer_uses_user_notes_once_per_active_lora(self):
        settings = {"lora_metadata": {"my_ink_style": {
            "summary": "Ink illustration", "prompting_notes": "Caption pattern: subject, light.\nExample: a fox, soft light.",
        }}}
        notes = offcut_server.enhancer_lora_notes(["my_ink_style", "my_ink_style", "unknown"], settings)
        self.assertEqual(len(notes), 1)
        self.assertIn("Caption pattern: subject, light.", notes[0])
        connection = {"protocol": "chat", "base_url": "https://example.test/v1"}
        with patch.object(offcut_server.STORE, "get_connection_key", return_value="test-key"), patch("offcut_server.connection_request", return_value={"choices": [{"message": {"content": "a fox"}}]}) as request:
            offcut_server.enhance_with_connection("a fox", [], connection, "test-model", lora_notes=notes)
        content = request.call_args.args[3]["messages"][1]["content"]
        self.assertIn(notes[0], content)
        self.assertIn("not verified training facts", content)


if __name__ == "__main__":
    unittest.main()


# AppState.generate clears and re-arms Comfy's global stop flag around every run, so the runtime
# stand-in has to answer those calls and remembers them for the cancellation tests.
class StubRuntime:
    def __init__(self):
        self.interrupts = []

    def request_interrupt(self):
        self.interrupts.append(True)

    def clear_interrupt(self):
        self.interrupts.append(False)


class GenerationDraftTests(unittest.TestCase):
    """The board draft is the form's memory, so it has to keep AUTO distinguishable from a
    typed number even though the sampler needs a concrete value for the same run."""

    def run_generation(self, payload, state=None, on_loras=None):
        state = state or offcut_server.AppState()
        recorded = {}

        def record_image(board_id, result, request, **kwargs):
            recorded["board_id"] = board_id
            recorded["result"] = result
            recorded["request"] = request
            return {"id": "image-1", "image_url": "/outputs/image.png"}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = {
                "comfy_root": str(root),
                "model_dir": str(root / "models"),
                "text_encoder": str(root / "text_encoder.safetensors"),
                "vae": str(root / "vae.safetensors"),
                "output_dir": str(root / "outputs"),
            }
            state.runtime = StubRuntime()
            state.runtime_root = root.resolve()

            class MockEngine:
                def __init__(self, *_args):
                    pass

                def set_preset(self, preset):
                    self.preset = preset

                def set_dimensions(self, *_args):
                    pass

                def set_loras(self, _loras, on_change=None):
                    if on_loras is not None:
                        on_loras()

                def generate(self, **kwargs):
                    return {
                        "output_path": str(root / "outputs" / "image.png"),
                        "raw_prompt": kwargs["raw_prompt"],
                        "final_prompt": kwargs["prompt"],
                        "preset": self.preset.name,
                        "width": 1024,
                        "height": 1024,
                        "seed": kwargs["seed"],
                        "steps": kwargs["steps"],
                        "guidance": kwargs["guidance"],
                        **{key: kwargs[key] for key in ("raw_portion", "raw_steps") if key in kwargs},
                    }

            with patch.object(offcut_cli, "load_settings", return_value=settings), \
                 patch.object(offcut_server, "discover_loras", return_value=[]), \
                 patch.object(offcut_server, "file_fingerprint", return_value=("hash", 0, 0)), \
                 patch.object(offcut_cli, "resolve_checkpoint", return_value=root / "offcut.safetensors"), \
                 patch.object(
                     offcut_cli,
                     "prepare_generation",
                     side_effect=lambda args, _settings: (
                         offcut_cli.PRESETS[args.preset], [], args.prompt, "",
                     ),
                 ), \
                 patch.object(offcut_cli, "KreaEngine", MockEngine), \
                 patch.object(offcut_server, "output_url", return_value="/outputs/image.png"), \
                 patch.object(offcut_server, "public_image", side_effect=lambda image, _settings: image), \
                 patch.object(offcut_server.STORE, "get_board", return_value={"id": "inbox"}), \
                 patch.object(offcut_server.STORE, "record_image", side_effect=record_image):
                state.generate({"board_id": "inbox", "prompt": "a fox", **payload})
        return recorded

    def test_blank_steps_and_guidance_stay_blank_in_the_board_draft(self):
        recorded = self.run_generation({"steps": None, "guidance": None})
        # The sampler and the stored image keep the preset's resolved values...
        self.assertEqual(recorded["result"]["steps"], offcut_cli.PRESETS[offcut_cli.DEFAULT_PRESET].default_steps)
        self.assertEqual(recorded["result"]["guidance"], offcut_cli.PRESETS[offcut_cli.DEFAULT_PRESET].default_guidance)
        # ...but the board draft must not turn AUTO into that default.
        self.assertIsNone(recorded["request"]["steps"])
        self.assertIsNone(recorded["request"]["guidance"])

    def test_typed_steps_and_guidance_are_kept_in_the_board_draft(self):
        # raw-int8 because a distilled route refuses a typed guidance outright.
        recorded = self.run_generation({"preset": "raw-int8", "steps": 12, "guidance": 2.5})
        self.assertEqual(recorded["result"]["steps"], 12)
        self.assertEqual(recorded["request"]["steps"], 12)
        self.assertEqual(recorded["request"]["guidance"], 2.5)

    def test_a_typed_guidance_on_a_distilled_route_is_refused_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "raw-int8"):
            self.run_generation({"preset": "turbo-int8", "steps": 12, "guidance": 2.5})

    def test_a_distilled_route_still_accepts_its_own_default_guidance(self):
        recorded = self.run_generation({"preset": "turbo-int8", "guidance": 0.0})
        self.assertEqual(recorded["result"]["guidance"], 0.0)


class GenerationReuseTests(unittest.TestCase):
    """Sampling is deterministic in everything the run key covers, so re-running an unchanged form
    can only reproduce a frame the board already holds."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.store = Store(self.root / "offcut.sqlite3")
        self.sampled = []

    def tearDown(self):
        self.temp_dir.cleanup()

    def generate(self, state, payload):
        root = self.root
        sampled = self.sampled
        settings = {
            "comfy_root": str(root),
            "model_dir": str(root / "models"),
            "text_encoder": str(root / "text_encoder.safetensors"),
            "vae": str(root / "vae.safetensors"),
            "output_dir": str(self.outputs),
        }
        state.runtime = StubRuntime()
        state.runtime_root = root.resolve()

        class MockEngine:
            def __init__(self, *_args):
                pass

            def set_preset(self, preset):
                self.preset = preset

            def set_dimensions(self, *_args):
                pass

            def set_loras(self, *_args, **_kwargs):
                pass

            def generate(self, **kwargs):
                sampled.append(kwargs["prompt"])
                # A real run writes a PNG, and the reuse check reads the file back off disk.
                output_path = root / "outputs" / f"frame-{len(sampled)}.png"
                output_path.write_bytes(b"png")
                return {
                    "output_path": str(output_path),
                    "raw_prompt": kwargs["raw_prompt"],
                    "final_prompt": kwargs["prompt"],
                    "preset": self.preset.name,
                    "width": 1024,
                    "height": 1024,
                    "seed": kwargs["seed"],
                    "steps": kwargs["steps"],
                    "guidance": kwargs["guidance"],
                    **{key: kwargs[key] for key in ("raw_portion", "raw_steps") if key in kwargs},
                }

        with patch.object(offcut_cli, "load_settings", return_value=settings), \
             patch.object(offcut_server, "STORE", self.store), \
             patch.object(offcut_server, "discover_loras", return_value=[]), \
             patch.object(offcut_server, "file_fingerprint", return_value=("hash", 0, 0)), \
             patch.object(offcut_cli, "resolve_checkpoint", return_value=root / "offcut.safetensors"), \
             patch.object(
                 offcut_cli,
                 "prepare_generation",
                 side_effect=lambda args, _settings: (
                     offcut_cli.PRESETS[args.preset], [], args.prompt, "",
                 ),
             ), \
             patch.object(offcut_cli, "KreaEngine", MockEngine):
            return state.generate({"board_id": "inbox", "prompt": "a fox", "seed": 42, **payload})

    def test_an_identical_run_is_served_back_instead_of_resampled(self):
        state = offcut_server.AppState()
        first = self.generate(state, {})
        second = self.generate(state, {})

        self.assertEqual(self.sampled, ["a fox"])
        self.assertEqual(second["image_id"], first["image_id"])
        self.assertFalse(first["reused"])
        self.assertTrue(second["reused"])
        # The frame comes back addressable, not just acknowledged: the browser selects it by id
        # and draws it from the URL.
        self.assertEqual(second["image"]["id"], first["image_id"])
        self.assertEqual(second["output_path"], first["output_path"])
        self.assertEqual(state.get_progress()["image_id"], first["image_id"])
        self.assertFalse(state.get_progress()["active"])

    def test_a_changed_prompt_or_seed_still_samples(self):
        state = offcut_server.AppState()
        self.generate(state, {})
        self.generate(state, {"prompt": "a heron"})
        self.generate(state, {"seed": 43})

        self.assertEqual(self.sampled, ["a fox", "a heron", "a fox"])

    def test_force_resamples_a_run_the_board_already_holds(self):
        state = offcut_server.AppState()
        first = self.generate(state, {})
        second = self.generate(state, {"force": True})

        self.assertEqual(self.sampled, ["a fox", "a fox"])
        self.assertNotEqual(second["image_id"], first["image_id"])
        self.assertFalse(second["reused"])

    def test_a_frame_whose_file_is_gone_is_sampled_again(self):
        state = offcut_server.AppState()
        first = self.generate(state, {})
        Path(first["output_path"]).unlink()
        second = self.generate(state, {})

        # Handing back a path that no longer resolves would be worse than the wasted run.
        self.assertEqual(self.sampled, ["a fox", "a fox"])
        self.assertNotEqual(second["image_id"], first["image_id"])

    def test_each_board_is_deduplicated_against_its_own_frames(self):
        state = offcut_server.AppState()
        board = self.store.create_board("Studies")
        self.generate(state, {})
        # Asking for the same frame on another board is a request to put it there, not a repeat.
        self.generate(state, {"board_id": board["id"]})

        self.assertEqual(self.sampled, ["a fox", "a fox"])

    def test_the_run_key_ignores_where_a_frame_is_filed(self):
        shared = dict(
            preset_name="turbo-int8",
            final_prompt="a fox",
            negative_prompt="",
            width=1024,
            height=1024,
            seed=42,
            steps=8,
            guidance=0.0,
            loras=[],
            model_fingerprints=("checkpoint", "text_encoder", "vae"),
        )
        self.assertEqual(
            offcut_server.generation_run_key(**shared),
            offcut_server.generation_run_key(**shared),
        )
        self.assertNotEqual(
            offcut_server.generation_run_key(**shared),
            offcut_server.generation_run_key(**{**shared, "steps": 9}),
        )
        # Editing a checkpoint or a LoRA in place has to stop matching frames sampled before it.
        self.assertNotEqual(
            offcut_server.generation_run_key(**shared),
            offcut_server.generation_run_key(
                **{**shared, "model_fingerprints": ("checkpoint-v2", "text_encoder", "vae")}
            ),
        )


class NegativePromptRouteTests(unittest.TestCase):
    def test_only_the_undistilled_route_accepts_a_negative_prompt(self):
        self.assertTrue(offcut_server.preset_uses_negative_prompt("raw-int8"))
        self.assertFalse(offcut_server.preset_uses_negative_prompt("turbo-int8"))
        self.assertFalse(offcut_server.preset_uses_negative_prompt("raw-int8-turbo-lora"))
        self.assertFalse(offcut_server.preset_uses_negative_prompt("nonsense"))

    def test_setting_a_negative_prompt_on_a_distilled_route_is_refused(self):
        current = {"preset": "turbo-int8", "negative_prompt": ""}
        with self.assertRaisesRegex(ValueError, "Raw"):
            offcut_server.validate_chat_setting_patch({"negative_prompt": "blurry, watermark"}, current)

    def test_switching_to_a_distilled_route_clears_a_stranded_negative(self):
        current = {"preset": "raw-int8", "negative_prompt": "blurry, watermark"}
        candidate = offcut_server.validate_chat_setting_patch({"preset": "turbo-int8"}, current)
        self.assertEqual(candidate["preset"], offcut_cli.DEFAULT_PRESET)
        self.assertEqual(candidate["negative_prompt"], "")


class GuidanceRouteTests(unittest.TestCase):
    def test_only_the_undistilled_route_has_a_guidance_dial(self):
        self.assertTrue(offcut_server.preset_uses_guidance("raw-int8"))
        self.assertFalse(offcut_server.preset_uses_guidance("turbo-int8"))
        self.assertFalse(offcut_server.preset_uses_guidance("raw-int8-turbo-lora"))
        self.assertFalse(offcut_server.preset_uses_guidance("nonsense"))

    def test_setting_guidance_on_a_distilled_route_is_refused(self):
        for preset in ("turbo-int8", "raw-int8-turbo-lora"):
            with self.subTest(preset=preset):
                current = {"preset": preset, "guidance": 0.0}
                with self.assertRaisesRegex(ValueError, "raw-int8"):
                    offcut_server.validate_chat_setting_patch({"guidance": 3.5}, current)

    def test_setting_guidance_to_zero_on_a_distilled_route_is_a_no_op(self):
        current = {"preset": "turbo-int8", "guidance": 0.0}
        candidate = offcut_server.validate_chat_setting_patch({"guidance": 0.0}, current)
        self.assertEqual(candidate["guidance"], 0.0)

    def test_switching_to_a_distilled_route_resets_a_stranded_guidance(self):
        current = {"preset": "raw-int8", "guidance": 3.5}
        candidate = offcut_server.validate_chat_setting_patch({"preset": "turbo-int8"}, current)
        self.assertEqual(candidate["preset"], offcut_cli.DEFAULT_PRESET)
        self.assertIsNone(candidate["guidance"])

    def test_the_undistilled_route_still_accepts_guidance(self):
        current = {"preset": "raw-int8", "guidance": 3.5}
        candidate = offcut_server.validate_chat_setting_patch({"guidance": 5.0}, current)
        self.assertEqual(candidate["guidance"], 5.0)

    def test_switching_to_the_undistilled_route_keeps_its_guidance(self):
        current = {"preset": "turbo-int8", "guidance": 0.0}
        candidate = offcut_server.validate_chat_setting_patch(
            {"preset": "raw-int8", "guidance": 3.5}, current
        )
        self.assertEqual(candidate["guidance"], 3.5)


class LoraStrengthLimitTests(unittest.TestCase):
    def test_the_agent_schema_and_the_server_agree_on_the_limit(self):
        schema = (Path(__file__).resolve().parent.parent / "agent" / "src" / "tools.js").read_text()
        limit = int(offcut_cli.LORA_STRENGTH_LIMIT)
        self.assertIn(f"minimum: -{limit}", schema)
        self.assertIn(f"maximum: {limit}", schema)

    def test_the_browser_and_the_server_agree_on_the_limit_and_the_default(self):
        # The library chip's strength field and the LoRA editor both bound themselves in the
        # browser. A field narrower than this would silently forbid a value the sampler accepts.
        web = Path(__file__).resolve().parent.parent / "web"
        script = (web / "library.js").read_text()
        markup = (web / "index.html").read_text()
        limit = int(offcut_cli.LORA_STRENGTH_LIMIT)
        self.assertIn(f"const LORA_STRENGTH_LIMIT = {limit};", script)
        self.assertIn(f"const DEFAULT_LORA_STRENGTH = {offcut_store.DEFAULT_LORA_STRENGTH};", script)
        self.assertIn(f'min="-{limit}" max="{limit}"', markup)

    def test_a_strength_past_the_limit_is_refused(self):
        current = {"preset": "turbo-int8", "loras": []}
        over = offcut_cli.LORA_STRENGTH_LIMIT + 1
        with patch.object(
            offcut_server, "discover_loras", return_value=[{"name": "offcut_style"}]
        ):
            with self.assertRaisesRegex(ValueError, "LoRA strength"):
                offcut_server.validate_chat_setting_patch(
                    {"loras": [{"name": "offcut_style", "strength": over}]}, current
                )

    def test_a_strength_above_two_is_now_allowed(self):
        current = {"preset": "turbo-int8", "loras": []}
        with patch.object(
            offcut_server, "discover_loras", return_value=[{"name": "offcut_style"}]
        ):
            candidate = offcut_server.validate_chat_setting_patch(
                {"loras": [{"name": "offcut_style", "strength": 3.0}]}, current
            )
        self.assertEqual(candidate["loras"], [{"name": "offcut_style", "strength": 3.0}])

    def test_the_raw_route_keeps_its_negative_prompt(self):
        current = {"preset": "raw-int8", "negative_prompt": ""}
        candidate = offcut_server.validate_chat_setting_patch({"negative_prompt": "blurry"}, current)
        self.assertEqual(candidate["negative_prompt"], "blurry")
        kept = offcut_server.validate_chat_setting_patch({"steps": 40}, candidate)
        self.assertEqual(kept["negative_prompt"], "blurry")


class SkillLoaderTests(unittest.TestCase):
    def load_from(self, files):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, text in files.items():
                (root / name).mkdir(parents=True, exist_ok=True)
                (root / name / "SKILL.md").write_text(text, encoding="utf-8")
            with patch.object(offcut_server, "SKILLS_ROOT", root):
                return offcut_server.load_skills()

    def test_frontmatter_and_body_are_parsed(self):
        skills = self.load_from(
            {"save-style": "---\nname: save-style\ndescription: \"Save an art style\"\n---\n\nBody text here.\n"}
        )
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0]["name"], "save-style")
        self.assertEqual(skills[0]["description"], "Save an art style")
        self.assertEqual(skills[0]["body"], "Body text here.")

    def test_a_name_that_disagrees_with_its_directory_is_skipped(self):
        skills = self.load_from({"save-style": "---\nname: other-name\ndescription: d\n---\n\nbody\n"})
        self.assertEqual(skills, [])

    def test_a_skill_without_a_description_or_body_is_skipped(self):
        self.assertEqual(self.load_from({"a-skill": "---\nname: a-skill\n---\n\nbody\n"}), [])
        self.assertEqual(self.load_from({"b-skill": "---\nname: b-skill\ndescription: d\n---\n"}), [])

    def test_the_shipped_save_style_skill_loads(self):
        names = [skill["name"] for skill in offcut_server.load_skills()]
        self.assertIn("save-style", names)

    def test_the_system_prompt_section_lists_names_but_not_bodies(self):
        section = offcut_server.skills_section(
            [{"name": "save-style", "description": "Save an art style", "body": "SECRET BODY"}]
        )
        self.assertIn("save-style", section)
        self.assertIn("Save an art style", section)
        self.assertNotIn("SECRET BODY", section)

    def test_load_skill_resolves_a_name_and_rejects_a_path(self):
        with self.assertRaises(ValueError):
            offcut_server.get_skill("../../etc/passwd")
        with self.assertRaises(ValueError):
            offcut_server.get_skill("save-style/../save-style")
        self.assertEqual(offcut_server.get_skill("save-style")["name"], "save-style")


class GenerationCancelTests(unittest.TestCase):
    """Stopping is two things at once: a flag inside Comfy that only bites during a forward pass,
    and checkpoints around the phases the engine does not own."""

    def test_a_stop_with_nothing_running_is_refused_rather_than_left_armed(self):
        state = offcut_server.AppState()
        state.runtime = StubRuntime()

        self.assertEqual(state.request_cancel(), {"cancelled": False, "generation_id": None})
        # Arming the flag here would abort whatever started next instead.
        self.assertEqual(state.runtime.interrupts, [])
        self.assertFalse(state.cancel_requested)

    def test_a_stop_mid_run_arms_comfy_and_ends_the_run_as_cancelled(self):
        state = offcut_server.AppState()
        stops = {}

        def stop_now():
            stops["result"] = state.request_cancel()

        with self.assertRaises(offcut_cli.GenerationCancelled):
            GenerationDraftTests().run_generation({}, state=state, on_loras=stop_now)

        self.assertTrue(stops["result"]["cancelled"])
        self.assertTrue(stops["result"]["generation_id"])
        progress = state.get_progress()
        self.assertEqual(progress["stage"], "stopped")
        self.assertFalse(progress["active"])
        # Reported as a stop, not as a failure the user has to read an error for.
        self.assertIsNone(progress["error"])
        # Armed on the way in, and disarmed again on the way out so it cannot bite the next run.
        self.assertEqual(state.runtime.interrupts[-1], False)
        self.assertIn(True, state.runtime.interrupts)
        self.assertFalse(state.cancel_requested)
        self.assertFalse(state.busy)


class PublicImageSeedTests(unittest.TestCase):
    def test_a_seed_too_large_for_a_json_number_is_carried_as_text(self):
        seed = 3360211806803355277  # over 2 ** 53, which is where float64 stops being exact
        public = offcut_server.public_image({"id": "abc", "seed": seed}, {})

        self.assertEqual(public["seed_text"], "3360211806803355277")
        # The number stays for anything already reading it, but only the text survives the trip.
        self.assertEqual(public["seed"], seed)
        self.assertNotEqual(str(int(float(seed))), public["seed_text"])


class CoverRecipeTests(unittest.TestCase):
    """The recipe is what makes a wall of covers a comparison: every value the sampler reads is
    pinned here, so two covers differ only by the adapter under test."""

    def base(self):
        return dict(offcut_cli.DEFAULT_SETTINGS["cover"])

    def test_the_seed_survives_the_round_trip_as_text(self):
        # Over 2 ** 53, where a JSON number stops being exact and a reuse would resample.
        seed = 7193796597390363204
        recipe = offcut_server.public_cover_recipe({"cover": {**self.base(), "seed": seed}})
        self.assertEqual(recipe["seed"], str(seed))
        cover = self.base()
        offcut_server.update_cover_recipe(cover, {"seed": str(seed)})
        self.assertEqual(cover["seed"], seed)

    def test_the_default_prompt_is_offered_back_for_restoring(self):
        recipe = offcut_server.public_cover_recipe({"cover": {**self.base(), "prompt": "something else"}})
        self.assertEqual(recipe["prompt"], "something else")
        self.assertEqual(recipe["default_prompt"], offcut_cli.DEFAULT_COVER_PROMPT)

    def test_a_recipe_cannot_be_saved_in_a_state_generation_would_refuse(self):
        cover = self.base()
        # Guidance on a distilled route is refused at generation time, so it is refused here too:
        # saving it would leave every cover run failing with an error from somewhere else.
        with self.assertRaisesRegex(ValueError, "raw-int8"):
            offcut_server.update_cover_recipe(cover, {"preset": "turbo-int8", "guidance": 2.5})
        # The same pair is fine on the undistilled route.
        offcut_server.update_cover_recipe(cover, {"preset": "raw-int8", "guidance": 2.5})
        self.assertEqual(cover["preset"], "raw-int8")
        self.assertEqual(cover["guidance"], 2.5)
        # And moving back to a distilled route with the guidance still set is refused rather than
        # silently sampled.
        with self.assertRaisesRegex(ValueError, "raw-int8"):
            offcut_server.update_cover_recipe(cover, {"preset": "turbo-int8"})

    def test_the_showcase_strength_is_held_to_the_same_bound_as_everything_else(self):
        cover = self.base()
        offcut_server.update_cover_recipe(cover, {"lora_strength": 2.0})
        self.assertEqual(cover["lora_strength"], 2.0)
        with self.assertRaises(ValueError):
            offcut_server.update_cover_recipe(cover, {"lora_strength": offcut_cli.LORA_STRENGTH_LIMIT + 1})

    def test_the_default_prompt_names_no_medium(self):
        # A medium word here would compete with the style text or LoRA trigger prepended in front
        # of it, which is the one thing a cover must let the adapter decide.
        prompt = offcut_cli.DEFAULT_COVER_PROMPT.lower()
        for word in ("photo", "photograph", "painting", "illustration", "render", "anime", "pixel art"):
            self.assertNotIn(word, prompt)


class CoverGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp_dir.name) / "offcut.sqlite3")
        self.settings = {**offcut_cli.DEFAULT_SETTINGS, "cover": {
            **offcut_cli.DEFAULT_SETTINGS["cover"],
            "prompt": "the shared cover frame",
            "seed": 4242,
            "width": 1024,
            "height": 1024,
            "lora_strength": 2.0,
        }}

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_cover(self, target_kind, target, loras=()):
        sent = {}

        # The real STATE.generate is what records the row, and the store refuses to point a cover
        # at an image that does not exist, so the stub has to record one too.
        def generate(payload):
            sent.update(payload)
            image = self.store.record_image(
                payload["board_id"],
                {
                    "output_path": str(Path(self.temp_dir.name) / "cover.png"),
                    "raw_prompt": payload["prompt"],
                    "preset": payload["preset"],
                    "width": payload["width"],
                    "height": payload["height"],
                    "seed": int(payload["seed"]),
                    "steps": payload["steps"],
                    "guidance": payload["guidance"],
                },
                payload,
                kind=payload["kind"],
            )
            return {"image_id": image["id"]}

        with patch.object(offcut_cli, "load_settings", return_value=self.settings), \
             patch.object(offcut_server, "STORE", self.store), \
             patch.object(offcut_server, "discover_loras", return_value=list(loras)), \
             patch.object(offcut_server.STATE, "generate", side_effect=generate):
            outcome = offcut_server.generate_library_cover(target_kind, target)
        return sent, outcome

    def test_a_style_cover_takes_every_sampler_value_from_the_recipe(self):
        style = self.store.create_style("Ink Wash", "when you want ink", "monochrome ink wash", None, "user")
        sent, outcome = self.run_cover("style", style["id"])
        self.assertEqual(sent["prompt"], "the shared cover frame")
        self.assertEqual(sent["seed"], "4242")
        self.assertEqual(sent["width"], 1024)
        self.assertEqual(sent["height"], 1024)
        self.assertEqual(sent["styles"], [style["id"]])
        self.assertEqual(sent["loras"], [])
        # Recorded as a cover, so it stays out of the boards and the gallery.
        self.assertEqual(sent["kind"], "cover")
        # A rewritten prompt would vary per run and break the comparison the grid exists to make.
        self.assertFalse(sent["enhance"])
        image_id = outcome["image_id"]
        self.assertEqual(self.store.get_style(style["id"])["reference_image_id"], image_id)
        self.assertEqual(outcome["entry"]["reference_url"], f"/api/image-files/{image_id}")
        # The cover is a real row, but not one of the user's frames.
        self.assertEqual(self.store.get_image(image_id)["kind"], "cover")
        self.assertEqual(self.store.list_images(), [])

    def test_a_lora_cover_is_driven_at_the_recipes_showcase_strength(self):
        loras = [{"name": "my_ink_style", "trigger": "ink strokes", "path": "/tmp/x.safetensors"}]
        sent, outcome = self.run_cover("lora", "my_ink_style", loras=loras)
        # Not the profile default: a cover is meant to make the adapter's hand obvious.
        self.assertEqual(sent["loras"], [{"name": "my_ink_style", "strength": 2.0}])
        self.assertEqual(sent["styles"], [])
        self.assertEqual(sent["kind"], "cover")
        image_id = outcome["image_id"]
        self.assertEqual(outcome["entry"]["cover_image_id"], image_id)
        self.assertEqual(
            self.store.get_lora_profile("my_ink_style")["cover_image_id"], image_id
        )

    def test_an_unknown_target_is_refused_before_the_engine_is_touched(self):
        with self.assertRaises(ValueError):
            self.run_cover("lora", "not_a_lora")
        with self.assertRaises(ValueError):
            self.run_cover("nonsense", "whatever")

    def test_the_agents_tool_list_and_the_bridge_schema_agree(self):
        # A schema the bridge does not carry lets the model call a tool the server scopes to it
        # and get an unknown-tool error back instead of a result.
        schema = (Path(__file__).resolve().parent.parent / "agent" / "src" / "tools.js").read_text()
        self.assertIn('"generate_cover"', schema)
        self.assertIn("generate_cover: {", schema)

    def test_legacy_modes_expose_create_tools(self):
        self.assertIn("generate_cover", offcut_server.chat_tool_names("draft"))
        for mode in ("create", "iterate"):
            self.assertIn("generate_cover", offcut_server.chat_tool_names(mode))


class ThemeContrastTests(unittest.TestCase):
    """Every palette in themes.css has to keep its surface ladder actually stepping.

    Nine of the ten themes once set --theme-chat-user-bg to the same value as
    --theme-abyss-light, the chat panel the bubble is drawn on, so a user's own
    turns were invisible in every theme but offcut-dark. The ratios below are the
    ones the file's header documents; a new palette that flattens one of them
    fails here rather than in someone's eyes.
    """

    # (token, reference token, minimum WCAG contrast ratio)
    LADDER = (
        ("--theme-chat-user-bg", "--theme-abyss-light", 1.20),
        ("--theme-chat-user-border", "--theme-abyss-light", 1.55),
        ("--theme-mid-gray", "--theme-abyss-light", 1.45),
        ("--theme-abyss-light", "--theme-abyss", 1.07),
        ("--theme-code-bg", "--theme-abyss-light", 1.08),
        ("--theme-panel-flash", "--theme-code-bg", 1.08),
        ("--theme-dim-gray", "--theme-abyss-light", 4.5),
        ("--theme-input-placeholder", "--theme-abyss-light", 3.0),
    )

    @staticmethod
    def _channels(value):
        digits = value.strip().lstrip("#")
        if len(digits) == 3:
            digits = "".join(pair * 2 for pair in digits)
        return [int(digits[index:index + 2], 16) for index in (0, 2, 4)]

    @classmethod
    def _relative_luminance(cls, value):
        total = 0.0
        for channel, weight in zip(cls._channels(value), (0.2126, 0.7152, 0.0722)):
            level = channel / 255
            level = level / 12.92 if level <= 0.03928 else ((level + 0.055) / 1.055) ** 2.4
            total += level * weight
        return total

    @classmethod
    def _contrast(cls, first, second):
        one = cls._relative_luminance(first)
        two = cls._relative_luminance(second)
        return (max(one, two) + 0.05) / (min(one, two) + 0.05)

    @staticmethod
    def _palettes():
        source = (Path(__file__).resolve().parent.parent / "web" / "themes.css").read_text()
        found = {}
        for name, body in re.findall(r'\[data-theme="([^"]+)"\]\s*\{(.*?)\n\}', source, re.S):
            tokens = dict(re.findall(r"(--[\w-]+):\s*([^;]+);", body))
            if "--theme-abyss" in tokens:
                found[name] = tokens
        return found

    def test_every_theme_is_registered_in_the_catalog(self):
        catalog = (Path(__file__).resolve().parent.parent / "web" / "themes.js").read_text()
        palettes = self._palettes()
        self.assertGreaterEqual(len(palettes), 10)
        for name in palettes:
            self.assertIn(name, catalog, f"{name} has a palette but no catalog row")

    def test_every_theme_holds_the_documented_surface_ladder(self):
        for name, tokens in self._palettes().items():
            for token, reference, floor in self.LADDER:
                with self.subTest(theme=name, token=token):
                    ratio = self._contrast(tokens[token], tokens[reference])
                    self.assertGreaterEqual(
                        round(ratio, 2),
                        floor,
                        f"{name}: {token} sits {ratio:.2f} against {reference}",
                    )
