import json
import os
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import MagicMock

import offcut_cli


class SettingsTests(unittest.TestCase):
    def test_setting_runtime_moves_conventional_paths_but_preserves_custom_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "settings.json"
            with patch.dict(os.environ, {"OFFCUT_CONFIG": str(config), "OFFCUT_COMFY_ROOT": str(Path(directory) / "old")}):
                settings = offcut_cli.load_settings()
                settings["text_encoder"] = "/custom/encoder.safetensors"
                settings["lora_dirs"].append("/custom/loras")
                new_root = Path(directory) / "new"
                with patch("builtins.print"):
                    offcut_cli.run_settings(SimpleNamespace(settings_command="set", key="comfy_root", value=str(new_root)), settings)
                saved = offcut_cli.load_settings()
            self.assertEqual(saved["comfy_root"], str(new_root))
            self.assertEqual(saved["text_encoder"], "/custom/encoder.safetensors")
            self.assertEqual(saved["vae"], str(new_root / "models/vae/qwen_image_vae.safetensors"))
            self.assertIn(str(new_root / "models/loras"), saved["lora_dirs"])
            self.assertIn("/custom/loras", saved["lora_dirs"])

    def test_fresh_model_paths_follow_environment_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "Comfy checkout"
            with patch.dict(os.environ, {"OFFCUT_CONFIG": str(Path(temp_dir) / "settings.json"), "OFFCUT_COMFY_ROOT": str(root)}):
                settings = offcut_cli.load_settings()
            self.assertEqual(settings["comfy_root"], str(root))
            self.assertEqual(settings["text_encoder"], str(root / "models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"))
            self.assertEqual(settings["vae"], str(root / "models/vae/qwen_image_vae.safetensors"))
            self.assertIn(str(root / "models/loras"), settings["lora_dirs"])

    def test_saved_paths_win_and_unspecified_paths_follow_saved_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "saved-runtime"
            config = Path(temp_dir) / "settings.json"
            saved = {"comfy_root": str(root), "text_encoder": "/custom/encoder.safetensors", "lora_dirs": ["/custom/loras"]}
            config.write_text(json.dumps(saved))
            with patch.dict(os.environ, {"OFFCUT_CONFIG": str(config), "OFFCUT_COMFY_ROOT": "/other/runtime"}):
                settings = offcut_cli.load_settings()
            for key, value in saved.items():
                self.assertEqual(settings[key], value)
            self.assertEqual(settings["vae"], str(root / "models/vae/qwen_image_vae.safetensors"))
            self.assertEqual(json.loads(config.read_text()), saved)

    def test_deep_merge_preserves_enhancer_defaults(self):
        merged = offcut_cli.deep_merge(
            offcut_cli.DEFAULT_SETTINGS,
            {"enhancer": {"model": "test-model"}},
        )
        self.assertEqual(merged["enhancer"]["model"], "test-model")
        self.assertIn("endpoint", merged["enhancer"])

    def test_settings_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "settings.json"
            with patch.dict(os.environ, {"OFFCUT_CONFIG": str(path)}):
                settings = offcut_cli.load_settings()
                offcut_cli.set_nested_value(settings, "enhancer.model", "local-model")
                offcut_cli.save_settings(settings)
                self.assertEqual(offcut_cli.load_settings()["enhancer"]["model"], "local-model")
                self.assertEqual(json.loads(path.read_text())["enhancer"]["model"], "local-model")

    def test_loaded_settings_do_not_mutate_defaults(self):
        settings = offcut_cli.deep_merge({}, offcut_cli.DEFAULT_SETTINGS)
        settings["enhancer"]["model"] = "changed"
        self.assertNotEqual(offcut_cli.DEFAULT_SETTINGS["enhancer"]["model"], "changed")


class PromptTests(unittest.TestCase):
    def test_prefix_prompt_deduplicates_triggers(self):
        result = offcut_cli.prefix_prompt("a fox", ["ink style", "ink style"])
        self.assertEqual(result, "ink style, a fox")

    def test_prefix_prompt_does_not_repeat_existing_prefix(self):
        result = offcut_cli.prefix_prompt("ink style, a fox", ["ink style"])
        self.assertEqual(result, "ink style, a fox")

    def test_clean_enhanced_prompt(self):
        result = offcut_cli.clean_enhanced_prompt("```text\nFinal prompt: a detailed fox\n```")
        self.assertEqual(result, "a detailed fox")

    def test_raw_sampling_shift_uses_official_resolution_anchors(self):
        self.assertAlmostEqual(offcut_cli.raw_sampling_shift(256, 256), 0.5)
        self.assertAlmostEqual(offcut_cli.raw_sampling_shift(1280, 1280), 1.15)

    def test_openai_compatible_enhancer_preserves_trigger(self):
        settings = offcut_cli.deep_merge({}, offcut_cli.DEFAULT_SETTINGS)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "a detailed fox"}}]}
        ).encode()
        opener = MagicMock()
        opener.open.return_value = response
        with patch.dict(os.environ, {"OPENCODE_GO_API_KEY": "test-key"}):
            with patch("urllib.request.build_opener", return_value=opener):
                result = offcut_cli.enhance_prompt("ink style, a fox", "ink style", settings)
        self.assertEqual(result, "ink style, a detailed fox")
        request = opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], settings["enhancer"]["model"])
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")


class LoraTests(unittest.TestCase):
    def test_split_lora_strength(self):
        self.assertEqual(offcut_cli.split_lora_spec("style:0.8"), ("style", 0.8))

    def test_split_lora_path_without_strength(self):
        self.assertEqual(
            offcut_cli.split_lora_spec("/tmp/style.safetensors"),
            ("/tmp/style.safetensors", 1.0),
        )

    def test_fresh_metadata_has_no_built_in_trigger_or_notes(self):
        self.assertEqual(offcut_cli.DEFAULT_SETTINGS["lora_metadata"], {})
        for name in ("my_ink_style", "another_adapter"):
            self.assertEqual(offcut_cli.lora_metadata(offcut_cli.DEFAULT_SETTINGS, name),
                             {"trigger": "", "summary": "", "prompting_notes": ""})

    def test_configured_trigger_is_used_by_cli_generation_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "my_ink_style.safetensors"
            path.touch()
            settings = offcut_cli.deep_merge(offcut_cli.DEFAULT_SETTINGS, {
                "lora_dirs": [directory], "lora_metadata": {path.stem: {"trigger": "ink strokes"}},
            })
            args = SimpleNamespace(preset="turbo-int8", prompt="ink strokes, a fox", lora=["my_ink_style:0.7"], trigger=[], enhance=False)
            _, loras, prompt, _ = offcut_cli.prepare_generation(args, settings)
            self.assertEqual(loras, [(path, 0.7)])
            self.assertEqual(prompt, "ink strokes, a fox")
            settings["lora_metadata"][path.stem]["trigger"] = ""
            args.prompt = "a fox"
            self.assertEqual(offcut_cli.prepare_generation(args, settings)[2], "a fox")

    def test_cli_enhancer_receives_only_active_lora_notes(self):
        settings = offcut_cli.deep_merge(offcut_cli.DEFAULT_SETTINGS, {"lora_metadata": {
            "active": {"summary": "Ink", "prompting_notes": "Keep sentences short."},
            "inactive": {"prompting_notes": "This must stay out of the request."},
        }})
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = json.dumps({"choices": [{"message": {"content": "a fox"}}]}).encode()
        with patch.dict(os.environ, {"OPENCODE_GO_API_KEY": "test-key"}), patch("urllib.request.build_opener", return_value=opener):
            offcut_cli.enhance_prompt("a fox", "", settings, lora_names=["active", "active"])
        content = json.loads(opener.open.call_args.args[0].data)["messages"][1]["content"]
        self.assertEqual(content.count("Keep sentences short."), 1)
        self.assertNotIn("This must stay out", content)
        self.assertIn("not requirements or verified training facts", content)


# Comfy's own stop signal is a BaseException so that nothing in the sampling stack swallows it.
# A MagicMock attribute cannot stand in for it: the except clause needs a real class.
class StubInterrupt(BaseException):
    pass


class EngineTests(unittest.TestCase):
    @staticmethod
    def stub_engine(side_effect):
        import torch

        engine = object.__new__(offcut_cli.KreaEngine)
        engine.runtime = SimpleNamespace(torch=torch, mm=MagicMock())
        engine.runtime.mm.InterruptProcessingException = StubInterrupt
        engine._generate = MagicMock(side_effect=side_effect)
        return engine

    def test_an_interrupt_becomes_a_cancellation_and_is_not_retried(self):
        engine = self.stub_engine(StubInterrupt())

        with self.assertRaises(offcut_cli.GenerationCancelled):
            engine.generate("prompt", "prompt", Path("."), 1, 4, 1.0, "")

        # A stop is not an out-of-memory failure, so it takes the single attempt it was given...
        self.assertEqual(engine._generate.call_count, 1)
        # ...and still hands the allocator cache back.
        engine.runtime.mm.soft_empty_cache.assert_called_once_with()

    def test_generate_disables_grad_without_creating_inference_tensors(self):
        import torch

        engine = self.stub_engine(
            lambda *args: (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        )

        state = engine.generate("prompt", "prompt", Path("."), 1, 4, 1.0, "")

        self.assertEqual(state, (False, False))

    def test_generate_trims_the_allocator_cache_even_when_it_fails(self):
        engine = self.stub_engine(ValueError("bad prompt"))

        with self.assertRaises(ValueError):
            engine.generate("prompt", "prompt", Path("."), 1, 4, 1.0, "")

        # A non-OOM failure must not be retried, but the cache is still handed back.
        self.assertEqual(engine._generate.call_count, 1)
        engine.runtime.mm.soft_empty_cache.assert_called_once_with()

    def test_generate_retries_once_after_trimming_on_out_of_memory(self):
        import torch

        attempts = []

        def flaky(*_args):
            attempts.append(1)
            if len(attempts) == 1:
                raise torch.OutOfMemoryError("out of memory")
            return "image"

        engine = self.stub_engine(flaky)

        self.assertEqual(engine.generate("prompt", "prompt", Path("."), 1, 4, 1.0, ""), "image")
        self.assertEqual(engine._generate.call_count, 2)
        # Once before the retry, once on the way out.
        self.assertEqual(engine.runtime.mm.soft_empty_cache.call_count, 2)

    def test_generate_gives_up_after_one_retry(self):
        import torch

        engine = self.stub_engine(torch.OutOfMemoryError("out of memory"))

        with self.assertRaises(torch.OutOfMemoryError):
            engine.generate("prompt", "prompt", Path("."), 1, 4, 1.0, "")

        self.assertEqual(engine._generate.call_count, 2)

    def test_unchanged_lora_stack_does_no_work(self):
        engine = object.__new__(offcut_cli.KreaEngine)
        engine.preset = offcut_cli.PRESETS["turbo-int8"]
        engine.lora_key = ()
        engine.runtime = None  # Any actual reload work would raise on this.
        changes = []

        engine.set_loras([], on_change=lambda: changes.append(1))

        self.assertEqual(changes, [])

    def test_changed_lora_stack_repatches_the_resident_base_model(self):
        base = SimpleNamespace(name="base", patches={})
        patched = SimpleNamespace(name="patched", patches={"weight": ["delta"]})
        engine = object.__new__(offcut_cli.KreaEngine)
        engine.preset = offcut_cli.PRESETS["turbo-int8"]
        engine.lora_key = ()
        engine.base_model = base
        engine.model = base
        engine.applied_loras = []
        engine.runtime = SimpleNamespace(
            utils=SimpleNamespace(load_torch_file=MagicMock(return_value=({}, {}))),
            sd=SimpleNamespace(load_lora_for_models=MagicMock(return_value=(patched, None))),
        )
        changes = []

        with tempfile.TemporaryDirectory() as temp_dir:
            lora = Path(temp_dir) / "style.safetensors"
            lora.write_bytes(b"weights")
            engine.set_loras([(lora, 0.8)], on_change=lambda: changes.append(1))

            # Cloned from the untouched base, so the checkpoint is never reloaded, and CLIP is
            # left alone so the conditioning cache survives the swap.
            call = engine.runtime.sd.load_lora_for_models.call_args.kwargs
            self.assertIs(call["model"], base)
            self.assertIsNone(call["clip"])
            self.assertEqual(call["strength_clip"], 0.0)

        self.assertIs(engine.model, patched)
        self.assertIs(engine.base_model, base)
        self.assertEqual(changes, [1])
        self.assertEqual(engine.applied_loras[0]["strength"], 0.8)

    def test_dimension_change_only_repatches_raw_sampling(self):
        engine = object.__new__(offcut_cli.KreaEngine)
        engine.width = 1024
        engine.height = 1024
        engine.preset = offcut_cli.PRESETS["raw-int8"]
        engine._patch_raw_sampling = MagicMock()

        engine.set_dimensions(1216, 832)

        self.assertEqual((engine.width, engine.height), (1216, 832))
        engine._patch_raw_sampling.assert_called_once_with()


class ConditioningCacheTests(unittest.TestCase):
    """A miss costs a full model round trip -- the text encoder in, the checkpoint out and back --
    so what matters is that going back to an earlier prompt still hits."""

    @staticmethod
    def stub_engine():
        import torch

        engine = object.__new__(offcut_cli.KreaEngine)
        engine.runtime = SimpleNamespace(torch=torch)
        engine.conditioning_cache = OrderedDict()
        return engine

    @staticmethod
    def conditioning(elements=4):
        import torch

        return [[torch.zeros(elements, dtype=torch.float32), {}]]

    def remember(self, engine, prompt, elements=4):
        conditioning = self.conditioning(elements)
        engine._remember_conditioning((prompt, "", True), conditioning, [])
        return conditioning

    def keys(self, engine):
        return [key[0] for key in engine.conditioning_cache]

    def test_more_than_one_prompt_is_kept(self):
        engine = self.stub_engine()
        for prompt in ("a fox", "a heron", "a fox in snow"):
            self.remember(engine, prompt)

        # The old cache held exactly one entry, so alternating between two prompts re-encoded
        # every time even though each conditioning is a few MB of CPU tensors.
        self.assertEqual(self.keys(engine), ["a fox", "a heron", "a fox in snow"])

    def test_the_least_recently_used_prompt_is_the_one_evicted(self):
        engine = self.stub_engine()
        with patch.object(offcut_cli, "CONDITIONING_CACHE_ENTRIES", 3):
            for prompt in ("a fox", "a heron", "a hare"):
                self.remember(engine, prompt)
            # A hit moves an entry to the back of the queue, which is what keeps a prompt someone
            # keeps returning to from ageing out behind ones they typed once.
            engine.conditioning_cache.move_to_end(("a fox", "", True))
            self.remember(engine, "a stoat")

        self.assertEqual(self.keys(engine), ["a hare", "a fox", "a stoat"])

    def test_the_byte_ceiling_evicts_before_the_entry_ceiling_is_reached(self):
        engine = self.stub_engine()
        # Room for exactly two of these 128-byte entries, well inside the entry ceiling.
        with patch.object(offcut_cli, "CONDITIONING_CACHE_BYTES", 2 * 32 * 4):
            for prompt in ("a fox", "a heron", "a hare"):
                self.remember(engine, prompt, elements=32)

        self.assertEqual(self.keys(engine), ["a heron", "a hare"])

    def test_the_newest_entry_survives_even_when_it_alone_is_oversized(self):
        engine = self.stub_engine()
        with patch.object(offcut_cli, "CONDITIONING_CACHE_BYTES", 8):
            self.remember(engine, "a fox", elements=64)

        # The run about to sample is holding this conditioning regardless, so dropping it frees
        # nothing and only guarantees a re-encode if the same prompt comes round again.
        self.assertEqual(self.keys(engine), ["a fox"])

    def test_an_entry_is_measured_across_its_metadata_tensors(self):
        import torch

        engine = self.stub_engine()
        conditioning = [[torch.zeros(4, dtype=torch.float32), {"pooled_output": torch.zeros(6)}]]

        # 4 + 6 float32 elements: pooled_output is real weight and has to count toward the ceiling.
        self.assertEqual(offcut_cli.conditioning_bytes(engine.runtime, conditioning), 40)


if __name__ == "__main__":
    unittest.main()
