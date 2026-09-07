"""Sampling/recipe boundaries with real CPU tensors and no checkpoints or GPU work."""

import copy
import json
import tempfile
import unittest
import uuid
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import offcut_cli as cli
import offcut_server as server
from offcut_store import Store
import test_server as existing


class Interrupt(BaseException):
    pass


class Patcher:
    def __init__(self):
        self.model = SimpleNamespace(model_config={})
        self.patches = {}
        self.patches_uuid = uuid.uuid4()
        self.object_patches = {}
        self.model_options = {}
        self.load_device = "cpu"

    def clone(self):
        clone = copy.copy(self)
        clone.patches = self.patches.copy()
        clone.object_patches = self.object_patches.copy()
        return clone

    def add_object_patch(self, key, value):
        self.object_patches[key] = value


class Sampling:
    def __init__(self, config):
        pass

    def set_parameters(self, shift):
        self.shift = shift


class TwoStageEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.adapter = self.root / cli.DOWNLOADS["turbo-lora"].relative_path
        self.adapter.parent.mkdir(parents=True)
        self.adapter.write_bytes(b"turbo")
        self.style = self.root / "style.safetensors"
        self.style.write_bytes(b"style")
        engine = self.engine = object.__new__(cli.KreaEngine)
        engine.preset = cli.PRESETS[cli.HYBRID_PRESET]
        engine.settings = {"model_dir": str(self.root)}
        engine.width = engine.height = 256
        engine.base_model = Patcher()
        engine.lora_stacks = OrderedDict()
        engine.conditioning_cache = OrderedDict()
        engine.load_seconds = engine.load_peak_allocated = engine.load_peak_reserved = 0

        def load_lora(**kwargs):
            model = kwargs["model"].clone()
            model.patches_uuid = uuid.uuid4()
            model.patches[str(len(model.patches))] = [kwargs["lora"], kwargs["strength_model"]]
            return model, None

        def sample(**kwargs):
            kwargs["callback"](kwargs["steps"] - 1, None, None, kwargs["steps"])
            return kwargs["latent_image"] + 1

        engine.runtime = SimpleNamespace(
            torch=torch, Image=Image, PngInfo=PngInfo, synchronize=MagicMock(),
            model_sampling=SimpleNamespace(ModelSamplingFlux=Sampling, CONST=type("CONST", (), {})),
            sd=SimpleNamespace(load_lora_for_models=MagicMock(side_effect=load_lora)),
            utils=SimpleNamespace(load_torch_file=MagicMock(side_effect=lambda path, **kw: (path, {}))),
            samplers=SimpleNamespace(KSampler=lambda model, **kw: SimpleNamespace(sigmas=torch.linspace(1, 0, kw["steps"] + 1))),
            sample=SimpleNamespace(
                sample=MagicMock(side_effect=sample),
                fix_empty_latent_channels=lambda model, latent, **kw: latent,
                prepare_noise=lambda latent, seed: torch.ones_like(latent),
            ),
            mm=SimpleNamespace(
                throw_exception_if_processing_interrupted=MagicMock(),
                InterruptProcessingException=Interrupt, soft_empty_cache=MagicMock(),
                intermediate_device=lambda: "cpu", intermediate_dtype=lambda: torch.float32,
                unload_model_and_clones=MagicMock(),
            ),
        )
        engine.clip = SimpleNamespace(
            tokenize=lambda text: text,
            encode_from_tokens_scheduled=lambda tokens, **kw: [[torch.ones(2), {}]],
            patcher=object(),
        )

        def decode(samples):
            # Simulate tiled_scale's decorated return, then the in-place VAE postprocess.
            with torch.inference_mode():
                output = torch.zeros(1, 8, 8, 3)
            output.add_(0.5)
            return output

        engine.vae = SimpleNamespace(decode=MagicMock(side_effect=decode))
        for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
            patcher = patch.object(torch.cuda, name, return_value=0)
            patcher.start()
            self.addCleanup(patcher.stop)
        engine.set_loras([(self.style, 0.8)])

    def generate(self):
        return self.engine.generate("a fox", "a fox", self.root, 123, 12, 3.0, "watermark", raw_portion=8, raw_steps=52)

    def test_handoff_preserves_noise_level_and_only_raw_uses_cfg(self):
        result = self.generate()
        raw, turbo = [call.kwargs for call in self.engine.runtime.sample.sample.call_args_list]
        self.assertEqual(raw["cfg"], 4.0)
        self.assertEqual(turbo["cfg"], 1.0)
        self.assertGreater(float(raw["sigmas"][-1]), 0)
        self.assertEqual(float(raw["sigmas"][-1]), float(turbo["sigmas"][0]))
        self.assertEqual(float(turbo["sigmas"][-1]), 0)
        self.assertFalse(raw["force_full_denoise"])
        self.assertTrue(turbo["disable_noise"])
        self.assertEqual(torch.count_nonzero(turbo["noise"]), 0)
        self.assertTrue(torch.equal(turbo["latent_image"], raw["latent_image"] + 1))
        self.assertGreater(torch.count_nonzero(raw["negative"][0][0]), 0)
        self.assertEqual(torch.count_nonzero(turbo["negative"][0][0]), 0)
        self.assertEqual(result["sampling"]["raw_executed_steps"], 4)
        self.assertEqual(result["sampling"]["turbo_executed_steps"], 11)
        with Image.open(result["output_path"]) as frame:
            recipe = json.loads(frame.info["krea2"])
        self.assertEqual(recipe["negative_prompt"], "watermark")
        self.assertEqual((recipe["raw_portion"], recipe["raw_steps"]), (8, 52))
        self.assertEqual(recipe["sampling"], result["sampling"])

    def test_route_and_dimension_changes_reuse_both_patch_identities(self):
        engine = self.engine
        raw_uuid, turbo_uuid = engine.raw_model.patches_uuid, engine.turbo_model.patches_uuid
        self.assertEqual(len(engine.raw_model.patches), 1)
        self.assertEqual(len(engine.turbo_model.patches), 2)
        loads = engine.runtime.sd.load_lora_for_models.call_count
        for name in (cli.DEFAULT_PRESET, "raw-int8", cli.HYBRID_PRESET, cli.HYBRID_PRESET):
            engine.set_preset(cli.PRESETS[name])
            engine.set_dimensions(1024, 1024)
            engine.set_loras([(self.style, 0.8)])
        self.assertEqual(engine.runtime.sd.load_lora_for_models.call_count, loads)
        self.assertEqual(engine.raw_model.patches_uuid, raw_uuid)
        self.assertEqual(engine.turbo_model.patches_uuid, turbo_uuid)
        self.assertIs(engine.raw_model.model, engine.turbo_model.model)
        self.assertEqual(engine.base_model.patches, {})
        self.assertEqual(engine.base_model.object_patches, {})
        self.assertAlmostEqual(engine.raw_model.object_patches["model_sampling"].shift, cli.raw_sampling_shift(1024, 1024))
        self.assertEqual(engine.turbo_model.object_patches["model_sampling"].shift, 1.15)
        self.assertTrue(all(c.kwargs["strength_clip"] == 0 for c in engine.runtime.sd.load_lora_for_models.call_args_list))

    def test_changed_style_and_adapter_files_invalidate_the_correct_stacks(self):
        engine = self.engine
        raw_uuid, turbo_uuid = engine.raw_model.patches_uuid, engine.turbo_model.patches_uuid
        self.adapter.write_bytes(b"new turbo weights")
        engine.set_loras([(self.style, 0.8)])
        self.assertEqual(engine.raw_model.patches_uuid, raw_uuid)
        self.assertNotEqual(engine.turbo_model.patches_uuid, turbo_uuid)
        self.style.write_bytes(b"new style weights")
        engine.set_loras([(self.style, 0.8)])
        self.assertNotEqual(engine.raw_model.patches_uuid, raw_uuid)
        self.assertLessEqual(len(engine.lora_stacks), 2)

    def test_stop_between_stages_never_starts_turbo_or_decode(self):
        self.engine.runtime.mm.throw_exception_if_processing_interrupted.side_effect = [None, Interrupt()]
        with self.assertRaises(cli.GenerationCancelled):
            self.generate()
        self.assertEqual(self.engine.runtime.sample.sample.call_count, 1)
        self.engine.vae.decode.assert_not_called()

    def test_turbo_oom_retries_from_raw_with_the_same_seed(self):
        original = self.engine.runtime.sample.sample.side_effect
        calls = []

        def fail_once(**kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise torch.OutOfMemoryError("turbo OOM")
            return original(**kwargs)

        self.engine.runtime.sample.sample.side_effect = fail_once
        self.generate()
        self.assertEqual([c["cfg"] for c in calls], [4, 1, 4, 1])
        self.assertEqual([c["seed"] for c in calls], [123] * 4)
        self.assertTrue(torch.equal(calls[0]["latent_image"], calls[2]["latent_image"]))
        self.assertEqual(self.engine.runtime.mm.soft_empty_cache.call_count, 2)


class HybridRecipeTests(unittest.TestCase):
    def test_all_handoffs_are_descending_and_sigma_locked(self):
        for raw_steps in (2, 52, 100):
            for turbo_steps in (2, 12, 100):
                for portion in (0.01, 4, 8, 16.67, 30, 99.99):
                    raw, turbo = cli.split_sigma_schedules(torch.linspace(1, 0, raw_steps + 1), torch.linspace(1, 0, turbo_steps + 1), portion)
                    self.assertTrue(torch.all(raw[:-1] > raw[1:]))
                    self.assertTrue(torch.all(turbo[:-1] > turbo[1:]))
                    self.assertEqual(raw[-1], turbo[0])
                    self.assertGreater(raw[-1], 0)

    def test_invalid_hybrid_controls_are_rejected_before_loading(self):
        for patch_value in ({"raw_portion": 0}, {"raw_portion": 100}, {"raw_portion": float("nan")},
                            {"raw_portion": True}, {"raw_steps": 1}, {"raw_steps": 52.5}, {"raw_steps": True}, {"steps": 1}):
            with self.subTest(value=patch_value), self.assertRaises(ValueError):
                existing.GenerationDraftTests().run_generation({"preset": cli.HYBRID_PRESET, **patch_value})

    def test_hybrid_controls_reach_engine_and_board_recipe(self):
        recorded = existing.GenerationDraftTests().run_generation({"preset": cli.HYBRID_PRESET, "raw_portion": 16.67, "raw_steps": 60})
        self.assertEqual(recorded["result"]["raw_portion"], 16.67)
        self.assertEqual(recorded["result"]["raw_steps"], 60)
        self.assertEqual(recorded["result"]["guidance"], 3)
        self.assertEqual(recorded["request"]["raw_portion"], 16.67)

    def test_chat_defaults_and_stage_specific_validation(self):
        settings = server.validate_chat_setting_patch({"preset": cli.HYBRID_PRESET}, {"preset": cli.DEFAULT_PRESET, "steps": 8, "guidance": 0})
        self.assertEqual(settings["raw_portion"], 8)
        self.assertEqual(settings["raw_steps"], 52)
        self.assertIsNone(settings["steps"])
        self.assertIsNone(settings["guidance"])
        settings = server.validate_chat_setting_patch({"negative_prompt": "watermark", "guidance": 3}, settings)
        self.assertEqual(settings["negative_prompt"], "watermark")
        settings = server.validate_chat_setting_patch({"preset": cli.DEFAULT_PRESET}, settings)
        self.assertNotIn("raw_portion", settings)
        self.assertEqual(settings["negative_prompt"], "")
        with self.assertRaises(ValueError):
            server.validate_chat_setting_patch({"raw_portion": 8}, settings)

    def test_cover_hybrid_recipe_validation(self):
        cover = copy.deepcopy(cli.DEFAULT_SETTINGS["cover"])
        server.update_cover_recipe(cover, {"preset": cli.HYBRID_PRESET, "steps": 12, "guidance": 3, "raw_portion": 16.67, "raw_steps": 52})
        self.assertEqual(server.public_cover_recipe({"cover": cover})["raw_portion"], 16.67)
        with self.assertRaises(ValueError):
            server.update_cover_recipe(cover, {"raw_portion": 100})

    def test_cli_defaults_and_retired_route_alias(self):
        parser = cli.build_parser()
        self.assertEqual(parser.parse_args(["generate", "fox"]).preset, cli.DEFAULT_PRESET)
        self.assertEqual(parser.parse_args(["generate", "fox", "--preset", "turbo-int8"]).preset, cli.DEFAULT_PRESET)
        self.assertNotIn("turbo-int8", cli.PRESETS)
        self.assertNotIn("turbo-int8", cli.DOWNLOADS)


class HybridReuseTests(unittest.TestCase):
    setUp = existing.GenerationReuseTests.setUp
    tearDown = existing.GenerationReuseTests.tearDown
    generate = existing.GenerationReuseTests.generate
    def test_route_switches_share_engine_but_never_share_run_keys(self):
        state = server.AppState()
        first = self.generate(state, {})
        engine = state.engine
        for preset in ("raw-int8", cli.HYBRID_PRESET):
            self.generate(state, {"preset": preset})
            self.assertIs(state.engine, engine)
        again = self.generate(state, {})
        self.assertEqual(again["image_id"], first["image_id"])
        self.assertEqual(len(self.sampled), 3)

    def test_each_hybrid_input_changes_the_run_key_and_is_recorded(self):
        state = server.AppState()
        base = {"preset": cli.HYBRID_PRESET}
        first = self.generate(state, base)
        for change in ({"raw_portion": 16.67}, {"raw_steps": 60}, {"steps": 15}, {"guidance": 4}, {"negative_prompt": "watermark"}):
            result = self.generate(state, {**base, **change})
            self.assertFalse(result["reused"])
        again = self.generate(state, {**base, "raw_portion": 8, "raw_steps": 52})
        self.assertEqual(first["image_id"], again["image_id"])
        with patch.object(server, "STORE", self.store):
            image = server.curated_image(self.store.get_image(first["image_id"]))
        self.assertEqual((image["raw_portion"], image["raw_steps"]), (8, 52))

    def test_live_settings_migrate_without_rewriting_historical_recipes(self):
        self.store.update_board("inbox", {"settings": {"preset": "turbo-int8", "seed": "123"}})
        self.store.record_image("inbox", {"output_path": str(self.outputs / "old.png"), "preset": "turbo-int8"}, {}, kind="cover")
        revision = self.store.get_board("inbox")["settings_revision"]
        reopened = Store(self.store.path)
        self.assertEqual(reopened.get_board("inbox")["settings"]["preset"], cli.DEFAULT_PRESET)
        self.assertEqual(reopened.get_board("inbox")["settings"]["seed"], "123")
        self.assertEqual(reopened.list_images(kinds=("cover",))[0]["preset"], "turbo-int8")
        self.assertEqual(Store(self.store.path).get_board("inbox")["settings_revision"], revision + 1)
