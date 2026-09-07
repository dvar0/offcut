"""The UI preview must agree with the engine's schedule, including restored percentages."""
import json
import math
import subprocess
import unittest
from pathlib import Path

import torch
import offcut_cli as cli


class SamplingPreviewTests(unittest.TestCase):
    def test_browser_counts_match_flux_simple_schedules(self):
        # These are Comfy's ModelSamplingFlux and simple_scheduler operations on CPU tensors.
        # Test the UI's independent arithmetic against them without loading a Comfy runtime.
        def schedule(shift, steps):
            time = torch.arange(1, 10001) / 10000
            sigmas = math.exp(shift) / (math.exp(shift) + (1 / time - 1) ** 1.0)
            return torch.tensor([float(sigmas[-(1 + int(index * (10000 / steps)))]) for index in range(steps)] + [0.0])

        cases, expected = [], []
        for width, height in [(256, 256), (768, 1344), (1024, 1024), (1536, 1024), (2048, 2048)]:
            for raw_density, turbo_density in [(2, 2), (52, 12), (60, 15), (100, 100)]:
                raw_sigmas = schedule(cli.raw_sampling_shift(width, height), raw_density)
                turbo_sigmas = schedule(1.15, turbo_density)
                for portion in [0.01, 8, 16.67, 30, 99.99, min(2.5, raw_density - 0.5) / raw_density * 100]:
                    raw, turbo = cli.split_sigma_schedules(raw_sigmas, turbo_sigmas, portion)
                    cases.append([width, height, portion, raw_density, turbo_density])
                    expected.append({"raw": len(raw) - 1, "turbo": len(turbo) - 1})
        result = subprocess.run(
            ["node", "--input-type=module", "-e",
             'import {hybridStepPlan} from "./web/sampling.js"; let text = ""; for await (const chunk of process.stdin) text += chunk; console.log(JSON.stringify(JSON.parse(text).map(args => hybridStepPlan(...args))));'],
            cwd=Path(__file__).resolve().parents[1], input=json.dumps(cases), text=True, capture_output=True, check=True,
        )
        for args, actual, wanted in zip(cases, json.loads(result.stdout), expected):
            with self.subTest(args=args):
                self.assertEqual(actual, wanted)
                self.assertEqual(cli.hybrid_step_plan(*args), wanted)
