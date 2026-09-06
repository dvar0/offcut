# Third-party software and model notices

The application code in this repository is licensed under GPL-3.0-only; see [LICENSE](LICENSE). Third-party software and model files retain their own copyright notices and licenses.

| Dependency | Use | License / source |
| --- | --- | --- |
| ComfyUI | Python inference runtime, loaded from a separate checkout | [GPLv3](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE) |
| Pi agent core and Pi AI | Creative chat state and provider adapters, installed through npm | MIT; [Pi repository](https://github.com/earendil-works/pi) |
| Pillow | Image metadata, thumbnails, and reference processing | [HPND and accompanying notices](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| PyTorch and the ComfyUI dependency stack | Tensor operations, CUDA, and model loading | See each installed package's license and the [ComfyUI requirements](https://github.com/Comfy-Org/ComfyUI/blob/master/requirements.txt) |

Node packages are installed from `agent/package-lock.json`; their sources and license notices are supplied by their respective packages. This repository does not vendor `node_modules`, ComfyUI, or development-only agent skills. If distributing a bundled application, include the dependencies' required notices and satisfy their source-distribution requirements too.

## Model weights

Krea 2 checkpoints and official adapters are available from [Krea AI](https://huggingface.co/krea) and [Comfy Org's repackaged model repository](https://huggingface.co/Comfy-Org/Krea-2). The model repository identifies its license as the **Krea 2 Community License**. Read the applicable model card and license before downloading or using the weights; the app's GPL license does not replace those terms.

The text encoder, VAE, and any additional LoRAs retain their respective upstream terms. No model weights, training datasets, or personal LoRAs are included in this repository. A supported filename or trigger phrase does not imply that an adapter is distributed here.

Krea and ComfyUI names identify compatible upstream projects. This is an independent, unofficial application.
