# Offcut

My own AI workspace, built because I don't really enjoy working in ComfyUI's node editor. It's mainly for me, but feel free to use it, take inspiration, or open a request. I'll take a look when I can.

It currently generates images with Krea 2 using ComfyUI under the hood. You don't need to run ComfyUI's server or use its editor.

## Screenshots

![Create workspace with generation controls and the Ideas image board](docs/workspace.png)

<details>
<summary>Chat and gallery</summary>

![Chat assistant revising a prompt and generating an image](docs/chat.png)

![Gallery of generated illustrations](docs/gallery.png)

</details>

## Setup

You'll need **Linux, Python 3.12, Git, Node.js 22.19+ and npm**. Tested on an **RTX 4090 (24 GB)**; **AMD RX 7900 XTX support is experimental and untested**. Allow about **19 GB for models**, plus dependencies and images.

Run these commands from the Offcut repository. The default setup expects a ComfyUI checkout beside it, with Python at `../ComfyUI/.venv/bin/python`.

<details>
<summary>Install ComfyUI and GPU dependencies</summary>

```bash
git clone https://github.com/Comfy-Org/ComfyUI.git ../ComfyUI
git -C ../ComfyUI checkout 9a9fdb10ed144ce760d9682cb247526ea23cc525
python3.12 -m venv ../ComfyUI/.venv
```

Install PyTorch for **one** GPU type:

**NVIDIA** — with a working NVIDIA driver:

```bash
../ComfyUI/.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

**AMD RX 7900 XTX** — first install the [ROCm 7.2 driver for your Linux distribution](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/native_linux/install-radeon.html), then use the [ROCm PyTorch wheels](https://pytorch.org/get-started/previous-versions/#v2110):

```bash
../ComfyUI/.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
../ComfyUI/.venv/bin/python -c 'import torch; print("ROCm:", torch.version.hip); print("GPU available:", torch.cuda.is_available())'
```

The check should print a ROCm version and `GPU available: True`. Offcut detects ROCm automatically and uses PyTorch quantization kernels instead of Triton. This may be slower; start with a single 512×512 Turbo image. Full generation still needs testing on the card. These instructions target native Linux.

Then, for either GPU:

```bash
../ComfyUI/.venv/bin/python -m pip install -r ../ComfyUI/requirements.txt
```

The pinned ComfyUI revision is the one used here. Other revisions need Krea 2 support and may require adjustments.

</details>

<details>
<summary>My ComfyUI installation is somewhere else</summary>

```bash
export OFFCUT_COMFY_ROOT="/path/to/ComfyUI"
export OFFCUT_PYTHON="$OFFCUT_COMFY_ROOT/.venv/bin/python"
./offcut-cli settings set comfy_root "$OFFCUT_COMFY_ROOT"
```

`OFFCUT_PYTHON` selects the interpreter; saved settings select the ComfyUI folder. Use `./offcut-cli settings show` to check your paths.

</details>

**1. Install the app dependency and download the model.**

```bash
"${OFFCUT_PYTHON:-../ComfyUI/.venv/bin/python}" -m pip install -r requirements.txt
./offcut-cli download turbo-int8
```

Download these separately into your ComfyUI folder:

| File | Put it in |
| --- | --- |
| [qwen3vl_4b_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) | `models/text_encoders/` |
| [qwen_image_vae.safetensors](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors) | `models/vae/` |

**2. Check the files and launch.**

```bash
./offcut-cli models
./offcut
```

Open **http://127.0.0.1:7862**. The first launch installs the Node dependencies automatically.

## Using it

- **Create** generates images; boards and the gallery keep them organized.
- **Styles** combines LoRAs with saved prompt recipes. Put Krea 2-compatible `.safetensors` files in `models/loras/`, refresh the library, and edit their triggers and strength there.
- **Chat** can write prompts, generate, and compare images. Add a provider connection in Settings to use it.

Generation stays on your GPU. Optional chat and enhancement send prompts and supplied images to your chosen provider. The app only listens on your own computer.

<details>
<summary>Other models and command-line use</summary>

```bash
./offcut-cli download raw-int8 turbo-lora
./offcut-cli generate "A red fox in fresh snow" --preset turbo-int8 --seed 42
./offcut-cli --help
```

Presets are `turbo-int8`, `raw-int8-turbo-lora`, and `raw-int8`. Only Raw uses negative prompts and adjustable guidance; Turbo keeps guidance at zero. The CLI doesn't need Node.

</details>

## Updating

Stop the app, then:

```bash
git pull
"${OFFCUT_PYTHON:-../ComfyUI/.venv/bin/python}" -m pip install -r requirements.txt
npm --prefix agent ci
./offcut
```

Your workspace stays local and is excluded from Git. Back it up separately. Generated PNGs include their prompts and settings.

<details>
<summary>Where things are saved</summary>

| Data | Location |
| --- | --- |
| Settings | `~/.config/offcut/settings.json` |
| Boards, chats and styles | `data/offcut.sqlite3` |
| Provider keys | `data/offcut.secrets.json` |
| Models / images | `models/` / `outputs/` |

Override the first three with `OFFCUT_CONFIG`, `OFFCUT_DATABASE`, and `OFFCUT_SECRETS`. Change `model_dir` and `output_dir` through `./offcut-cli settings set`.

</details>

## License

Copyright (C) 2026 Offcut contributors. [GPL-3.0-only](LICENSE). Models and dependencies have [their own terms](THIRD_PARTY_NOTICES.md). Not affiliated with Krea AI or Comfy Org.
