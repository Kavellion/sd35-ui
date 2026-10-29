# SD 3.5 Medium UI

## Before you start

- **This model is GATED — two steps, both required, before the first Generate can succeed:**
  1. Go to the [SD 3.5 Medium model page](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium), sign in, and **accept the Stability AI Community License** there.
  2. Create a Hugging Face **read token** on the [Access Tokens page](https://huggingface.co/settings/tokens), and **paste it into the token box in the app**. Same account for both steps.
  Without both, Hugging Face refuses the download and Generate reports that plainly. No way around it — that is Stability's gate, not a bug in this app.
- A Windows PC. No Python install needed — Python 3.12 is included in this package, in the `python_embeded` folder, ComfyUI portable style. Any Python you already have is left alone and is not needed.
- An NVIDIA graphics card (12 GB is plenty), about 20 GB free disk space, and an internet connection.
- The first Generate downloads the model weights: about 6.5 GB on the default CLIP-only load (the T5 files are skipped), about 16 GB if you switch to T5 full.

**SD 3.5 Medium, without Jupyter and without Colab.**

Stable Diffusion 3.5 Medium is Stability AI's 2.5B MMDiT image model: strong prompt following (three text encoders — CLIP-L, CLIP-G, and T5-XXL), good text rendering for its size, and a look of its own next to Kandinsky, Sana, and Z-Image. This is a plain app: run it, a page opens in your browser, type a prompt, pictures appear. Built in the same shape as Kandinsky UI, Sana UI, and Z-Image UI, free forever.

![The app](ui-screenshot.png)

*The app*

![Example render — porcelain cyborg (CLIP-only load)](example-porcelain-cyborg.png)

*Example render — porcelain cyborg (CLIP-only load)*

![Example render — porcelain cyborg, full prompt (T5 full load)](example-circuit-cyborg.png)

*Example render — porcelain cyborg, full prompt (T5 full load)*

![Example render — porcelain cyborg, dark metal (T5 full load)](example-dark-metal-cyborg.jpg)

*Example render — porcelain cyborg, dark metal (T5 full load)*

![Example render — porcelain cyborg with medallion (T5 full load)](example-medallion-cyborg.jpg)

*Example render — porcelain cyborg with medallion (T5 full load)*

![Example render — opal lucifer princess (T5 full load)](example-opal-flame-profile.jpg)

*Example render — opal lucifer princess (T5 full load)*

## What it does

- Text-to-image generation on your own GPU -- no cloud, no Stability account (a free Hugging Face account is needed for the gate, above)
- Prompt + negative prompt, steps slider (default 40), guidance slider (default 4.5) — the model card's own snippet values
- Text-encoder choice: **No T5 — CLIP only (the default, lighter)** or **T5 full (best prompt understanding, heaviest RAM)** — see Requirements for why the default is what it is
- CPU offload toggle (default ON) — together with the CLIP-only default, that is what makes this fit a 12 GB card with 16 GB of RAM
- Two LoRA slots with weight sliders (0.0–1.5, default 0.8) — SD 3.5 LoRA files (.safetensors) dropped in the `loras` folder are picked in the dropdowns; see **LoRAs** below
- Hugging Face token box (password field — the token is used for the download and never written to settings.txt or anywhere else)
- Size presets (Square 1024, Small square 768, Portrait 768x1024, Landscape 1024x768) plus custom width/height boxes, up to 2048
- Up to 10 pictures per prompt, each with its own seed, saved as 000001.png, 000002.png, and so on
- Seed shown in the status line (-1 = random; a fixed seed S gives S, S+1, S+2, ... across a multi-image run)
- A Random prompt button (word banks built in, no list file), and a session-only prompt history dropdown
- A Stop button to cancel a run, and Star batch to copy a batch into images_out/favorites/
- A prompt queue (one prompt per line -- when it has lines, Generate runs the queue instead of the single prompt box, all lines sharing the one batch folder, frames numbered continuously)
- Dynamic prompts: {a|b|c} groups, one random pick per group, re-expanded for every single image (prompt and negative prompt). settings.txt records the expanded prompt actually used
- An Unload models button to free VRAM and RAM, and memory freed between every image in a run (long queues keep working)
- Every run adds its pictures to the batch folder and appends a block to `settings.txt` next to them (date, time, frame range, settings including T5 mode, offload and LoRA state, prompt, negative prompt, expanded prompts, seeds)

No animation, no upscaling, no editing, no ControlNet, no inpainting, no img2img -- v1 is text-to-image only, kept simple on purpose.

## LoRAs

Two slots, each with a weight slider (0.0 to 1.5, default 0.8 — 0.8 to 1.0 is the usual working range; lower is subtler, higher is stronger and can start to fight the prompt).

1. Drop `.safetensors` LoRA files into the `loras` folder. It sits **next to app.py, inside the `sd35-ui` subfolder** of the app folder (the folder that holds `run.bat` has an `sd35-ui` folder inside it; `app.py` and `loras` are in there).
2. Pick a file in the **LoRA 1** / **LoRA 2** dropdowns — hit **Refresh list** after adding a file while the app is running — set the weights, and Generate. `None` in a slot means no LoRA in that slot.
3. Each run's `settings.txt` block records what was applied, e.g. `loras: my-style.safetensors @ 0.8`. Switching the picks between runs unloads the old adapters first, so a LoRA never leaks from one run into the next.

**Only SD 3.5 LoRAs fit.** SDXL and SD 1.5 LoRAs are built for different models and do not work here — a file that is not a LoRA checkpoint at all is refused by the loader with a plain status message, and a wrong-model LoRA is not a sensible render waiting to happen. SD 3.5 LoRAs may carry transformer weights, CLIP text-encoder weights, or both; whichever parts a file has are the parts that get applied. The slots work the same in both text-encoder modes: a LoRA's text-encoder parts land on the two CLIP encoders, and T5 itself never takes a LoRA (Diffusers' SD3 loader has no T5 LoRA path), so the default CLIP-only load is no obstacle.

**Format matters (honest limit, Oct 29, 2026).** The slots load **Diffusers-format** LoRAs — files whose weights are keyed `transformer.` / `text_encoder.` / `text_encoder_2.`, typically named `pytorch_lora_weights.safetensors` (DreamBooth/Diffusers exports). **Kohya / sd-scripts format files do NOT load yet**: those are keyed `lora_unet_joint_blocks_...`, the Diffusers SD3 loader performs no conversion for them, and the app refuses them with a plain status message instead of rendering. Most SD 3.5 LoRAs on Civitai are Kohya format, so check the file before downloading; a converter is planned for a later update.

**PC-proven Oct 29, 2026** (author's machine, RTX 4070 12 GB / 16 GB RAM): a Diffusers-format LoRA in slot 1 at 0.8 vs the same prompt and seed with the slot on None produced clearly different pictures, and `settings.txt` recorded the applied LoRA.

## Requirements

- Windows, an NVIDIA GPU (12 GB is plenty), and a recent driver
- Python 3.12 is bundled in the `python_embeded` folder: Diffusers 0.42 dropped Python 3.10, so this app runs on its own included 3.12 and nothing else. No Python install, no PATH changes, no venv — your other Pythons are left completely alone, and this app lives entirely in this folder
- Disk space: about 20 GB free for the first download, plus room for pictures
- System RAM: this is the honest part. Stability says SD 3.5 Medium needs 9.9 GB of VRAM *excluding* the text encoders, and with CPU offload the weights live in **system RAM** while one component at a time uses the GPU. T5-XXL alone is about 9.4 GB of weights, so the full three-encoder set is about 16 GB — on a machine with 16 GB of RAM total, T5 full will page hard or run out of memory; it is not the default for that reason. The default CLIP-only load is about 6.5 GB of weights and fits comfortably in 16 GB of RAM.
- What you give up with CLIP only: the two CLIP encoders carry the prompt in 77-token windows and the pipeline feeds zeros where the T5 embeddings would go (a documented Diffusers SD3 pattern). Short and medium prompts work well; long, complex, multi-clause prompts follow noticeably less closely than with T5. If you have 32 GB of RAM, T5 full is the better load — switch the radio and the next Generate reloads.

## Install

1. Do the two gate steps in "Before you start" (license acceptance + read token). Install works without them, but the first Generate will not.
2. Extract the whole zip somewhere plain, like `C:\AI\sd35-ui` — the folder holding `install.bat` is the app folder.
3. Double-click `install.bat` and wait -- it sets up pip inside the bundled Python, then installs PyTorch with CUDA plus the packages into it. Run it again if anything interrupts it; it resumes where it stopped.
4. Double-click `run.bat`. A page opens in your browser. Paste your token, type a prompt, hit Generate.

install.bat does NOT need re-running for code-only updates: if only app.py changed, just replace the file and run run.bat again.

## First test

1. Leave everything at the defaults — CLIP-only (no T5), offload on, 1024x1024, 40 steps, guidance 4.5 — paste your token, and hit Generate.
2. The first run downloads the model (about 6.5 GB on this load), so it takes a while -- that is the one-time download, not a stuck render.
3. If the status says the load could not happen and mentions the gate, the token or the license acceptance is the problem, in that order: check the token is a READ token from the same Hugging Face account that accepted the license.
4. When the default test works, try T5 full *if your machine has the RAM for it* (see Requirements) — expect a much bigger download (~16 GB total) and a slower load, in exchange for better long-prompt following.
5. If a run fails with an out-of-memory error on a 16 GB RAM machine, you are almost certainly on T5 full or offload off — go back to CLIP-only with offload ON; that combination is the default for this class of machine and was PC-proven on the author's own 16 GB machine on Oct 27, 2026 — see Code verification.

## Where pictures go

`images_out/` next to `run.bat`, one subfolder per batch name, reused across runs -- repeat runs with the same batch name add pictures to the same folder, numbered continuously (`000001.png`, `000002.png`, ...), so nothing is ever overwritten. `settings.txt` in the folder grows by one block per run (date, time, frame range, prompt, negative prompt, expanded prompts, size, steps, guidance, max sequence length, T5 mode, offload, seeds). The token is never recorded. Starred batches land in `images_out/favorites/`.

## How it works

Diffusers 0.42.0, StableDiffusion3Pipeline:

1. `StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16, token=<token box>)`. With "No T5" selected, `text_encoder_3=None` and `tokenizer_3=None` are also passed — verified in the diffusers 0.42.0 wheel: its `encode_prompt` checks `text_encoder_3 is None` and returns zero embeddings of the right shape, and the CLIP encoders carry the prompt. With "T5 full", nothing is overridden and all three encoders load in bfloat16. (Loading T5 separately in float16 — a pattern in some guides — was checked and deliberately not used: float16 is the same 2 bytes per parameter as bfloat16, so it saves no RAM and only adds code.)
2. With offload ON (the default), `enable_model_cpu_offload()` keeps the components in RAM and moves one at a time to the GPU. With offload OFF, everything goes to GPU with `.to("cuda")` -- only for cards with much more than 12 GB of VRAM.
3. The pipeline call is `pipe(prompt, negative_prompt, height, width, num_inference_steps=<steps slider>, guidance_scale=<guidance slider>, max_sequence_length=256, generator)` with one CPU generator per image (seeded from the seed box), following the model card's own snippet (which uses 40 steps and guidance 4.5). The CPU generator is deliberate: it works identically with offload on or off.
4. `max_sequence_length=256` is the pipeline's own default, and the plain model-card snippet's implicit value. A community quant example uses 512; that was not copied, because the pipeline builds a (batch, sequence, joint-attention-dim) embedding at that length — even with T5 off — so 512 costs memory on every image for a benefit only very long T5 prompts would see.
5. LoRAs load through the pipeline's own adapter API: each picked file loads once per Generate click with `load_lora_weights(<loras folder>, weight_name=<file>, adapter_name="lora<i>")`, then one `set_adapters(names, adapter_weights=weights)` applies both slots' weights; changing the picks unloads first with `unload_lora_weights()`. Verified against the diffusers 0.42.0 wheel: `SD3LoraLoaderMixin._lora_loadable_modules` is exactly `["transformer", "text_encoder", "text_encoder_2"]`, a component with no weights in the file is skipped with a warning (never an error), and the API is PEFT-backed — that is why `peft` is pinned in requirements.txt.

Notes and known limits of this build:

- The gate is real: no token, or a token from an account that has not accepted the license, means no download. The app surfaces that as a plain status message with the two things to check.
- On 16 GB of RAM, T5 full is the heaviest mode — but it was PC-proven on the author's own 16 GB machine on Oct 27, 2026 (CPU offload on): a 100-step portrait rendered in about 50 seconds with no out-of-memory error, and an unattended queue ran to completion. CLIP-only stays the default because it is lighter and faster to load, not because T5 full fails.
- Nothing about the model is bundled here; weights download from Stability's repo under their license (below).

## Code verification (no GPU in the build environment)

- `app.py` compiles clean (`py_compile`).
- The pipeline class and the No-T5 pattern were verified against the diffusers 0.42.0 wheel itself (not just docs): `StableDiffusion3Pipeline` exists, `text_encoder_3=None` is an explicitly handled path, and `max_sequence_length` defaults to 256 with a hard cap of 512.
- The requirements set (minus accelerate — see the note at the top of requirements.txt for why) was verified end-to-end with pip's resolver for Windows / Python 3.12, resolving to exactly the pinned versions; accelerate 1.15.0 was verified separately from its wheel metadata.
- The loader and generation paths are stub-tested with fake pipeline modules: CLIP-only vs T5-full arguments to from_pretrained, token passing (and its absence when the box is empty), offload on/off, load-failure message, steps/guidance/max-sequence pass-through, seed resolution, dynamic expansion, batch-folder/settings logic (continuous 6-digit frames, appended settings blocks, T5 mode recorded), queue-over-prompt, history, star, and the early-exit paths. The stub test is not shipped in the zip.
- The LoRA slots (added Oct 28, 2026) are stub-tested the same way, with the fake pipeline's adapter methods mirroring the real 0.42.0 validation read from the wheel: the all-keys-must-contain-'lora' checkpoint check, the duplicate-adapter-name refusal, and set_adapters' missing-adapter refusal, each with the wheel's exact error text as negative controls. Covered: picker population from `loras/`, adapter mapping in both encoder modes, two adapters with separate weights, transformer-only files, switching picks between runs with no adapter leak, unload on None, and the LoRA line in settings.txt. peft 0.21.2 was verified from its wheel metadata against this requirements set (see the note in requirements.txt for why the Linux-host dry run cannot cover it). The app's exact LoRA call shapes were ALSO run against the real pinned stack (diffusers 0.42.0 + transformers 5.20.2 + peft 0.21.2 on CPU torch, a pipeline assembled from tiny random-init components with `text_encoder_3=None`, i.e. the CLIP-only shape): one adapter and two adapters load and set with their weights, the duplicate-name and missing-adapter refusals match the stub's error texts word for word, a non-LoRA checkpoint is refused, and unload empties the registry.
- PC status (author's machine, RTX 4070 12 GB / 16 GB RAM): the base app was proven Oct 27, 2026 — real install, gated download, first pictures in both CLIP-only and T5-full modes, and an unattended queue that ran to completion. The LoRA slots were proven Oct 29, 2026 with a real Diffusers-format SD 3.5 LoRA (see **LoRAs** above). Kohya-format LoRAs remain unproven-by-design: they do not load until a converter ships.

## Models and permissions

The model belongs to Stability AI — see its [Hugging Face page](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) for terms. SD 3.5 Medium is under the **Stability AI Community License**: free for research, non-commercial, and commercial use for organizations under $1M in annual revenue; an enterprise license is required above that. Outputs are yours subject to that license's terms. **The model is NOT MIT or Apache, and this app does not change that.** Nothing is bundled here and nothing is sold. This UI (the Gradio app, the wrapper, the installer) is by Alan Carmack.

## License

MIT for this UI code -- free to use, share, and change. The model has its own license, above.
