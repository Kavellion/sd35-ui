"""SD 3.5 Medium Gradio UI -- text-to-image only.

Same shape as Kandinsky UI, Sana UI, and Z-Image UI: no Jupyter,
no Colab. Run it, a page opens in your browser, type a prompt, hit
Generate, a picture appears underneath. Stable Diffusion 3.5
Medium is Stability AI's 2.5B MMDiT image model (three text
encoders: CLIP-L, CLIP-G, and T5-XXL), under the Stability AI
Community License (free under $1M annual revenue -- see README).
The weights ARE gated: you must accept that license on the model
page AND paste a Hugging Face read token into the token box
before the first Generate can succeed.

How to run (on a Windows PC with an NVIDIA GPU):

    install.bat      (once -- builds a Python 3.12 venv, installs
                      torch from cu130 + the packages)
    run.bat          (starts the app)

Python 3.12 is required because Diffusers 0.42 dropped Python 3.10.
The installer builds its own .venv from 3.12 and never touches any
other Python on the machine -- a Python 3.10 install used by other
apps is left completely alone.

The app file lives in the sd35-ui/ subfolder; run.bat runs it from
the folder above, so pictures land in images_out/ next to
run.bat -- one subfolder per batch name, reused across runs,
frames numbered continuously, and settings.txt appended once per
run, exactly like the other apps.

First Generate takes a while: the model downloads from Hugging
Face (about 16 GB with T5, about 6.5 GB without -- the CLIP-only
load skips the T5 files) and loads once. Repeat generations reuse
what is already loaded; changing the T5 mode, the offload toggle,
or the token reloads.

How it works (Diffusers 0.42.0, StableDiffusion3Pipeline -- call
shapes match the SD 3.5 Medium model card's own snippet):
  1. The pipeline loads from stabilityai/stable-diffusion-3.5-
     medium in bfloat16 with the token from the token box.
  2. Steps (default 40) and guidance (default 4.5) are sliders,
     because unlike a distilled model this one uses both -- the
     defaults are the model card's own snippet values.

Fitting on a 12 GB card (RTX 4070) with 16 GB of system RAM --
the binding constraint is RAM, not VRAM, as the Klein build
proved:
  - CPU offload defaults ON (enable_model_cpu_offload()): one
    component at a time goes to the GPU; the rest live in RAM.
  - T5-XXL is the problem child: about 9.4 GB of weights in
    bfloat16 (float16 is the same 2 bytes per parameter, so the
    documented separate-float16 T5 load saves nothing in RAM --
    it is not used here; plain and simple wins).
  - With T5, the full weight set is about 16 GB in RAM -- on a
    16 GB machine that pages hard or runs out of memory. So the
    DEFAULT T5 mode is "No T5 -- CLIP only": the loader passes
    text_encoder_3=None and tokenizer_3=None (a documented
    Diffusers SD3 pattern; the pipeline feeds zeros where the
    T5 embeddings would go and the two CLIP encoders carry the
    prompt). That set is about 6.5 GB and fits comfortably.
    Long, complex prompts follow less closely without T5 --
    that is the honest trade, and "T5 full" is one click away
    for machines with more RAM.
  - max_sequence_length is 256, the pipeline default and the
    plain model-card snippet's implicit value. 512 (seen in a
    community quant example) doubles the joint-attention
    sequence tensor -- the pipeline builds a (batch, seq, dim)
    embedding at that length even with T5 off -- for no benefit
    on ordinary prompts, so 256 it is.

Extras, in the same plain style as the other apps: size presets,
several pictures per prompt, seed shown in the status line, a
Stop button, a Random prompt button (word banks in code), a
session-only prompt history dropdown, Star batch (copies a batch
into images_out/favorites/), a prompt queue (one prompt per
line), single-level dynamic prompts {a|b|c}, and an Unload
button to free VRAM/RAM. Memory is freed between every image
in a run (the Z-Image batch lesson).

No img2img, no inpainting, no ControlNet, no blending, no
upscaling, no animation -- v1 is text-to-image only, kept simple
on purpose.
"""

import gc
import os
import random
import re
import shutil
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)  # folder holding run.bat
IMAGES_DIR = os.path.join(ROOT_DIR, "images_out")
FAVORITES_DIR = os.path.join(IMAGES_DIR, "favorites")

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"

# Model-card snippet values.
DEFAULT_STEPS = 40
DEFAULT_GUIDANCE = 4.5
# Pipeline default; see the docstring for why not 512.
MAX_SEQUENCE_LENGTH = 256

DEFAULT_PROMPT = (
    "A red fox sitting in a sunflower field at golden hour, "
    "detailed fur, photorealistic"
)

_pipe = None
_pipe_key = None  # (t5_mode, offload, has_token) of the loaded pipe
_pipe_note = ""


def _safe_batch_name(name):
    """Strip characters Windows forbids in folder names. Falls back
    to a timestamped name when the box is empty."""
    name = (name or "").strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(" .")
    if not name:
        name = datetime.now().strftime("sd35-%Y%m%d-%H%M%S")
    return name


def _batch_folder(base):
    """The one folder for a batch name: images_out/<base>, created
    if missing, reused if it already exists. Repeat runs never get
    a _2/_3 folder -- frames just keep numbering upward."""
    return os.path.join(IMAGES_DIR, base)


def _unique_folder(base, root=IMAGES_DIR):
    """<root>/<base>, or <base>_2, <base>_3... if it already exists.
    Used only by Star batch; generations reuse one folder per batch
    name (see _batch_folder) instead of numbered folders."""
    folder = os.path.join(root, base)
    if not os.path.exists(folder):
        return folder
    i = 2
    while True:
        candidate = "{}_{}".format(folder, i)
        if not os.path.exists(candidate):
            return candidate
        i += 1


_FRAME_RE = re.compile(r"^(\d{6})\.png$")


def _next_frame_number(folder):
    """First unused frame number in a folder: one past the highest
    existing 000001-style frame, or 1 for a fresh folder."""
    highest = 0
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            match = _FRAME_RE.match(name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def _frame_range_text(start, count):
    """'frame 000006' for one image, 'frames 000006-000008' for
    several -- used in status lines and settings headers."""
    if count == 1:
        return "frame {:06d}".format(start)
    return "frames {:06d}-{:06d}".format(start, start + count - 1)


def _seeds_for(seed, count):
    """Resolve the seed box to one seed per image. -1 (or empty)
    draws a fresh random seed per image; a fixed seed S gives S,
    S+1, S+2... so a multi-image run with a fixed seed stays
    reproducible."""
    if seed is None:
        seed = -1
    seed = int(seed)
    if seed < 0:
        return [random.randint(0, 2**31 - 1) for _ in range(count)]
    return [seed + i for i in range(count)]


# Dynamic prompts: single level only, one random pick per group.
# "{a|b|c}" becomes a, b, or c. Groups do not nest; a group with no
# pipe is just its own text with the braces removed.
_TEMPLATE_RE = re.compile(r"\{([^{}]*)\}")


def _expand_templates(prompt):
    """Expand {a|b|c} groups, one random pick each. Plain prompts
    (no braces) come back unchanged."""
    return _TEMPLATE_RE.sub(
        lambda match: random.choice(match.group(1).split("|")), prompt
    )


def _append_settings(folder, header, entries):
    """Append one run's block to settings.txt: a header line (date,
    time, frame range) followed by the run's 'key: value' lines and
    a blank line. Creates the file on first use; earlier runs'
    blocks are never touched. A missing trailing newline is
    repaired so blocks never run together."""
    path = os.path.join(folder, "settings.txt")
    prefix = ""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                prefix = "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(prefix + header + "\n")
        handle.write("\n".join(entries) + "\n\n")


def _run_header(start, count):
    """The settings.txt header for one run: date, time, and the
    frame-number range it produced."""
    return "Run {}, {}".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        _frame_range_text(start, count))


def _seed_entry(seeds):
    """The settings.txt seed line: 'seed: X' for one image,
    'seeds: X, Y' for several."""
    if len(seeds) == 1:
        return "seed: {}".format(seeds[0])
    return "seeds: {}".format(", ".join(str(s) for s in seeds))


def _expanded_entries(expanded_list):
    """The settings.txt expanded-prompt line(s): one line for one
    image, a comma-joined list for several."""
    if len(expanded_list) == 1:
        return ["expanded_prompt: {}".format(expanded_list[0])]
    return ["expanded_prompts: {}".format(", ".join(expanded_list))]


def _round16(value):
    """Nearest multiple of 16, at least 16. Keeps the VAE's 8x
    downsampling exact even for odd-sized custom sizes."""
    return max(16, int(round(int(value) / 16.0)) * 16)


def _free_memory():
    """Drop cached GPU memory between images and after unloading
    a model. Best effort: never raises, so cleanup cannot break
    a run or the next load."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- freeing memory must not fail
        pass


def _unload_pipe():
    """Drop the loaded pipeline so the next Generate loads fresh
    (used when T5 mode/offload/token change, and by Unload)."""
    global _pipe, _pipe_key, _pipe_note
    _pipe = None
    _pipe_key = None
    _pipe_note = ""
    _free_memory()


def _t5_mode(choice):
    """Map the T5 radio label to the loader's mode string."""
    return "not5" if str(choice).startswith("No T5") else "t5"


def _load_pipe(t5_mode, offload, token):
    """Load (or reuse) the SD 3.5 Medium pipeline. Returns
    (pipe, note). t5_mode is "not5" (default: CLIP encoders only,
    text_encoder_3/tokenizer_3 passed as None) or "t5" (full
    three-encoder load). Offload True uses the standard
    enable_model_cpu_offload(); False loads everything onto the
    GPU (only for cards with much more VRAM). The token (from the
    UI's token box) is passed to from_pretrained; the model repo
    is gated, so without a token -- and without the license
    accepted on the model page -- the load fails, and the error
    says what to check.

    The token itself is never stored in the pipe key or printed;
    only whether one was supplied. That is enough: a load only
    caches after it succeeds, so a wrong token never sticks --
    fix the token and the next Generate loads fresh; adding or
    removing a token also reloads."""
    global _pipe, _pipe_key, _pipe_note
    token = (token or "").strip() or None
    key = (t5_mode, bool(offload), bool(token))
    if _pipe is not None and _pipe_key == key:
        return _pipe, _pipe_note
    _unload_pipe()

    import torch
    from diffusers import StableDiffusion3Pipeline

    kwargs = {"torch_dtype": torch.bfloat16}
    if token:
        kwargs["token"] = token
    if t5_mode == "not5":
        kwargs["text_encoder_3"] = None
        kwargs["tokenizer_3"] = None
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash
        raise RuntimeError(
            "Could not load SD 3.5 Medium ({}). This model is "
            "GATED: check (1) you accepted the Stability AI "
            "Community License on the model page at Hugging Face, "
            "(2) the token box holds a Hugging Face READ token "
            "from that same account, and (3) the internet "
            "connection -- then try Generate again.".format(exc))
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if t5_mode == "not5":
        note = ("Loaded CLIP-only (no T5): lighter load, simpler "
                "prompt understanding.")
    else:
        note = ("Loaded with T5-XXL: best prompt understanding, "
                "heaviest RAM use.")
    _pipe = pipe
    _pipe_key = key
    _pipe_note = note
    return pipe, note


def _history_add(history, prompt):
    """Add a prompt to the session history (most recent last, no
    duplicates, last 10 kept). Returns the new history list."""
    history = [item for item in (history or []) if item != prompt]
    history.append(prompt)
    return history[-10:]


def _history_update(history):
    """The history dropdown update: choices are the history, most
    recent first, value cleared so picking the same prompt twice
    still fires."""
    import gradio as gr
    return gr.update(choices=list(reversed(history or [])), value=None)


def _early(message, last_folder, history):
    """A single yield for an early exit (bad input, load failure):
    empty gallery, the message as status, history untouched."""
    yield [], message, _history_update(history), last_folder, history


def _preset_size(preset):
    """Size preset -> (width, height). Custom leaves the boxes
    alone (returns them unchanged)."""
    import gradio as gr
    sizes = {
        "Square 1024x1024": (1024, 1024),
        "Small square 768x768": (768, 768),
        "Portrait 768x1024": (768, 1024),
        "Landscape 1024x768": (1024, 768),
    }
    if preset in sizes:
        return sizes[preset]
    return gr.update(), gr.update()


def _noop():
    """Stop-button target: cancels the Generate event, does
    nothing itself."""
    return None


_SUBJECTS = [
    "a lighthouse", "a fox", "a mountain cabin", "a castle",
    "a harbor town", "a red panda", "a sailing ship",
    "a robot gardener", "a treehouse village", "a steam train",
]
_SETTINGS = [
    "a rocky cliff", "a misty forest", "desert dunes",
    "a snowy village", "a neon city street", "a quiet lake",
    "a sunflower field", "a space station", "a coral reef",
    "an old library",
]
_LIGHTS = [
    "golden hour light", "storm light", "moonlight",
    "neon dusk", "soft morning fog", "candlelight",
    "sunset clouds", "starlight",
]
_STYLES = [
    "photorealistic", "oil painting", "anime style",
    "watercolor", "cyberpunk", "storybook illustration",
]


def random_prompt():
    """One random prompt from the built-in word banks."""
    return "A {} in {}, {}, {}, detailed.".format(
        random.choice(_SUBJECTS),
        random.choice(_SETTINGS),
        random.choice(_LIGHTS),
        random.choice(_STYLES),
    )


def generate(prompt_text, negative_text, queue_text, batch_name,
             width, height, seed, count, steps, guidance,
             t5_choice, offload, token,
             last_folder, history):
    """Text-to-image generation. A generator function: it yields
    (gallery images, status, history dropdown update, last batch
    folder, history list) after every image, so the page updates
    live, and the Stop button can cancel it mid-run.

    Queue beats the single prompt: when the queue box has lines,
    each line is one job; otherwise the prompt box is the one job.
    Every job renders `count` images. Dynamics {a|b|c} are
    re-expanded for every single image (prompt and negative), so
    no two images in a run share an expansion by accident. All
    jobs in one click share one batch folder, frames numbered
    continuously."""
    import torch

    count = int(count) if count else 1
    count = max(1, min(10, count))
    try:
        steps = int(steps) if steps else DEFAULT_STEPS
    except (TypeError, ValueError):
        steps = DEFAULT_STEPS
    steps = max(1, min(100, steps))
    try:
        guidance = float(guidance) if guidance is not None \
            else DEFAULT_GUIDANCE
    except (TypeError, ValueError):
        guidance = DEFAULT_GUIDANCE
    guidance = max(0.0, min(20.0, guidance))
    history = history or []
    negative_template = (negative_text or "").strip()

    lines = [line.strip() for line in (queue_text or "").splitlines()
             if line.strip()]
    if lines:
        templates = lines
    elif prompt_text and prompt_text.strip():
        templates = [prompt_text.strip()]
    else:
        yield from _early("Type a prompt first, or fill the queue.",
                          last_folder, history)
        return

    try:
        width = _round16(max(16, min(2048, int(width))))
        height = _round16(max(16, min(2048, int(height))))
    except (TypeError, ValueError):
        yield from _early("Width and height must be numbers.",
                          last_folder, history)
        return

    t5_mode = _t5_mode(t5_choice)

    base = _safe_batch_name(batch_name)
    folder = _batch_folder(base)
    os.makedirs(folder, exist_ok=True)

    try:
        pipe, note = _load_pipe(t5_mode, bool(offload), token)
    except RuntimeError as exc:
        yield from _early(str(exc), last_folder, history)
        return

    gallery = []
    status = ""
    for template in templates:
        start = _next_frame_number(folder)
        seeds = _seeds_for(seed, count)
        expanded_list = []
        for i in range(count):
            expanded = _expand_templates(template)
            expanded_list.append(expanded)
            expanded_negative = _expand_templates(negative_template)
            try:
                image = pipe(
                    prompt=expanded,
                    negative_prompt=expanded_negative,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                    generator=torch.Generator("cpu").manual_seed(
                        seeds[i]),
                ).images[0]
            except Exception as exc:  # noqa: BLE001 -- plain status
                yield (gallery,
                       "Generation failed on {}: {}".format(
                           _frame_range_text(start + i, 1), exc),
                       _history_update(history), folder, history)
                return
            frame = start + i
            image.save(os.path.join(folder, "{:06d}.png".format(frame)))
            gallery.append(image)
            status = (
                "Saved {} in {} ({}x{}, steps {}, guidance {}). "
                "Seeds: {}. {}".format(
                    _frame_range_text(start, i + 1), folder,
                    width, height, steps, guidance,
                    ", ".join(str(s) for s in seeds[:i + 1]),
                    note))
            yield (gallery, status, _history_update(history),
                   folder, history)
            # Free cached memory between images. Without this, a
            # long queue creeps upward in RAM/VRAM until Windows
            # kills the process mid-batch (the Z-Image lesson:
            # died 3 images into a 10-image queue, per-step time
            # climbing as it went). _free_memory is best-effort,
            # never raises, so cleanup cannot break a run.
            _free_memory()
        _append_settings(folder, _run_header(start, count), [
            "model: {}".format(MODEL_ID),
            "t5_mode: {}".format(t5_mode),
            "offload: {}".format(bool(offload)),
            "size: {}x{}".format(width, height),
            "steps: {}".format(steps),
            "guidance: {}".format(guidance),
            "max_sequence_length: {}".format(MAX_SEQUENCE_LENGTH),
            "prompt: {}".format(template),
            "negative_prompt: {}".format(negative_template),
            _seed_entry(seeds),
        ] + _expanded_entries(expanded_list))
        history = _history_add(history, template)
        status = (
            "Done: {} in {} ({}x{}, steps {}, guidance {}). "
            "Seeds: {}. {}".format(
                _frame_range_text(start, count), folder,
                width, height, steps, guidance,
                ", ".join(str(s) for s in seeds), note))
        yield (gallery, status, _history_update(history),
               folder, history)


def star_batch(last_folder):
    """Copy the most recently generated batch folder into
    images_out/favorites/ under a unique name (never overwrite).
    If nothing has run yet this session, say so in the status."""
    if not last_folder or not os.path.isdir(last_folder):
        return "No batch has run yet in this session."
    os.makedirs(FAVORITES_DIR, exist_ok=True)
    base = os.path.basename(os.path.normpath(last_folder))
    dest = _unique_folder(base, root=FAVORITES_DIR)
    shutil.copytree(last_folder, dest)
    return "Starred to {}".format(dest)


def unload_models():
    """Free the generator (VRAM and RAM)."""
    _unload_pipe()
    return "Models unloaded -- VRAM and RAM freed."


def build_ui():
    """The Gradio page: one tab, text-to-image only, kept simple
    on purpose. Steps and guidance ARE sliders here -- this is
    not a distilled model, and the defaults (40 / 4.5) are the
    model card's own."""
    import gradio as gr

    with gr.Blocks(title="SD 3.5 Medium UI") as demo:
        gr.Markdown(
            "# SD 3.5 Medium UI\n"
            "Type a prompt, hit Generate, pictures appear below. "
            "This model is GATED: accept the Stability AI "
            "Community License on its Hugging Face page, then "
            "paste a Hugging Face read token into the token box "
            "-- both, before the first Generate.\n"
            "Default load is CLIP-only (no T5) with CPU offload "
            "on: that is the load that fits a 16 GB RAM machine. "
            "T5 full understands longer prompts better and is "
            "one click away.")
        with gr.Row():
            prompt_box = gr.Textbox(
                label="Prompt", value=DEFAULT_PROMPT, lines=3)
            batch_box = gr.Textbox(
                label="Batch name (one folder per name)",
                value="sd35-pictures")
        negative_box = gr.Textbox(
            label="Negative prompt", value="", lines=2)
        with gr.Row():
            preset = gr.Dropdown(
                label="Size preset",
                choices=["Square 1024x1024", "Small square 768x768",
                         "Portrait 768x1024", "Landscape 1024x768",
                         "Custom"],
                value="Square 1024x1024")
            width_box = gr.Number(label="Width", value=1024,
                                  precision=0)
            height_box = gr.Number(label="Height", value=1024,
                                   precision=0)
        with gr.Row():
            steps_box = gr.Slider(label="Steps", minimum=1,
                                  maximum=100, value=DEFAULT_STEPS,
                                  step=1)
            guidance_box = gr.Slider(label="Guidance", minimum=0,
                                     maximum=20,
                                     value=DEFAULT_GUIDANCE,
                                     step=0.1)
        with gr.Row():
            seed_box = gr.Number(label="Seed (-1 = random)",
                                 value=-1, precision=0)
            count_box = gr.Slider(label="Images per prompt",
                                  minimum=1, maximum=10, value=1,
                                  step=1)
        with gr.Row():
            t5_box = gr.Radio(
                label="Text encoders",
                choices=[
                    "No T5 -- CLIP only (lighter, recommended "
                    "for 16 GB RAM)",
                    "T5 full (best prompt understanding, "
                    "heaviest RAM)"],
                value="No T5 -- CLIP only (lighter, recommended "
                      "for 16 GB RAM)")
            offload_box = gr.Checkbox(
                label="CPU offload (keep ON for a 12 GB card)",
                value=True)
        token_box = gr.Textbox(
            label="Hugging Face token (read token -- required, "
                  "the model is gated)",
            value="", type="password")
        queue_box = gr.Textbox(
            label="Prompt queue (one prompt per line -- when this "
                  "has lines, Generate runs the queue instead of "
                  "the prompt box; all lines share one batch folder)",
            lines=4)
        with gr.Row():
            random_btn = gr.Button("Random prompt")
            history_dd = gr.Dropdown(
                label="History (last 10 -- pick one to reload it)",
                choices=[])
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary")
            stop_btn = gr.Button("Stop")
            star_btn = gr.Button("Star batch")
            unload_btn = gr.Button("Unload models")
        status_box = gr.Textbox(label="Status", lines=4)
        gallery = gr.Gallery(label="Pictures", columns=3)

        last_folder_state = gr.State("")
        history_state = gr.State([])

        preset.change(fn=_preset_size, inputs=preset,
                      outputs=[width_box, height_box])
        random_btn.click(fn=random_prompt, outputs=prompt_box)
        history_dd.change(fn=lambda choice: choice,
                          inputs=history_dd, outputs=prompt_box)
        generate_event = generate_btn.click(
            fn=generate,
            inputs=[prompt_box, negative_box, queue_box, batch_box,
                    width_box, height_box, seed_box, count_box,
                    steps_box, guidance_box, t5_box, offload_box,
                    token_box, last_folder_state, history_state],
            outputs=[gallery, status_box, history_dd,
                     last_folder_state, history_state])
        stop_btn.click(fn=_noop, cancels=[generate_event])
        star_btn.click(fn=star_batch, inputs=last_folder_state,
                       outputs=status_box)
        unload_btn.click(fn=unload_models, outputs=status_box)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", inbrowser=True)
