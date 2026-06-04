import copy
import gc
import textwrap

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import comfy.model_management
import comfy.samplers
import comfy.utils
from nodes import common_ksampler


MAX_SEED = 0xFFFFFFFFFFFFFFFF
DEFAULT_MAX_COMBINATIONS = 100
NONE_OPTION = "None"
SAMPLER_SLOT_COUNT = 9
SCHEDULER_SLOT_COUNT = 9
UNKNOWN_LABEL = "unknown"


def collect_selected_slots(values):
    seen = set()
    selected = []

    for value in values:
        name = str(value).strip() if value is not None else NONE_OPTION
        if not name or name == NONE_OPTION or name.lower() == "none" or name in seen:
            continue
        seen.add(name)
        selected.append(name)

    return selected


def get_available_samplers():
    return list(comfy.samplers.KSampler.SAMPLERS)


def get_available_schedulers():
    return list(comfy.samplers.KSampler.SCHEDULERS)


def validate_selected_names(selected, available, label):
    if not selected:
        raise ValueError(f"No {label} selected. Choose at least one {label} slot that is not None.")

    available_set = set(available)
    unknown = [name for name in selected if name not in available_set]
    if unknown:
        available_text = ", ".join(available)
        unknown_text = ", ".join(unknown)
        raise ValueError(
            f"Unknown {label}: {unknown_text}. Available {label}: {available_text}"
        )


def get_prompt_node(prompt, node_id):
    if not isinstance(prompt, dict) or node_id is None:
        return None

    return prompt.get(str(node_id)) or prompt.get(node_id)


def get_input_link(prompt, node_id, input_name):
    node = get_prompt_node(prompt, node_id)
    if not isinstance(node, dict):
        return None

    value = node.get("inputs", {}).get(input_name)
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return None


def get_first_widget_value(node, names):
    if not isinstance(node, dict):
        return None

    inputs = node.get("inputs", {})
    for name in names:
        value = inputs.get(name)
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value).strip()
    return None


def describe_source_node(node, preferred_inputs):
    value = get_first_widget_value(node, preferred_inputs)
    if value:
        return value

    if isinstance(node, dict):
        class_type = node.get("class_type")
        if class_type:
            return str(class_type)

    return UNKNOWN_LABEL


def infer_clip_label(prompt, unique_id, model_label):
    for conditioning_input in ("positive", "negative"):
        conditioning_node_id = get_input_link(prompt, unique_id, conditioning_input)
        clip_source_id = get_input_link(prompt, conditioning_node_id, "clip")
        clip_source = get_prompt_node(prompt, clip_source_id)
        clip_label = describe_source_node(
            clip_source,
            ("clip_name", "clip_name1", "clip_name2", "ckpt_name", "checkpoint", "model_name"),
        )
        if clip_label != UNKNOWN_LABEL:
            if model_label != UNKNOWN_LABEL and clip_label == model_label:
                return "same as model checkpoint"
            return clip_label

    return UNKNOWN_LABEL


def infer_run_metadata(prompt, unique_id, steps, cfg, denoise):
    model_source_id = get_input_link(prompt, unique_id, "model")
    vae_source_id = get_input_link(prompt, unique_id, "vae")
    model_source = get_prompt_node(prompt, model_source_id)
    vae_source = get_prompt_node(prompt, vae_source_id)

    model_label = describe_source_node(
        model_source,
        ("ckpt_name", "checkpoint", "unet_name", "model_name", "model"),
    )
    vae_label = describe_source_node(
        vae_source,
        ("vae_name", "ckpt_name", "checkpoint", "model_name", "vae"),
    )
    if model_label != UNKNOWN_LABEL and vae_label == model_label:
        vae_label = "same as model checkpoint"

    clip_label = infer_clip_label(prompt, unique_id, model_label)

    return {
        "model": model_label,
        "vae": vae_label,
        "clip": clip_label,
        "steps": str(steps),
        "cfg": f"{float(cfg):g}",
        "denoise": f"{float(denoise):g}",
    }


def clone_latent(latent):
    cloned = {}
    for key, value in latent.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            try:
                cloned[key] = copy.deepcopy(value)
            except Exception:
                cloned[key] = value
    return cloned


def run_single_sample(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise,
):
    return common_ksampler(
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        clone_latent(latent_image),
        denoise=denoise,
    )[0]


def decode_latent_to_image(vae, sampled_latent):
    latent = sampled_latent["samples"]
    if getattr(latent, "is_nested", False):
        latent = latent.unbind()[0]

    images = vae.decode(latent)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def tensor_to_pil(image_tensor):
    image = image_tensor
    if len(image.shape) == 4:
        image = image[0]

    array = (
        image.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .numpy()
    )
    array = (array * 255.0).round().astype(np.uint8)

    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")

    if array.shape[-1] == 1:
        return Image.fromarray(array[..., 0], mode="L").convert("RGB")

    return Image.fromarray(array[..., :3], mode="RGB")


def pil_to_tensor(image):
    image = image.convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def resolve_background(background):
    colors = {
        "white": (246, 246, 242),
        "gray": (42, 44, 48),
        "black": (12, 13, 15),
    }
    return colors.get(background, colors["white"])


def text_color_for_background(background):
    if background in ("black", "gray"):
        return (238, 238, 232)
    return (28, 30, 34)


def line_color_for_background(background):
    if background in ("black", "gray"):
        return (92, 96, 104)
    return (184, 184, 176)


def label_background_for_background(background):
    colors = {
        "white": (235, 235, 229),
        "gray": (34, 36, 40),
        "black": (24, 25, 29),
    }
    return colors.get(background, colors["white"])


def load_font(size):
    for font_name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(draw, text, font, max_width):
    if not text:
        return [""]

    wrapped = []
    for raw_line in str(text).splitlines():
        words = raw_line.split()
        if not words:
            wrapped.append("")
            continue

        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if text_size(draw, candidate, font)[0] <= max_width:
                line = candidate
            else:
                wrapped.append(line)
                line = word

        while text_size(draw, line, font)[0] > max_width and len(line) > 1:
            approx_chars = max(1, int(len(line) * max_width / max(text_size(draw, line, font)[0], 1)))
            wrapped.extend(textwrap.wrap(line, width=approx_chars, break_long_words=True))
            line = ""

        if line:
            wrapped.append(line)

    return wrapped or [str(text)]


def draw_centered_text(draw, box, text, font, fill):
    x, y, width, height = box
    max_width = max(1, width - 12)
    lines = wrap_text(draw, text, font, max_width)
    line_heights = [text_size(draw, line, font)[1] for line in lines]
    total_height = sum(line_heights) + max(0, len(lines) - 1) * 4
    cursor_y = y + max(0, (height - total_height) // 2)

    for line, line_height in zip(lines, line_heights):
        line_width, _ = text_size(draw, line, font)
        draw.text((x + (width - line_width) // 2, cursor_y), line, font=font, fill=fill)
        cursor_y += line_height + 4


def draw_left_label(draw, box, text, font, fill):
    x, y, width, height = box
    max_width = max(1, width - 18)
    lines = wrap_text(draw, text, font, max_width)
    line_heights = [text_size(draw, line, font)[1] for line in lines]
    total_height = sum(line_heights) + max(0, len(lines) - 1) * 4
    cursor_y = y + max(0, (height - total_height) // 2)

    for line, line_height in zip(lines, line_heights):
        draw.text((x + 10, cursor_y), line, font=font, fill=fill)
        cursor_y += line_height + 4


def build_run_header_text(metadata):
    return (
        "KSampler Matrix Lab Benchmark\n"
        f"Model: {metadata['model']} | VAE: {metadata['vae']} | CLIP: {metadata['clip']}\n"
        f"Steps: {metadata['steps']} | CFG: {metadata['cfg']} | Denoise: {metadata['denoise']}"
    )


def wrapped_text_height(draw, text, font, max_width):
    lines = wrap_text(draw, text, font, max_width)
    line_heights = [text_size(draw, line, font)[1] for line in lines]
    return sum(line_heights) + max(0, len(lines) - 1) * 4


def draw_run_header(draw, box, metadata, font, fill):
    text = build_run_header_text(metadata)
    draw_centered_text(draw, box, text, font, fill)


def placeholder_size_from_latent(latent_image):
    samples = latent_image.get("samples")
    if torch.is_tensor(samples) and samples.ndim >= 4:
        return max(1, int(samples.shape[-1]) * 8), max(1, int(samples.shape[-2]) * 8)
    return 512, 512


def compose_error_cell(size, sampler, scheduler, error_message, background, font):
    cell = Image.new("RGB", size, resolve_background(background))
    draw = ImageDraw.Draw(cell)
    fill = (210, 76, 68) if background == "white" else (255, 146, 132)
    short_error = str(error_message).replace("\n", " ")[:140]
    text = f"ERROR\n{sampler} / {scheduler}\n{short_error}"
    draw_centered_text(draw, (8, 8, size[0] - 16, size[1] - 16), text, font, fill)
    return cell


def compose_grid(
    rows,
    columns,
    cells,
    latent_image,
    cell_scale,
    font_size,
    padding,
    header_height,
    left_header_width,
    background,
    show_grid_lines,
    show_cell_labels,
    show_run_header,
    run_metadata,
):
    font = load_font(font_size)
    label_font = load_font(max(10, int(font_size * 0.92)))
    cell_label_font = load_font(max(9, int(font_size * 0.78)))

    first_image = next((cell["image"] for cell in cells.values() if cell.get("image") is not None), None)
    base_width, base_height = first_image.size if first_image is not None else placeholder_size_from_latent(latent_image)

    cell_width = max(1, int(round(base_width * cell_scale)))
    cell_height = max(1, int(round(base_height * cell_scale)))
    cell_label_height = max(42, int(font_size * 2.7)) if show_cell_labels else 0
    cell_block_height = cell_height + cell_label_height

    background_rgb = resolve_background(background)
    label_background_rgb = label_background_for_background(background)
    text_rgb = text_color_for_background(background)
    line_rgb = line_color_for_background(background)

    total_width = left_header_width + len(columns) * cell_width + (len(columns) + 1) * padding
    if show_run_header:
        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        header_text_height = wrapped_text_height(
            measure,
            build_run_header_text(run_metadata),
            label_font,
            max(1, total_width - 2 * padding),
        )
        run_header_height = max(70, header_text_height + 2 * padding)
    else:
        run_header_height = 0

    total_height = run_header_height + header_height + len(rows) * cell_block_height + (len(rows) + 1) * padding

    grid = Image.new("RGB", (total_width, total_height), background_rgb)
    draw = ImageDraw.Draw(grid)

    grid_y = run_header_height
    if show_run_header:
        draw.rectangle((0, 0, total_width - 1, run_header_height - 1), fill=label_background_rgb)
        draw_run_header(
            draw,
            (padding, padding, total_width - 2 * padding, max(1, run_header_height - 2 * padding)),
            run_metadata,
            label_font,
            text_rgb,
        )
        if show_grid_lines:
            draw.line((0, run_header_height - 1, total_width, run_header_height - 1), fill=line_rgb, width=1)

    draw_centered_text(
        draw,
        (0, grid_y, left_header_width, header_height),
        "KSampler\nMatrix Lab",
        label_font,
        text_rgb,
    )

    for column_index, scheduler in enumerate(columns):
        x = left_header_width + padding + column_index * (cell_width + padding)
        draw_centered_text(draw, (x, grid_y, cell_width, header_height), scheduler, font, text_rgb)

    for row_index, sampler in enumerate(rows):
        y = grid_y + header_height + padding + row_index * (cell_block_height + padding)
        draw_left_label(draw, (0, y, left_header_width, cell_block_height), sampler, font, text_rgb)

    for row_index, sampler in enumerate(rows):
        for column_index, scheduler in enumerate(columns):
            x = left_header_width + padding + column_index * (cell_width + padding)
            y = grid_y + header_height + padding + row_index * (cell_block_height + padding)
            cell = cells[(sampler, scheduler)]

            if cell.get("image") is not None:
                image = cell["image"]
                if image.size != (cell_width, cell_height):
                    image = image.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
            else:
                image = compose_error_cell(
                    (cell_width, cell_height),
                    sampler,
                    scheduler,
                    cell.get("error", "Unknown error"),
                    background,
                    label_font,
                )

            if show_cell_labels:
                draw.rectangle(
                    (x, y, x + cell_width - 1, y + cell_label_height - 1),
                    fill=label_background_rgb,
                )
                draw_centered_text(
                    draw,
                    (x + 4, y + 3, cell_width - 8, cell_label_height - 6),
                    f"Sampler: {sampler}\nScheduler: {scheduler}",
                    cell_label_font,
                    text_rgb,
                )
                if show_grid_lines:
                    draw.line(
                        (x, y + cell_label_height - 1, x + cell_width - 1, y + cell_label_height - 1),
                        fill=line_rgb,
                        width=1,
                    )

            grid.paste(image, (x, y + cell_label_height))

            if show_grid_lines:
                draw.rectangle(
                    (x, y, x + cell_width - 1, y + cell_block_height - 1),
                    outline=line_rgb,
                    width=1,
                )

    if show_grid_lines:
        draw.line((left_header_width, grid_y, left_header_width, total_height), fill=line_rgb, width=1)
        draw.line((0, grid_y + header_height, total_width, grid_y + header_height), fill=line_rgb, width=1)

    return grid


class KSamplerMatrixLab:
    @classmethod
    def INPUT_TYPES(cls):
        samplers = get_available_samplers()
        schedulers = get_available_schedulers()
        sampler_choices = [NONE_OPTION] + samplers
        scheduler_choices = [NONE_OPTION] + schedulers
        default_sampler = samplers[0] if samplers else NONE_OPTION
        default_scheduler = schedulers[0] if schedulers else NONE_OPTION

        required = {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "vae": ("VAE",),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_SEED,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }

        for index in range(1, SAMPLER_SLOT_COUNT + 1):
            required[f"sampler_{index:02d}"] = (
                sampler_choices,
                {
                    "default": default_sampler if index == 1 else NONE_OPTION,
                    "tooltip": "Select a sampler for a matrix row, or None to ignore this slot.",
                },
            )

        for index in range(1, SCHEDULER_SLOT_COUNT + 1):
            required[f"scheduler_{index:02d}"] = (
                scheduler_choices,
                {
                    "default": default_scheduler if index == 1 else NONE_OPTION,
                    "tooltip": "Select a scheduler for a matrix column, or None to ignore this slot.",
                },
            )

        required.update(
            {
                "seed_mode": (["same_seed_for_all", "increment_per_cell"], {"default": "same_seed_for_all"}),
                "cell_scale": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 4.0, "step": 0.05}),
                "font_size": ("INT", {"default": 22, "min": 8, "max": 96}),
                "padding": ("INT", {"default": 10, "min": 0, "max": 96}),
                "header_height": ("INT", {"default": 72, "min": 24, "max": 256}),
                "left_header_width": ("INT", {"default": 180, "min": 48, "max": 512}),
                "background": (["white", "gray", "black"], {"default": "white"}),
                "show_grid_lines": ("BOOLEAN", {"default": True}),
                "show_cell_labels": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Repeat sampler and scheduler labels inside every grid cell for easier zoomed inspection.",
                    },
                ),
                "show_run_header": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Add a top header with model, VAE, CLIP, steps, CFG, and denoise metadata.",
                    },
                ),
                "continue_on_error": ("BOOLEAN", {"default": True}),
                "max_combinations": (
                    "INT",
                    {
                        "default": DEFAULT_MAX_COMBINATIONS,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "Safety limit for sampler x scheduler combinations.",
                    },
                ),
            }
        )

        return {
            "required": required,
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid_image",)
    FUNCTION = "generate_matrix"
    CATEGORY = "ComfyUI-KSampler-Matrix-Lab"
    DESCRIPTION = "Runs selected sampler and scheduler combinations sequentially, then returns one labeled benchmark grid image."

    def generate_matrix(
        self,
        model,
        positive,
        negative,
        latent_image,
        vae,
        seed,
        steps,
        cfg,
        denoise,
        seed_mode,
        cell_scale,
        font_size,
        padding,
        header_height,
        left_header_width,
        background,
        show_grid_lines,
        show_cell_labels,
        show_run_header,
        continue_on_error,
        max_combinations,
        prompt=None,
        unique_id=None,
        **selection_slots,
    ):
        selected_samplers = collect_selected_slots(
            selection_slots.get(f"sampler_{index:02d}")
            for index in range(1, SAMPLER_SLOT_COUNT + 1)
        )
        selected_schedulers = collect_selected_slots(
            selection_slots.get(f"scheduler_{index:02d}")
            for index in range(1, SCHEDULER_SLOT_COUNT + 1)
        )

        validate_selected_names(selected_samplers, get_available_samplers(), "sampler")
        validate_selected_names(selected_schedulers, get_available_schedulers(), "scheduler")

        total = len(selected_samplers) * len(selected_schedulers)
        if total > max_combinations:
            raise ValueError(
                f"KSampler Matrix Lab refused to run {total} combinations. "
                f"Current max_combinations is {max_combinations}."
            )

        progress = comfy.utils.ProgressBar(total)
        cells = {}
        cell_index = 0

        for sampler_name in selected_samplers:
            for scheduler in selected_schedulers:
                current_seed = seed if seed_mode == "same_seed_for_all" else (seed + cell_index) & MAX_SEED
                try:
                    sampled_latent = run_single_sample(
                        model,
                        current_seed,
                        steps,
                        cfg,
                        sampler_name,
                        scheduler,
                        positive,
                        negative,
                        latent_image,
                        denoise,
                    )
                    decoded_images = decode_latent_to_image(vae, sampled_latent)
                    cells[(sampler_name, scheduler)] = {"image": tensor_to_pil(decoded_images)}
                except Exception as exc:
                    if not continue_on_error:
                        raise RuntimeError(
                            f"KSampler Matrix Lab failed for sampler '{sampler_name}' "
                            f"and scheduler '{scheduler}': {exc}"
                        ) from exc
                    cells[(sampler_name, scheduler)] = {"image": None, "error": str(exc)}
                finally:
                    cell_index += 1
                    progress.update(1)
                    gc.collect()
                    comfy.model_management.soft_empty_cache()

        grid = compose_grid(
            selected_samplers,
            selected_schedulers,
            cells,
            latent_image,
            cell_scale,
            font_size,
            padding,
            header_height,
            left_header_width,
            background,
            show_grid_lines,
            show_cell_labels,
            show_run_header,
            infer_run_metadata(prompt, unique_id, steps, cfg, denoise),
        )

        return (pil_to_tensor(grid),)


NODE_CLASS_MAPPINGS = {
    "KSamplerMatrixLab": KSamplerMatrixLab,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerMatrixLab": "KSampler Matrix Lab",
}
