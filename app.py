"""Local, full-image RGB-to-HSI demo for the official MST++ model zoo."""

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".cache/gradio"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import torch

from demo.inference import MODELS, Reconstructor
from demo.spectral import (
    DEFAULT_WAVELENGTH_TEXT,
    WAVELENGTHS,
    display_band,
    export_channels,
    export_cube,
    gallery,
    parse_channels,
    parse_wavelengths,
    spectrum_figure,
)

LOG = logging.getLogger(__name__)
READY = "Ready. Upload an RGB image to begin."
CSS = """
.gradio-container {max-width:1240px !important; width:100% !important; min-width:0 !important; margin:auto;}
footer {display:none !important;}
#hero h1 {font-size:34px; letter-spacing:-.7px; margin-bottom:4px;}
#input-column, #result-column {gap:12px;}
#channel-strip-label p, #example-strip-label p {margin:0; font-size:13px;}
#example-strip-label, #channel-strip-label {min-height:20px;}
#rgb-input .image-frame, #channel-preview .image-frame {width:100% !important; height:100% !important;}
#rgb-input .image-frame img, #channel-preview .image-frame img {width:100% !important; height:100% !important; object-fit:contain !important;}
:is(#spectral-gallery, #rgb-examples) {height:116px; width:100% !important; min-width:0 !important; overflow:hidden;}
:is(#spectral-gallery, #rgb-examples) .grid-wrap {width:100%; min-width:0; max-width:100%; overflow-x:auto !important; overflow-y:hidden !important; height:114px !important; padding:8px !important; box-sizing:border-box; scrollbar-width:thin;}
:is(#spectral-gallery, #rgb-examples) .grid-container {display:grid !important; grid-auto-flow:column !important; grid-template-columns:none !important; grid-auto-columns:86px !important; grid-template-rows:92px !important; grid-auto-rows:92px !important; gap:8px !important; width:max-content !important; min-width:100%;}
:is(#spectral-gallery, #rgb-examples) .gallery-item {width:86px !important; height:92px !important; border-radius:7px !important; box-shadow:none !important;}
:is(#spectral-gallery, #rgb-examples) .caption-label {left:4px; right:4px; bottom:3px; padding:2px; max-width:calc(100% - 8px); font-size:10px; line-height:1.2; white-space:normal; text-align:center;}
:is(#spectral-gallery, #rgb-examples) .thumbnail-lg:hover .caption-label {opacity:1;}
#model-panel, #preview-panel {min-height:100px; padding:12px; background:var(--block-background-fill) !important; border:1px solid var(--border-color-primary) !important; border-radius:10px !important;}
#model-controls, #preview-controls {gap:12px; min-height:74px; align-items:center;}
#model-controls {align-items:flex-end;}
:is(#model-panel, #preview-panel) .form {border:0; box-shadow:none; background:transparent;}
:is(#model-panel, #preview-panel) .styler {background:transparent !important;}
:is(#model-panel, #preview-panel) [data-testid="block-info"] {background:transparent; color:var(--body-text-color); padding:0; font-size:13px; font-weight:600; line-height:18px;}
#model-weights .wrap:not(.default) {min-height:40px; border:1px solid var(--input-border-color) !important; border-radius:7px !important;}
:is(#model-weights, #output-brightness, #pseudocolor) {border:0; padding:0; background:transparent;}
#refresh-weights {white-space:nowrap; min-height:40px; height:40px; align-self:flex-end; border:1px solid var(--border-color-primary) !important; border-radius:8px !important;}
#action-row {gap:12px;}
#reconstruct, #clear-images {height:44px; min-height:44px;}
#run-status {min-height:44px; padding:10px 12px; border:1px solid var(--border-color-primary) !important; border-radius:8px; background:var(--block-background-fill) !important;}
#run-status p {font-size:12px; line-height:22px; margin:0;}
#export-help p {font-size:13px; margin:0;}
#export-selection-panel, #wavelength-settings {min-height:44px; padding:10px 12px;}
:is(#export-selection-panel, #wavelength-settings) > .label-wrap {min-height:22px; align-items:center;}
#inference-settings, #spectrum-settings {padding:12px; flex:1 1 auto;}
:is(#inference-settings, #spectrum-settings) > .label-wrap {min-height:24px; align-items:center;}
:is(#export-selection-panel, #wavelength-settings, #inference-settings, #spectrum-settings) > .label-wrap {gap:12px; font-size:13px; font-weight:600;}
#spectral-plot {height:220px !important; min-height:220px;}
#spectral-plot .matplotlib {height:100% !important;}
#spectral-plot img {max-width:100%; max-height:100%; object-fit:contain;}
:is(#rgb-input, #channel-preview) > label {max-width:calc(100% - 68px); white-space:normal; line-height:1.3;}
#device-info p {font-size:12px; color:var(--body-text-color-subdued); margin:0;}
@media (max-width:900px) {
 .row {flex-direction:column !important;}
 .row > * {min-width:0 !important; width:100% !important; flex-basis:auto !important;}
 img {max-width:100%;}
 #hero h1 {font-size:27px;}
 #model-panel, #preview-panel, #model-controls, #preview-controls {min-height:0;}
 #spectral-plot {height:auto !important; min-height:160px;}
 #spectral-plot .matplotlib {height:auto !important;}
}
"""


def cleanup_result(result):
    if result and result.get("output_dir"):
        shutil.rmtree(result["output_dir"], ignore_errors=True)


def band_choices(waves):
    return [(f"Ch {i + 1:02d}", i + 1) for i in range(len(waves))]


def default_view():
    return {"channels": list(range(31)), "wavelengths": WAVELENGTHS.tolist()}


def build_demo(engine):
    choices = engine.available_models()
    initial = (
        "mst_plus_plus"
        if any(k == "mst_plus_plus" for _, k in choices)
        else (choices[0][1] if choices else None)
    )
    output_root = ROOT / "demo/outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    theme = gr.themes.Soft(
        primary_hue="indigo",
        neutral_hue="slate",
        font=["Arial", "sans-serif"],
        font_mono=["Consolas", "monospace"],
    ).set(
        block_border_width="1px",
        block_border_width_dark="1px",
        block_radius="10px",
        block_shadow="none",
        block_shadow_dark="none",
        block_label_background_fill="*neutral_50",
        block_label_background_fill_dark="*neutral_900",
        block_label_text_color="*neutral_700",
        block_label_text_color_dark="*neutral_200",
        block_label_text_size="13px",
        block_label_text_weight="600",
        block_title_text_color="*neutral_700",
        block_title_text_color_dark="*neutral_200",
        block_label_shadow="none",
        input_radius="7px",
        button_border_width="1px",
        button_border_width_dark="1px",
        button_large_text_size="14px",
        button_large_text_weight="600",
        button_large_padding="10px 16px",
        button_large_radius="8px",
        button_small_radius="8px",
        button_secondary_shadow="none",
        button_secondary_shadow_hover="none",
        button_secondary_shadow_active="none",
        button_primary_shadow="none",
        button_primary_shadow_hover="none",
        button_primary_shadow_active="none",
    )
    with gr.Blocks(
        title="MST++ · CVPRW 2022", theme=theme, css=CSS, analytics_enabled=False
    ) as demo:
        gr.Markdown("# MST++ · CVPRW 2022", elem_id="hero")
        gr.Markdown(
            "Upload an RGB image, choose a model, and click **Reconstruct**. Click a spectral thumbnail to inspect a band."
        )
        state = gr.State(value=None, time_to_live=3600, delete_callback=cleanup_result)
        current_band = gr.State(value=9)
        view_config = gr.State(value=default_view())
        with gr.Row(equal_height=False, elem_id="comparison-row"):
            with gr.Column(scale=1, min_width=320, elem_id="input-column"):
                source = gr.Image(
                    label="RGB input",
                    type="pil",
                    image_mode="RGB",
                    sources=["upload"],
                    height=370,
                    show_share_button=False,
                    elem_id="rgb-input",
                )
                examples = sorted((ROOT / "predict_code/demo").glob("*.jpg"))
                names = {
                    "ARAD_1K_0912": "ARAD",
                    "astronaut": "Astronaut",
                    "cat": "Cat",
                    "coffee": "Coffee",
                    "galaxies": "Galaxies",
                    "rocket": "Rocket",
                }
                gr.Markdown("**6 RGB examples**", elem_id="example-strip-label")
                example_gallery = gr.Gallery(
                    value=[(str(path), names.get(path.stem, path.stem)) for path in examples],
                    label="RGB examples",
                    show_label=False,
                    columns=6,
                    rows=1,
                    height=116,
                    object_fit="contain",
                    format="png",
                    interactive=False,
                    allow_preview=False,
                    preview=False,
                    show_share_button=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    elem_id="rgb-examples",
                )
                with gr.Group(elem_id="model-panel"):
                    with gr.Row(elem_id="model-controls"):
                        models = gr.Dropdown(
                            choices=choices,
                            value=initial,
                            label="Reconstruction model",
                            scale=4,
                            elem_id="model-weights",
                        )
                        refresh = gr.Button(
                            "Refresh weights", scale=1, min_width=160, elem_id="refresh-weights"
                        )
                with gr.Row(elem_id="action-row"):
                    run = gr.Button(
                        "Reconstruct", variant="primary", scale=3, elem_id="reconstruct"
                    )
                    clear = gr.Button("Clear", scale=1, elem_id="clear-images")
                status = gr.Markdown(READY, elem_id="run-status")
            with gr.Column(scale=1, min_width=320, elem_id="result-column"):
                result = gr.Image(
                    label="Reconstructed HSI · Ch 9 · 480 nm",
                    type="numpy",
                    interactive=False,
                    format="png",
                    height=370,
                    show_share_button=False,
                    show_download_button=True,
                    elem_id="channel-preview",
                )
                gr.Markdown(
                    "**Reconstructed spectral channels** · scroll and click to preview",
                    elem_id="channel-strip-label",
                )
                channels = gr.Gallery(
                    show_label=False,
                    columns=31,
                    rows=1,
                    height=116,
                    object_fit="contain",
                    format="png",
                    interactive=False,
                    allow_preview=False,
                    preview=False,
                    selected_index=None,
                    show_share_button=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    elem_id="spectral-gallery",
                )
                with gr.Group(elem_id="preview-panel"):
                    with gr.Row(elem_id="preview-controls"):
                        brightness = gr.Slider(
                            0.1,
                            3,
                            value=1,
                            step=0.05,
                            scale=3,
                            label="Preview brightness",
                            elem_id="output-brightness",
                        )
                        color = gr.Checkbox(
                            value=True,
                            label="Pseudocolor",
                            scale=1,
                            container=False,
                            elem_id="pseudocolor",
                        )
                with gr.Accordion(
                    "Export selected channels", open=False, elem_id="export-selection-panel"
                ):
                    export_selection = gr.CheckboxGroup(
                        choices=band_choices(WAVELENGTHS),
                        value=list(range(1, 32)),
                        label="Channels to export",
                        info="Tick the channels below. Thumbnail clicks only change the preview.",
                        elem_id="export-channel-selection",
                    )
                    with gr.Row():
                        select_all = gr.Button("Select all", size="sm")
                        select_none = gr.Button("Select none", size="sm")
                        select_current = gr.Button("Current channel only", size="sm")
                    export_help = gr.Markdown(
                        "Reconstruct an image first to enable export.", elem_id="export-help"
                    )
                    export = gr.Button(
                        "Export selected channels",
                        variant="primary",
                        interactive=False,
                        elem_id="export-selected",
                    )
                    with gr.Row():
                        zip_file = gr.File(
                            label="Selected PNG channels (ZIP)",
                            interactive=False,
                            height=85,
                            visible=False,
                            elem_id="channels-download",
                        )
                        mat_file = gr.File(
                            label="Full HSI cube (MAT)",
                            interactive=False,
                            height=85,
                            visible=False,
                            elem_id="cube-download",
                        )
                    gr.Markdown(
                        "Choose checkboxes → **Export selected channels** → download the ZIP. The MAT always contains all 31 bands."
                    )
                with gr.Accordion(
                    "Wavelength labels (advanced)", open=False, elem_id="wavelength-settings"
                ):
                    selected = gr.Textbox(
                        value="1-31",
                        label="Thumbnail channels",
                        info="1-31, or a list such as 9, 13, 27.",
                        elem_id="display-channels",
                    )
                    wavelengths = gr.Textbox(
                        value=DEFAULT_WAVELENGTH_TEXT,
                        lines=3,
                        label="Wavelength labels (nm)",
                        elem_id="wavelength-labels",
                    )
                    gr.Markdown(
                        "Labels and pseudocolors are editable. Models keep their 31 trained bands."
                    )
                    with gr.Row():
                        apply = gr.Button("Apply wavelength labels")
                        reset = gr.Button("Reset repository defaults")
        with gr.Row(equal_height=True, elem_id="settings-row"):
            with gr.Column(scale=1, min_width=320):
                with gr.Accordion("Inference settings", open=True, elem_id="inference-settings"):
                    with gr.Row():
                        max_size = gr.Dropdown(
                            choices=[
                                ("Original resolution", 0),
                                ("256 px", 256),
                                ("512 px", 512),
                                ("1024 px", 1024),
                                ("2048 px", 2048),
                            ],
                            value=512,
                            label="Maximum image size",
                            info="Longest edge; aspect ratio is preserved.",
                            elem_id="max-image-size",
                        )
                        ensemble = gr.Checkbox(value=False, label="Self-ensemble (8 passes)")
                    gr.Markdown(
                        "The complete RGB image is sent to the selected model. The output contains 31 spectral bands."
                    )
            with gr.Column(scale=1, min_width=320):
                with gr.Accordion(
                    "Spectrum · click the large result image to inspect a pixel",
                    open=True,
                    elem_id="spectrum-settings",
                ) as spectrum_panel:
                    spectrum = gr.Plot(show_label=False, elem_id="spectral-plot")
        gr.Markdown(
            f"**Device:** `{engine.device}` · **31 bands:** 400–700 nm · All inference and downloads remain local.",
            elem_id="device-info",
        )

        def export_summary(prediction, values):
            count = len(values or [])
            if prediction is None:
                return "Reconstruct an image first to enable export."
            return f"{count} channel{'s' if count != 1 else ''} selected. Click Export selected channels to prepare the ZIP."

        def render(prediction, band, gain, use_color, indices, waves, values):
            band = int(band) - 1
            if band not in indices:
                band = indices[0]
            preview = (
                None
                if prediction is None
                else display_band(prediction["cube"], band, waves, gain, use_color)
            )
            tiles = (
                []
                if prediction is None
                else gallery(prediction["cube"], indices, waves, gain, use_color, thumbnail_size=96)
            )
            return {
                result: gr.update(
                    value=preview, label=f"Reconstructed HSI · Ch {band + 1} · {waves[band]:g} nm"
                ),
                channels: gr.update(
                    value=tiles, selected_index=indices.index(band) if prediction else None
                ),
                spectrum: None
                if prediction is None
                else spectrum_figure(prediction["cube"], waves),
                zip_file: gr.update(value=None, visible=False),
                current_band: band + 1,
                view_config: {"channels": indices, "wavelengths": list(map(float, waves))},
                export_selection: gr.update(choices=band_choices(waves)),
                export_help: export_summary(prediction, values),
                export: gr.update(interactive=prediction is not None and bool(values)),
            }

        view_outputs = [
            result,
            channels,
            spectrum,
            zip_file,
            current_band,
            view_config,
            export_selection,
            export_help,
            export,
        ]
        outputs = [*view_outputs, mat_file, state, status]
        serial = dict(concurrency_id="session", concurrency_limit=1)

        def reconstruct(
            image,
            key,
            size,
            use_ensemble,
            band,
            gain,
            use_color,
            selection,
            labels,
            values,
            old,
            progress=gr.Progress(),
        ):
            prediction = None
            try:
                indices, waves = parse_channels(selection), parse_wavelengths(labels)
                prediction = engine.reconstruct(image, key, size, use_ensemble, progress)
                prediction["output_dir"] = tempfile.mkdtemp(
                    prefix="reconstruction-", dir=output_root
                )
                updates = render(prediction, band, gain, use_color, indices, waves, values)
                w, h = prediction["output_size"]
                updates.update(
                    {
                        mat_file: gr.update(value=export_cube(prediction), visible=True),
                        state: prediction,
                        status: f"**{MODELS[key][0]}** · {w} × {h} × 31 · {prediction['seconds']:.2f} s · Full image",
                    }
                )
                cleanup_result(old)
                progress(1, desc="Done")
                return updates
            except (ValueError, FileNotFoundError) as exc:
                cleanup_result(prediction)
                raise gr.Error(str(exc)) from exc
            except Exception as exc:
                cleanup_result(prediction)
                LOG.exception("Reconstruction failed")
                raise gr.Error(f"Reconstruction failed: {exc}") from exc

        run.click(
            reconstruct,
            [
                source,
                models,
                max_size,
                ensemble,
                current_band,
                brightness,
                color,
                selected,
                wavelengths,
                export_selection,
                state,
            ],
            outputs,
            api_name="reconstruct",
            **serial,
        )

        def clear_result(old):
            cleanup_result(old)
            return {
                result: gr.update(value=None),
                channels: gr.update(value=[], selected_index=None),
                spectrum: None,
                zip_file: gr.update(value=None, visible=False),
                mat_file: gr.update(value=None, visible=False),
                state: None,
                current_band: 9,
                status: READY,
                export_help: export_summary(None, []),
                export: gr.update(interactive=False),
            }

        source.change(clear_result, [state], outputs, api_name=False, **serial)
        models.change(clear_result, [state], outputs, api_name=False, **serial)

        def clear_all(old):
            return {source: None, **clear_result(old)}

        clear.click(clear_all, [state], [source, *outputs], api_name=False, **serial)

        def refresh_weights(current):
            available = engine.available_models()
            keys = [key for _, key in available]
            return gr.update(
                choices=available, value=current if current in keys else (keys[0] if keys else None)
            )

        refresh.click(refresh_weights, [models], models, api_name=False, **serial)

        def apply_view(prediction, band, gain, use_color, selection, labels, values):
            try:
                return render(
                    prediction,
                    band,
                    gain,
                    use_color,
                    parse_channels(selection),
                    parse_wavelengths(labels),
                    values,
                )
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc

        apply.click(
            apply_view,
            [state, current_band, brightness, color, selected, wavelengths, export_selection],
            view_outputs,
            api_name="update_channels",
            **serial,
        )

        def refresh_display(prediction, band, gain, use_color, config, values):
            return render(
                prediction, band, gain, use_color, config["channels"], config["wavelengths"], values
            )

        appearance = [state, current_band, brightness, color, view_config, export_selection]
        brightness.release(refresh_display, appearance, view_outputs, api_name=False, **serial)
        color.change(refresh_display, appearance, view_outputs, api_name=False, **serial)

        def thumbnail_select(prediction, gain, use_color, config, event: gr.SelectData):
            if prediction is None:
                return gr.skip(), gr.skip()
            position = int(event.index)
            if not 0 <= position < len(config["channels"]):
                raise gr.Error("The channel list changed. Click a thumbnail again.")
            index = config["channels"][position]
            waves = config["wavelengths"]
            return (
                gr.update(
                    value=display_band(prediction["cube"], index, waves, gain, use_color),
                    label=f"Reconstructed HSI · Ch {index + 1} · {waves[index]:g} nm",
                ),
                index + 1,
            )

        channels.select(
            thumbnail_select,
            [state, brightness, color, view_config],
            [result, current_band],
            api_name=False,
            **serial,
        )

        def select_example(event: gr.SelectData):
            return str(examples[int(event.index)])

        example_gallery.select(select_example, outputs=source, api_name=False, **serial)

        def defaults(prediction, band, gain, use_color):
            updates = render(
                prediction, band, gain, use_color, list(range(31)), WAVELENGTHS, list(range(1, 32))
            )
            updates.update(
                {
                    selected: "1-31",
                    wavelengths: DEFAULT_WAVELENGTH_TEXT,
                    export_selection: gr.update(
                        choices=band_choices(WAVELENGTHS), value=list(range(1, 32))
                    ),
                }
            )
            return updates

        reset.click(
            defaults,
            [state, current_band, brightness, color],
            [selected, wavelengths, *view_outputs],
            api_name=False,
            **serial,
        )
        select_current.click(
            lambda band: [int(band)], [current_band], export_selection, api_name=False, **serial
        )
        select_all.click(
            lambda: list(range(1, 32)), outputs=export_selection, api_name=False, **serial
        )
        select_none.click(lambda: [], outputs=export_selection, api_name=False, **serial)

        def selection_changed(values, prediction):
            return {
                export_help: export_summary(prediction, values),
                export: gr.update(interactive=prediction is not None and bool(values)),
                zip_file: gr.update(value=None, visible=False),
            }

        export_selection.change(
            selection_changed,
            [export_selection, state],
            [export_help, export, zip_file],
            api_name=False,
            **serial,
        )

        def export_views(prediction, values, config, gain, use_color):
            try:
                if prediction is None:
                    raise ValueError("Reconstruct an image before exporting channels.")
                if not values:
                    raise ValueError("Select at least one channel to export.")
                path = export_channels(
                    prediction,
                    [int(value) - 1 for value in values],
                    config["wavelengths"],
                    gain,
                    use_color,
                )
                return gr.update(value=path, visible=True)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc

        export.click(
            export_views,
            [state, export_selection, view_config, brightness, color],
            zip_file,
            api_name="export_channels",
            **serial,
        )

        def pixel_spectrum(prediction, config, event: gr.SelectData):
            if prediction is None:
                return gr.skip(), gr.skip()
            x, y = map(int, event.index)
            h, w, _ = prediction["cube"].shape
            return (
                spectrum_figure(
                    prediction["cube"],
                    config["wavelengths"],
                    (min(max(x, 0), w - 1), min(max(y, 0), h - 1)),
                ),
                gr.update(open=True),
            )

        result.select(
            pixel_spectrum,
            [state, view_config],
            [spectrum, spectrum_panel],
            api_name=False,
            **serial,
        )
    return demo.queue(max_size=16, default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "pretrained_weights")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7875)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(max(1, args.threads))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    engine = Reconstructor(args.weights_dir, args.device)
    build_demo(engine).launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
