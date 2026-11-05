"""Adapter Tuning Explorer: a Streamlit app for seeing how adapter tuning adapts a pretrained model.

Run with:  uv run streamlit run app.py
"""

import gc
import html
import json
from dataclasses import asdict

import altair as alt
import pandas as pd
import streamlit as st
import torch

from adapter_lab.data import (
    NOTEBOOK_TEST_IMAGE,
    NOTEBOOK_TRAIN_IMAGE,
    download_sample,
    is_downloaded,
    load_split,
)
from adapter_lab.labels import IGNORE, token_rows
from adapter_lab.lora import full_params, lora_params, total_lora_params, toy_layer
from adapter_lab.messages import format_message
from adapter_lab.quant import (
    BLOCK_SIZE,
    FP4_LEVELS,
    NF4_LEVELS,
    bits_per_weight,
    mean_abs_error,
    quantize,
    weights_gb,
)
from adapter_lab.recorded import (
    GEMMA_COUNTS,
    GEMMA_LOSS,
    GEMMA_TRAINING_TIME,
    GEMMA_TUNED_CAPTION,
    GPT_CAPTION,
)
from adapter_lab.runs import REFERENCE_NAME, delete_run, list_runs, new_run_name
from adapter_lab.training import (
    BATCH_SIZE,
    DROPOUT,
    GRADIENT_ACCUMULATION,
    MAX_GRAD_NORM,
    REFERENCE_CONFIG,
    WARMUP_RATIO,
    TrainConfig,
    TrainingJob,
)
from adapter_lab.vlm import (
    IMAGE_EDGE,
    MODEL_ID,
    CaptionModel,
    encode,
    image_token_ids,
    labels_for,
    linear_layers,
    load_processor,
    read_weight,
    read_weights,
)

DOC_LINK = "docs/how-adapter-tuning-works.md"
WARNING = (
    "**Learning tool only.** The captions here can invent findings that are not in the image. "
    "Never use them for medical interpretation."
)
REGION_STYLE = {
    "prompt": ("#fde68a", "Prompt"),
    "image": ("#d1d5db", "Image"),
    "caption": ("#a7f3d0", "Caption"),
    "padding": ("#e5e7eb", "Padding"),
}

SERIES_COLORS = {
    "This run: what training minimises": "#111827",
    "This run: prompt tokens only": "#d97706",
    "This run: caption tokens only": "#059669",
    "Notebook: Gemma 3 4B (recorded)": "#9ca3af",
}

st.set_page_config(page_title="Adapter Tuning Explorer", page_icon="🩻", layout="wide")


# ---- shared resources -------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the base model (the first run downloads about 500 MB)…")
def get_base_model() -> CaptionModel:
    return CaptionModel()


@st.cache_resource(max_entries=1, show_spinner="Loading the base model with this adapter…")
def get_tuned_model(run_name: str, adapter_dir: str) -> CaptionModel:
    return CaptionModel(adapter_dir)


@st.cache_resource(show_spinner="Loading the processor…")
def get_processor():
    return load_processor()


@st.cache_resource
def job_slot() -> dict:
    """Survives page refreshes, so a run in progress isn't lost. Holds at most one job."""
    return {"job": None}


@st.cache_data(show_spinner="Reading the model's layers…")
def get_linear_layers() -> list[dict]:
    return linear_layers()


@st.cache_data(show_spinner="Reading the checkpoint (the first run downloads about 500 MB)…")
def get_weight_names() -> dict:
    return read_weights()


@st.cache_data
def get_split(split: str):
    return load_split(split)


def heatmap(matrix: torch.Tensor, title: str, domain: float, cell: int = 18):
    rows = [
        {"row": i, "column": j, "value": matrix[i, j].item()}
        for i in range(matrix.shape[0])
        for j in range(matrix.shape[1])
    ]
    return (
        alt.Chart(pd.DataFrame(rows), title=title)
        .mark_rect()
        .encode(
            x=alt.X("column:O", axis=None),
            y=alt.Y("row:O", axis=None),
            color=alt.Color(
                "value:Q", scale=alt.Scale(scheme="redblue", domain=[-domain, domain], reverse=True), legend=None
            ),
            tooltip=[alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(width=cell * matrix.shape[1], height=cell * matrix.shape[0])
    )


def token_spans(rows) -> str:
    """The training text with each token coloured by region. The run of image slots is collapsed."""
    parts = []
    i = 0
    while i < len(rows):
        row = rows[i]
        if row.token == "<image>":
            end = i
            while end < len(rows) and rows[end].token == "<image>":
                end += 1
            text = f"&lt;image&gt; × {end - i}"
            i = end
        else:
            text = html.escape(row.token.replace("Ġ", "␣").replace("Ċ", "⏎"))
            i += 1
        color, _ = REGION_STYLE[row.region]
        strike = "text-decoration:line-through;" if row.label == IGNORE else ""
        parts.append(
            f'<span style="background:{color};color:#111;padding:1px 3px;margin:1px;'
            f'border-radius:3px;display:inline-block;font-family:monospace;font-size:13px;{strike}">{text}</span>'
        )
    return "".join(parts)


def loss_chart(frame: pd.DataFrame):
    return (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=18))
        .encode(
            x=alt.X("optimizer step:Q", axis=alt.Axis(format="d", tickMinStep=1)),
            y=alt.Y("loss:Q", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(SERIES_COLORS), range=list(SERIES_COLORS.values())),
                legend=alt.Legend(orient="bottom", title=None, labelLimit=420),
            ),
            tooltip=["series", "optimizer step", alt.Tooltip("loss:Q", format=".3f")],
        )
        .properties(height=320)
    )


def per_step(history: list[dict], column: str) -> list[tuple[int, float]]:
    """Average a per-example measure over each group of examples that shares an optimizer step."""
    groups: dict[int, list[float]] = {}
    for index, row in enumerate(history):
        if row.get(column) is not None:
            groups.setdefault(index // GRADIENT_ACCUMULATION + 1, []).append(row[column])
    return [(step, sum(values) / len(values)) for step, values in groups.items()]


def history_frame(history: list[dict], include_gemma: bool) -> pd.DataFrame:
    rows = []
    columns = {"loss": "what training minimises", "prompt_loss": "prompt tokens only", "caption_loss": "caption tokens only"}
    for column, name in columns.items():
        label = f"This run: {name}"
        rows += [{"optimizer step": s, "loss": v, "series": label} for s, v in per_step(history, column)]
    if include_gemma:
        rows += [{"optimizer step": s, "loss": v, "series": "Notebook: Gemma 3 4B (recorded)"} for s, v in GEMMA_LOSS]
    return pd.DataFrame(rows)


def show_results(run) -> None:
    """The measurements a finished run saved, shown in the training tab."""
    r = run.results
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trainable parameters", f"{r['trainable_params']:,}",
              f"{r['trainable_params'] / r['total_params']:.2%} of all", delta_color="off", delta_arrow="off")
    c2.metric("Adapter on disk", f"{r['adapter_bytes'] / 1e6:.0f} MB")
    c3.metric("Training time", f"{r['training_seconds'] / 60:.1f} min",
              f"{r['examples_trained']} examples, {r['unique_images']} unique images", delta_color="off", delta_arrow="off")
    c4.metric("Peak memory", f"{r['peak_memory_gb']:.2f} GB")

    if r["frozen_weights_changed"]:
        st.error(f"{len(r['frozen_weights_changed'])} base weights changed: {r['frozen_weights_changed'][:5]}")
    else:
        st.success(
            f"**Base model frozen.** Fingerprints of all {r['frozen_weights_checked']:,} original weight tensors "
            "were identical before and after training."
        )
    st.markdown("**Where the trainable parameters are**")
    st.dataframe(
        pd.DataFrame([{"part of the model": k, "trainable parameters": f"{v:,}"} for k, v in r["trainable_by_region"].items()]),
        hide_index=True,
    )


def describe_config(config: dict) -> str:
    return (
        f"{config['examples']} examples, r={config['r']}, α={config['alpha']}, lr={config['learning_rate']:g}"
        + (", prompt masked" if config["mask_prompt"] else "")
        + (", modules_to_save" if config["modules_to_save"] else "")
    )


# ---- sidebar ----------------------------------------------------------------------------

with st.sidebar:
    st.title("🩻 Adapter Tuning Explorer")
    st.caption(f"Base model: `{MODEL_ID}`")
    st.markdown(
        "**Learning path**\n"
        "1. Why tune, and the data\n"
        "2. Inside LoRA\n"
        "3. QLoRA and memory\n"
        "4. Train an adapter\n"
        "5. Test the adapter\n\n"
        f"Read the companion guide: `{DOC_LINK}`"
    )
    st.warning(WARNING)

if not is_downloaded():
    st.title("🩻 Adapter Tuning Explorer")
    st.info("First, download the radiology sample: 100 training and 100 test images, about 7 MB.")
    if st.button("Download the sample", type="primary"):
        with st.spinner("Downloading…"):
            download_sample()
        st.rerun()
    st.stop()

train_examples = get_split("train")
test_examples = get_split("test")

# Each model copy takes about 1 GB, which is a lot on an 8 GB laptop, so captioning waits until training ends.
current_job = job_slot()["job"]
busy = current_job is not None and current_job.running
LIVE_REFRESH_SECONDS = 10
BUSY_NOTE = "Captioning is paused while an adapter trains, to leave it enough memory."

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① Why tune, and the data", "② Inside LoRA", "③ QLoRA and memory", "④ Train an adapter", "⑤ Test the adapter",
])

# ---- ① why tune, and the data -----------------------------------------------------------

with tab1:
    st.header("① Why tune, and the data")
    st.markdown(
        "A general-purpose model can describe a scan, but it doesn't write like a radiology paper. "
        "**Adapter tuning** teaches it that style from a small set of examples, by training only a small "
        "part of the model. Before training anything, look at what the model has to learn from."
    )

    st.subheader("What the base model writes today")
    default = next((i for i, e in enumerate(train_examples) if e.image_id == NOTEBOOK_TRAIN_IMAGE), 0)
    index = st.selectbox(
        "Training example", range(len(train_examples)), index=default,
        format_func=lambda i: f"{train_examples[i].image_id}: {train_examples[i].caption[:70]}…",
    )
    example = train_examples[index]
    left, right = st.columns([1, 2])
    left.image(example.path)
    with right:
        st.markdown(f"**Real caption, from the published paper**\n\n> {example.caption}")
        if example.image_id == NOTEBOOK_TRAIN_IMAGE:
            st.markdown(f"**GPT-4o-mini** (foundational model, recorded in notebook 2)\n\n> {GPT_CAPTION}")
            st.caption(
                "Notice that the real caption mentions bacterial contamination and a 15-month temporary filling. "
                "Neither can be seen in the image, so no model could get them right from the picture alone."
            )
        if busy:
            st.caption(BUSY_NOTE)
        if st.button("Caption with the base model", key="t1_caption", disabled=busy):
            st.session_state["t1"] = (example.image_id, get_base_model().caption(example.path)[0])
        if st.session_state.get("t1", (None,))[0] == example.image_id:
            st.markdown(f"**SmolVLM base model** (live)\n\n> {st.session_state['t1'][1]}")

    st.subheader("One training example")
    st.markdown(
        "Every example is three chat messages: the instructions, the image with the request, and the "
        "answer we want. This is the same format the notebook writes to `messages.jsonl`."
    )
    with st.expander("Show the message"):
        st.json(format_message(example.path, example.caption))

    st.subheader("What the model is actually graded on")
    st.markdown(
        "The processor turns the message into tokens. To train, the model predicts each token from the ones "
        "before it. **Labels** say which predictions count: a label of `-100` means *ignore this one*. "
        "Struck-through tokens below are ignored."
    )
    mask_prompt_view = st.toggle(
        "Also mask the prompt from the labels", value=False,
        help="The notebook leaves this off, so the model is graded on predicting the fixed prompt too.",
    )
    processor = get_processor()
    encoded = encode(processor, example.path, example.caption)
    labels = labels_for(processor, encoded, mask_prompt_view)
    tokens = processor.tokenizer.convert_ids_to_tokens(encoded.input_ids)
    rows = token_rows(
        encoded.input_ids, tokens, labels, image_token_ids(processor),
        processor.tokenizer.pad_token_id, encoded.prompt_length,
    )
    legend = " ".join(
        f'<span style="background:{color};color:#111;padding:1px 6px;border-radius:3px">{name}</span>'
        for key, (color, name) in REGION_STYLE.items() if key != "padding"
    )
    st.markdown(legend, unsafe_allow_html=True)
    st.markdown(f'<div style="line-height:1.9">{token_spans(rows)}</div>', unsafe_allow_html=True)
    count = lambda region, graded: sum(r.region == region and (r.label != IGNORE) == graded for r in rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("Image tokens (ignored)", count("image", False))
    c2.metric("Prompt tokens graded", count("prompt", True), f"{count('prompt', False)} ignored", delta_color="off", delta_arrow="off")
    c3.metric("Caption tokens graded", count("caption", True))
    if not mask_prompt_view:
        st.info(
            "With the notebook's labels, the prompt is graded too, and it's the same in every example. "
            "Tab ④ measures what that does to the loss."
        )

# ---- ② inside LoRA ----------------------------------------------------------------------

with tab2:
    st.header("② Inside LoRA")
    st.markdown(
        "A model is mostly **linear layers**, each one a big matrix of weights **W**. LoRA leaves W frozen and "
        "adds a correction built from two thin matrices, **A** and **B**:"
    )
    st.latex(r"\text{output} = \left(W + \frac{\alpha}{r}\,B A\right)\text{input}")
    st.markdown(
        "**r** (the rank) sets how thin A and B are, so it limits how complicated the correction can be. "
        "**α** (alpha) with r sets how strongly the correction applies."
    )

    c1, c2, c3 = st.columns(3)
    toy_r = c1.slider("r (rank)", 1, 8, 2, key="toy_r")
    toy_alpha = c2.slider("α (alpha)", 1, 16, 4, key="toy_alpha")
    trained = c3.toggle("After some training", value=True, help="Off shows how LoRA starts: B is all zeros.")
    layer = toy_layer(8, toy_r, toy_alpha, trained)
    domain = max(layer.W.abs().max().item(), 1e-6)
    st.altair_chart(
        alt.hconcat(
            heatmap(layer.W, "W (frozen)", domain),
            heatmap(layer.B, f"B ({8}×{toy_r})", domain),
            heatmap(layer.A, f"A ({toy_r}×{8})", domain),
            heatmap(layer.delta, f"(α/r)·BA", domain),
            heatmap(layer.adapted, "W + (α/r)·BA", domain),
        ).resolve_scale(color="independent"),
    )
    st.caption(
        "An illustrative 8×8 layer with made-up numbers. "
        f"Scale α/r = {layer.scale:.2f}. The rank of the correction is {layer.delta_rank}."
    )
    if not trained:
        st.info("B starts at zero, so the correction is zero and the tuned model begins identical to the base model.")

    st.subheader("Why this saves so much")
    layers = get_linear_layers()
    shapes = [(l["d_in"], l["d_out"]) for l in layers]
    real_r = st.slider("r for the real model", 1, 64, 16, key="real_r")
    names = [l["name"] for l in layers]
    choice = st.selectbox("A real SmolVLM layer", names, index=next(i for i, n in enumerate(names) if "text_model" in n))
    chosen = layers[names.index(choice)]
    full = full_params(chosen["d_in"], chosen["d_out"])
    small = lora_params(chosen["d_in"], chosen["d_out"], real_r)
    c1, c2, c3 = st.columns(3)
    c1.metric("Shape of W", f"{chosen['d_out']} × {chosen['d_in']}")
    c2.metric("Weights in W", f"{full:,}")
    c3.metric("Weights in A + B", f"{small:,}", f"{small / full:.1%} of W", delta_color="off", delta_arrow="off")
    st.caption(
        f"The count for A + B is r × (inputs + outputs) = {real_r} × ({chosen['d_in']} + {chosen['d_out']}). "
        "The saving shrinks as r grows and for small layers. At r=64 the smallest layers here reach 44% of W."
    )

    whole = total_lora_params(shapes, real_r)
    all_weights = sum(full_params(a, b) for a, b in shapes)
    gemma_lora = GEMMA_COUNTS.lora_at_r16 * real_r // 16
    st.dataframe(
        pd.DataFrame([
            {"model": "SmolVLM 256M (this app)", "linear layers": len(layers), "weights in linear layers": f"{all_weights:,}",
             "LoRA weights": f"{whole:,}", "LoRA as % of linear": f"{whole / all_weights:.2%}"},
            {"model": "Gemma 3 4B (notebook)", "linear layers": 400, "weights in linear layers": f"{GEMMA_COUNTS.quantizable:,}",
             "LoRA weights": f"{gemma_lora:,}", "LoRA as % of linear": f"{gemma_lora / GEMMA_COUNTS.quantizable:.2%}"},
        ]),
        hide_index=True,
    )
    st.warning(
        f"**The notebook's quirk.** Its `modules_to_save` also trains full copies of `embed_tokens` and `lm_head`: "
        f"{GEMMA_COUNTS.modules_to_save:,} more weights, about {GEMMA_COUNTS.modules_to_save / GEMMA_COUNTS.lora_at_r16:.0f} "
        "times the LoRA weights at r=16. Tab ④ lets you try it on SmolVLM."
    )

# ---- ③ QLoRA and memory -----------------------------------------------------------------

with tab3:
    st.header("③ QLoRA and memory")
    st.markdown(
        "LoRA shrinks what you *train*, but the whole base model still has to fit in memory. **QLoRA** adds "
        "**quantization**: storing each frozen weight in 4 bits instead of 16 or 32. Four bits only allow 16 "
        "different values, so the question is which 16."
    )
    st.info(
        "**This tab is a simulation.** The notebook uses `bitsandbytes`, which isn't practical on a Mac, so the "
        "training in tab ④ is plain LoRA. Here the same arithmetic runs on SmolVLM's real weights."
    )

    weight_names = get_weight_names()
    keys = sorted(weight_names)
    default_key = next((i for i, k in enumerate(keys) if "text_model.layers.0" in k and "mlp" in k), 0)
    key = st.selectbox("A real weight matrix", keys, index=default_key, format_func=lambda k: f"{k} {weight_names[k]}")
    double_quant = st.toggle("Double quantization (also compress each block's scale)", value=True)

    weights = read_weight(key).flatten()[:200_000]
    nf4 = quantize(weights, NF4_LEVELS)
    fp4 = quantize(weights, FP4_LEVELS)
    c1, c2, c3 = st.columns(3)
    c1.metric("NF4 average error", f"{mean_abs_error(weights, nf4):.5f}")
    c2.metric("Evenly spaced average error", f"{mean_abs_error(weights, fp4):.5f}")
    c3.metric("Bits per weight (down from 32)", f"{bits_per_weight(double_quant):.3f}")

    scales = torch.stack([
        weights[i : i + BLOCK_SIZE].abs().max() for i in range(0, weights.numel(), BLOCK_SIZE)
    ]).repeat_interleave(BLOCK_SIZE)[: weights.numel()]
    scaled = (weights / scales.clamp_min(1e-12)).numpy()
    counts, edges = torch.histogram(torch.from_numpy(scaled).double(), bins=80, range=(-1.0, 1.0))
    bars = pd.DataFrame({"start": edges[:-1].numpy(), "end": edges[1:].numpy(), "weights": counts.numpy()})
    histogram = (
        alt.Chart(bars)
        .mark_rect(opacity=0.5, color="#6b7280")
        .encode(
            x=alt.X("start:Q", title="weight ÷ its block's largest weight"),
            x2="end:Q",
            y=alt.Y("weights:Q"),
            y2=alt.datum(0),
        )
        .properties(height=280)
    )
    levels = pd.DataFrame(
        [{"level": v.item(), "scheme": "NF4 levels"} for v in NF4_LEVELS]
        + [{"level": v.item(), "scheme": "Evenly spaced levels"} for v in FP4_LEVELS]
    )
    rules = alt.Chart(levels).mark_rule(strokeWidth=1.5).encode(
        x="level:Q",
        color=alt.Color(
            "scheme:N",
            scale=alt.Scale(domain=["NF4 levels", "Evenly spaced levels"], range=["#ea580c", "#2563eb"]),
            legend=alt.Legend(orient="bottom", title=None),
        ),
    )
    st.altair_chart(histogram + rules)
    st.caption(
        "Real weights bunch up around zero. NF4 places its 16 levels closer together there, so most weights land "
        "near a level. Up to 200,000 weights of this matrix are used."
    )

    st.subheader("What this means for the notebook's Gemma 3 4B")
    memory = [
        ("All weights at 16 bits", weights_gb(GEMMA_COUNTS, quantized=False)),
        ("4-bit NF4 (linear layers only)", weights_gb(GEMMA_COUNTS, quantized=True, double_quant=double_quant)),
        ("+ LoRA at r=16", weights_gb(GEMMA_COUNTS, quantized=True, double_quant=double_quant, lora=True)),
        ("+ the notebook's modules_to_save",
         weights_gb(GEMMA_COUNTS, quantized=True, double_quant=double_quant, lora=True, modules_to_save=True)),
    ]
    frame = pd.DataFrame([{"setup": name, "GB": gb} for name, gb in memory])
    order = [name for name, _ in memory]
    st.altair_chart(
        alt.Chart(frame).mark_bar().encode(
            x=alt.X("GB:Q", title="memory for weights (GB)"),
            y=alt.Y("setup:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
            tooltip=[alt.Tooltip("GB:Q", format=".2f")],
        ).properties(height=170)
    )
    st.caption(
        "Estimates from parameter counts, assuming 16-bit storage for anything not quantized. They cover weights "
        "only. Training also needs gradients, optimizer state and the batch itself. PEFT may keep trainable "
        "weights at 32 bits, which would raise the last two rows."
    )

# ---- ④ train an adapter -----------------------------------------------------------------


def start_job(config: TrainConfig, name: str) -> None:
    get_base_model.clear()
    get_tuned_model.clear()
    gc.collect()
    job = TrainingJob(config, train_examples, test_examples, name)
    job_slot()["job"] = job
    job.start()


with tab4:
    st.header("④ Train an adapter")
    st.markdown(
        "This trains a real LoRA adapter on your machine. You can move between tabs while it runs, but "
        "captioning is paused until it ends. After the last example (or Stop), it spends another two or three "
        "minutes measuring held-out loss and captioning test images, so that even a stopped run can be tested."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        examples = st.slider("Training examples", 20, 200, 60, step=4, help="About 4 seconds each on a laptop CPU.")
        learning_rate = st.select_slider(
            "Learning rate", [5e-5, 1e-4, 2e-4, 5e-4, 1e-3], value=2e-4, format_func=lambda v: f"{v:g}"
        )
    with c2:
        r = st.slider("r (rank)", 4, 32, 16, step=4)
        alpha = st.slider("α (alpha)", 4, 32, 16, step=4)
    with c3:
        mask_prompt = st.toggle("Mask the prompt from the labels", value=False,
                                help="Off is what the notebook does: the prompt counts towards the loss.")
        modules_to_save = st.toggle("modules_to_save", value=False,
                                    help="On is what the notebook does: also trains full copies of two large layers.")
        st.caption(
            f"Fixed at the notebook's values: batch size {BATCH_SIZE}, gradient accumulation "
            f"{GRADIENT_ACCUMULATION}, gradient clipping {MAX_GRAD_NORM}, dropout {DROPOUT}, warmup {WARMUP_RATIO:.0%}."
        )

    b1, b2, _ = st.columns([1, 2, 3])
    if b1.button("Train", type="primary", disabled=busy):
        start_job(TrainConfig(examples, r, alpha, learning_rate, mask_prompt, modules_to_save), new_run_name())
        st.rerun()
    reference_exists = any(run.name == REFERENCE_NAME for run in list_runs())
    if b2.button(
        "Train the reference run", disabled=busy or reference_exists,
        help=f"The run the guide's numbers come from: {describe_config(asdict(REFERENCE_CONFIG))}, fixed seed. "
             "It takes about 8 minutes." + (" It already exists." if reference_exists else ""),
    ):
        start_job(REFERENCE_CONFIG, REFERENCE_NAME)
        st.rerun()

    # Redrawing competes with training for the CPU: refreshing an Altair chart every 2 seconds made each
    # example take about 3 times as long. So the live view refreshes rarely and uses a lighter chart.
    @st.fragment(run_every=LIVE_REFRESH_SECONDS if busy else None)
    def live_progress():
        job = job_slot()["job"]
        if job is None:
            return
        if job.error:
            st.error(f"Training failed: {job.error}")
            return
        if job.running:
            done = len(job.history)
            st.progress(done / job.config.examples, text=f"{job.stage}{' (stopping…)' if job.stopping else ''}")
            if st.button("Stop", disabled=job.stopping):
                job.stop()
            if job.history:
                live = history_frame(job.history, include_gemma=False)
                live["series"] = live["series"].str.replace("This run: ", "")
                st.line_chart(live, x="optimizer step", y="loss", height=280,
                              color="series")
                last = job.history[-1]
                c1, c2, c3 = st.columns(3)
                c1.metric("Adapter size (norm of B)", f"{last['adapter_norm']:.2f}")
                c2.metric("Seconds per example", f"{last['seconds']:.1f}")
                c3.metric("Peak memory", f"{last['peak_memory_gb']:.2f} GB")
        else:
            # The fragment's timer belongs to the page that started it, so refresh the whole page once.
            if not st.session_state.get(f"seen_{job.name}"):
                st.session_state[f"seen_{job.name}"] = True
                st.rerun()

    live_progress()

    runs = list_runs()
    if runs:
        st.subheader("Finished runs")
        labels = {run.name: f"{run.name}: {describe_config(run.config)} ({run.results['status']})" for run in runs}
        selected = st.selectbox("Run", [run.name for run in runs], format_func=labels.get, key="t4_run")
        run = next(run for run in runs if run.name == selected)
        include_gemma = st.toggle("Show the notebook's recorded Gemma curve", value=True)
        st.altair_chart(loss_chart(history_frame(run.history, include_gemma)))
        st.caption(
            f"Each point averages {GRADIENT_ACCUMULATION} examples. The Gemma curve is from notebook 3 "
            f"({GEMMA_TRAINING_TIME}), logged every 5 steps, so its points are averages of 20 examples."
        )
        show_results(run)
        if not busy and st.button(f"Delete {selected}"):
            delete_run(selected)
            st.rerun()

# ---- ⑤ test the adapter -----------------------------------------------------------------

with tab5:
    st.header("⑤ Test the adapter")
    st.warning(WARNING)
    runs = list_runs()
    if not runs:
        st.info("No adapter yet. Train one in tab ④, or start the reference run there.")
    else:
        labels = {run.name: f"{run.name}: {describe_config(run.config)}" for run in runs}
        selected = st.selectbox("Adapter", [run.name for run in runs], format_func=labels.get, key="t5_run")
        run = next(run for run in runs if run.name == selected)

        st.subheader("Did it learn anything real?")
        st.markdown(
            f"Training loss can't answer this. Here is the loss on **{run.results['held_out_images']} images the "
            "adapter never saw**, counting **caption tokens only**, so remembering the prompt earns nothing."
        )
        base_loss, tuned_loss = run.results["held_out_base"], run.results["held_out_tuned"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Held-out loss, base model", f"{base_loss:.3f}")
        c2.metric("Held-out loss, tuned model", f"{tuned_loss:.3f}", f"{tuned_loss - base_loss:+.3f}", delta_color="inverse")
        c3.metric("Training loss on captions, last 20 examples",
                  f"{sum(h['caption_loss'] for h in run.history[-20:]) / max(len(run.history[-20:]), 1):.3f}")

        st.subheader("Captions side by side")
        st.caption("Saved when the run finished. Compare style first, then check each claim against the real caption.")
        for item in run.results["gallery"]:
            left, right = st.columns([1, 3])
            left.image(item["path"], caption=item["image_id"])
            right.markdown(
                f"**Real:** {item['real']}\n\n**Base model:** {item['base']}\n\n**Tuned model:** {item['tuned']}"
            )
            if item["image_id"] == NOTEBOOK_TEST_IMAGE:
                right.markdown(f"**Gemma 3 4B tuned** (recorded in notebook 4): {GEMMA_TUNED_CAPTION}")
            st.divider()

        st.subheader("Try it yourself")
        default = next((i for i, e in enumerate(test_examples) if e.image_id == NOTEBOOK_TEST_IMAGE), 0)
        index = st.selectbox(
            "Test image", range(len(test_examples)), index=default,
            format_func=lambda i: f"{test_examples[i].image_id}: {test_examples[i].caption[:70]}…", key="t5_image",
        )
        example = test_examples[index]
        strength = st.slider(
            "Adapter strength", 0.0, 1.0, 1.0, step=0.1,
            help="0 is the base model and 1 is the adapter as trained. It multiplies the LoRA correction.",
        )
        if run.config["modules_to_save"] and 0 < strength < 1:
            st.caption("This run trained modules_to_save. Strength only scales the LoRA part, not those full copies.")

        if busy:
            st.caption(BUSY_NOTE)
        if st.button("Caption this image", type="primary", disabled=busy):
            tuned = get_tuned_model(run.name, str(run.adapter_dir))
            text, ids = tuned.caption(example.path, strength)
            st.session_state["t5"] = {
                "key": (run.name, example.image_id), "strength": strength, "text": text, "ids": ids,
            }

        left, right = st.columns([1, 2])
        left.image(example.path)
        right.markdown(f"**Real caption:** {example.caption}")
        stored = st.session_state.get("t5")
        if stored and stored["key"] == (run.name, example.image_id):
            right.markdown(f"**At strength {stored['strength']:.1f}:** {stored['text']}")
        if stored and stored["key"] == (run.name, example.image_id) and not busy:
            tuned = get_tuned_model(run.name, str(run.adapter_dir))
            ids = stored["ids"]
            st.markdown("**How the adapter shifts the next word**")
            position = st.slider("Tokens written so far", 0, max(len(ids) - 1, 1), 0, key="t5_position")
            written = ids[:position]
            so_far = tuned.processor.tokenizer.decode(written, skip_special_tokens=True)
            st.caption(f"Text so far: “{so_far}”")
            with st.spinner("Scoring the next token…"):
                base_table = tuned.next_token_table(example.path, written, 0.0)
                tuned_table = tuned.next_token_table(example.path, written, stored["strength"])
            column_config = {
                "probability": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)
            }
            c1, c2 = st.columns(2)
            c1.markdown("Base model")
            c1.dataframe(pd.DataFrame(base_table, columns=["next token", "probability"]),
                         column_config=column_config, hide_index=True)
            c2.markdown(f"Tuned model at strength {stored['strength']:.1f}")
            c2.dataframe(pd.DataFrame(tuned_table, columns=["next token", "probability"]),
                         column_config=column_config, hide_index=True)
