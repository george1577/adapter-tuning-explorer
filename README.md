# Adapter Tuning Explorer

An interactive Streamlit app for **seeing** how adapter tuning adapts a pretrained model. It trains a
real LoRA adapter on your laptop, so you can inspect every step instead of just reading about it:
the labels, the weights, the loss, and how the captions change.

It's a companion to the *Adapter Tuning* examples in
[Generative AI Design Patterns](https://github.com/lakshmanok/generative-ai-design-patterns)
(`examples/15_adapter_tuning`), and covers notebooks 1 to 4.

> **New to this?** Read [docs/how-adapter-tuning-works.md](docs/how-adapter-tuning-works.md) first.
> It explains the ideas in plain language with diagrams and real numbers from this app.
> Terms are defined in [GLOSSARY.md](GLOSSARY.md).

> **Learning tool only.** The captions this app produces can invent findings that are not in the
> image. Never use them for medical interpretation.

---

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), about 2 GB of free RAM, and an internet connection
for the first run (to download the model, about 500 MB, and the images, about 7 MB). No API keys
or GPU are needed.

```bash
uv sync                          # installs Python 3.12 and dependencies into .venv
uv run streamlit run app.py      # opens http://localhost:8501
```

The app offers to download the images on first start. You can also do it yourself with
`uv run python scripts/fetch_data.py`.

To follow the guide with the same settings it uses, train the reference run from tab ④, or with
`uv run python scripts/reference_run.py`.

Run the tests:

```bash
uv run pytest -m "not slow"      # quick tests of the arithmetic and data handling
uv run pytest                    # also trains on the real model, about 5 minutes
```

### Configuration

| Setting | How | Default |
|---|---|---|
| Device | `ATE_DEVICE=cuda uv run streamlit run app.py` | `cpu`, or `cuda` if available |

> **Slow training?** On an 8 GB laptop, other open apps can push the machine into swap and make single
> examples take much longer. Close what you can while training.

> **Why not Apple's MPS?** LoRA training crashed on MPS on macOS 14 when this was tested (exit code
> 134), so MPS is opt-in with `ATE_DEVICE=mps`.

---

## The five tabs

| Tab | What you do and see |
|---|---|
| **① Why tune, and the data** | Compare base and real captions, read a training message, and see which tokens count towards the loss |
| **② Inside LoRA** | Move `r` and `α` on one layer written as `W + (α/r)·B·A`, and count parameters on real layers |
| **③ QLoRA and memory** | Simulate NF4 quantization on real weights, and estimate the memory Gemma needs |
| **④ Train an adapter** | Train with live loss, compare with the notebook's recorded Gemma curve, and confirm the base stayed frozen |
| **⑤ Test the adapter** | Compare held-out loss, browse a caption gallery, turn the adapter's strength up and down, and see next-word probabilities shift |

Runs are saved in `runs/`, and the images in `data/`. Neither is committed.

---

## Project layout

```text
app.py                    Streamlit UI
adapter_lab/
  messages.py             Chat messages, copied from the notebooks
  data.py                 The radiology image sample
  labels.py               Which tokens count towards the loss
  lora.py                 The arithmetic of LoRA
  quant.py                Simulated quantization and memory estimates
  recorded.py             Results recorded in the notebooks
  training.py             The training loop and background job
  runs.py                 Saved runs on disk
  vlm.py                  The only module that touches the real model
scripts/                  Download the images, train the reference run
tests/
docs/
  how-adapter-tuning-works.md   The guide
  adr/                          Decisions and why they were made
```

---

## Data and licensing

- **Images and captions:** a sample of the first 100 training and 100 test rows of
  [ROCO, Radiology Objects in COntext](https://labels.tue-image.nl/wp-content/uploads/2018/09/AM-04.pdf)
  (Pelka et al., 2018), downloaded from the Hugging Face mirror
  [`eltorio/ROCO-radiology`](https://huggingface.co/datasets/eltorio/ROCO-radiology). The mirror is
  tagged MIT, but the images come from published medical papers, and that tag could not be confirmed
  as correct. For that reason they are downloaded rather than redistributed here. Check the licensing
  yourself before reusing them.
- **Model:** [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct), Apache-2.0.
- **Recorded results:** the Gemma and GPT-4o-mini results come from the book's notebooks.
