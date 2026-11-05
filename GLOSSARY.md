# Adapter Tuning Explorer

A learning tool for seeing how a pretrained model is adapted to a new task by training only a small part of it.

## Language

### Techniques

**Adapter tuning**:
The design pattern of adapting a pretrained model to a task with a small dataset, by training only a small part of the model.
_Avoid_: LoRA (one way of doing it), fine-tuning (also covers retraining the whole model)

**LoRA**:
A technique for adapter tuning that adds small low-rank matrices to frozen layers of the model.
_Avoid_: Adapter tuning (the pattern, not this technique)

**QLoRA**:
LoRA applied to a base model whose weights have been quantized to save memory.
_Avoid_: LoRA (which does not imply quantization)

### Artifacts

**Base model**:
The pretrained model being adapted, whose original weights stay frozen.
_Avoid_: Foundational model (reserved for the hosted model used as a comparison)

**Adapter**:
The small set of weights produced by adapter tuning, which is loaded on top of a base model.
_Avoid_: Tuned model (that is the base model and adapter together)

**Tuned model**:
A base model with an adapter loaded on top.
_Avoid_: Adapter (only the small trained weights)

**Foundational model**:
The hosted general-purpose model used as a comparison for captioning, GPT-4o-mini in the reference notebook.
_Avoid_: Base model (the model being adapted)

### Evaluation

**Training loss**:
The loss on the examples being trained on, which in the reference notebook also counts the prompt tokens.
_Avoid_: Quality, accuracy

**Held-out loss**:
The loss on images never used in training, counting caption tokens only.
_Avoid_: Test accuracy, validation score

**Reference run**:
The training run with fixed settings and seed that the guide's numbers come from.
_Avoid_: Default run, baseline
