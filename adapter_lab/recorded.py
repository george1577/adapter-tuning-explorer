"""Results recorded in the reference notebooks. Nothing here is computed by this app."""

from adapter_lab.quant import ModelCounts

# 3_adapter_tuning.ipynb: (optimizer step, training loss), logged every 5 steps of 125.
GEMMA_LOSS = [
    (5, 14.7725), (10, 7.3997), (15, 6.1076), (20, 4.7924), (25, 5.5631),
    (30, 4.6355), (35, 4.6211), (40, 3.9563), (45, 4.8736), (50, 4.6371),
    (55, 4.5041), (60, 4.6020), (65, 4.8559), (70, 4.8942), (75, 4.3278),
    (80, 4.1730), (85, 5.0174), (90, 4.3223), (95, 4.0581), (100, 4.1150),
    (105, 5.2801), (110, 3.9692), (115, 4.7790), (120, 4.0694), (125, 4.1324),
]
GEMMA_TRAINING_TIME = "42 min 23 s on an NVIDIA L4 GPU with 32 GB of memory"

# 2_foundational.ipynb, for image ROCO_00003.
GPT_CAPTION = (
    "Radiographic image displaying the periapical region of tooth roots. The image highlights the "
    "presence of apical radiolucencies, indicative of potential periapical pathology. The distinct "
    "radiographic characteristics suggest chronic apical periodontitis, warranting further clinical "
    "evaluation and management."
)

# 4_test_model.ipynb, for image ROCO_00006.
GEMMA_TUNED_CAPTION = "CT scan of the abdomen showing the size and density of the intra-abdominal mass."

# Counted from Gemma 3 4B's configuration on an empty (meta) model, without downloading any weights.
GEMMA_COUNTS = ModelCounts(
    total=4_300_079_472,
    quantizable=3_619_713_024,  # linear layers except lm_head
    lora_at_r16=38_497_792,  # LoRA on all linear layers, r=16
    modules_to_save=1_342_504_960,  # one trainable copy each of embed_tokens and lm_head
)
