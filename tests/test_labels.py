from adapter_lab.labels import IGNORE, build_labels, region_of

IMAGE = {50, 51}
PAD = 0
# prompt: 1 2 [image 50 51] 3 | caption: 7 8, then padding
IDS = [1, 2, 50, 51, 3, 7, 8, 0]


def test_notebook_labels_keep_the_prompt_and_drop_image_and_padding():
    assert build_labels(IDS, IMAGE, PAD, prompt_length=5) == [1, 2, IGNORE, IGNORE, 3, 7, 8, IGNORE]


def test_masking_the_prompt_leaves_only_the_caption():
    labels = build_labels(IDS, IMAGE, PAD, prompt_length=5, mask_prompt=True)
    assert labels == [IGNORE] * 5 + [7, 8, IGNORE]


def test_regions():
    regions = [region_of(i, t, IMAGE, PAD, 5) for i, t in enumerate(IDS)]
    assert regions == ["prompt", "prompt", "image", "image", "prompt", "caption", "caption", "padding"]
