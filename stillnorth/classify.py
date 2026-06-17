"""Zero-shot CLIP image classifier into the 4 content classes A/B/C/D.

Mirrors the project's `tsn_classify.py` (same model, same class prompts) so the
repo is self-contained: it does not import from the ComfyUI output folder. torch
/ open_clip / Pillow are imported lazily, so the server and the rest of the
pipeline run even on a machine where they are not installed -- only the classify
stage needs them.

A = aerial forest canopy / snowy plains (land, no open water)
B = mountain ridges, peaks, drifting clouds/mist
C = rivers, lakes, open liquid water, reflections
D = glaciers, frozen lakes, arctic ice, snow-covered ground
"""
import os

MODEL = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"
BATCH = 16
LOWCONF = 0.55
ORDER = "ABCD"

PROMPTS = {
    "A": [
        "an aerial photo of dense forest canopy filling the whole frame, no water",
        "overhead view of endless evergreen forest treetops and snowy ground",
        "an aerial autumn forest with colorful treetops and no lake or river",
        "a drone view over boreal forest stretching to the horizon, land only",
        "a snow-covered forest landscape of trees seen from above",
    ],
    "B": [
        "an aerial photograph of snowy mountain ridges rising above a layer of clouds",
        "dramatic rocky mountain peaks and ridges partially covered in snow and mist",
        "elevated view of snowy mountain slopes and a high alpine plateau",
        "a mountain valley between tall ridges with clouds drifting through",
        "jagged snow-capped peaks above a sea of clouds",
    ],
    "C": [
        "an aerial photo of a river of open flowing liquid water winding through the land",
        "a lake of dark open unfrozen water reflecting the sky, seen from above",
        "a drone following a river with sparkling liquid water and forested banks",
        "a still mountain lake with mirror-like open water reflections",
        "open blue water surface scattering sunlight, not frozen",
    ],
    "D": [
        "an aerial photo of a frozen lake covered in pale ice and surrounded by forest",
        "a snowy frozen river of solid ice winding through a forest, seen from above",
        "an aerial view of a glacier and arctic ice field",
        "patterns of cracks and crevasses on a blue-white ice surface from above",
        "frozen snow-covered ice in a cold winter landscape, no open water",
    ],
}


def _lazy_imports():
    try:
        import torch  # noqa
        import open_clip  # noqa
        from PIL import Image  # noqa
        return torch, open_clip, Image
    except Exception as e:  # pragma: no cover - depends on host env
        raise RuntimeError(
            "The classify stage needs torch + open_clip + Pillow. Install them "
            "in the same environment as ComfyUI, e.g.:\n"
            "    pip install open_clip_torch pillow\n"
            "(torch ships with your ComfyUI install).\n"
            f"Original import error: {e}"
        )


def classify_all(files, progress=None):
    """Return [(path, class_letter, probs_dict), ...] for each readable image.

    `progress(done, total)` is called after each batch if supplied.
    """
    torch, open_clip, Image = _lazy_imports()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    tok = open_clip.get_tokenizer(MODEL)
    model = model.to(dev).eval()

    with torch.no_grad():
        embs = []
        for k in ORDER:
            t = tok(PROMPTS[k]).to(dev)
            e = model.encode_text(t)
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(0)
            e = e / e.norm()
            embs.append(e)
        T = torch.stack(embs)
    scale = model.logit_scale.exp().item()

    results = []
    total = len(files)
    for i in range(0, total, BATCH):
        batch = files[i:i + BATCH]
        ims, ok = [], []
        for f in batch:
            try:
                ims.append(pre(Image.open(f).convert("RGB"))); ok.append(f)
            except Exception:
                pass
        if ims:
            with torch.no_grad():
                x = torch.stack(ims).to(dev)
                ie = model.encode_image(x)
                ie = ie / ie.norm(dim=-1, keepdim=True)
                probs = (scale * ie @ T.T).softmax(-1).cpu()
            for j, f in enumerate(ok):
                p = {ORDER[k]: float(probs[j, k]) for k in range(4)}
                results.append((f, max(p, key=p.get), p))
        if progress:
            progress(min(i + BATCH, total), total)
    return results
