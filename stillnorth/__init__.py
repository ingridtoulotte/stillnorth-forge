"""StillNorth Forge -- staged-batch local pipeline:
HTML prompts -> FLUX.2 images -> x2 upscale -> CLIP classify -> Wan 2.2 clips
-> last frame -> x4 upscale -> continuation clips -> concat -> x4 final upscale.
"""
__version__ = "0.2.0"
