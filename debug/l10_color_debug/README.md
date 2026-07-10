# L10 Color Debug Tools

This directory contains the scripts and generated previews used to tune the
right Linker Hand L10 colors in `dual_nero_linker_l10_combined.urdf`.

Apply the configured materials to the URDF:

```bash
conda activate newton
python debug/l10_color_debug/apply_l10_photo_colors.py
```

Regenerate the color previews in `previews/`:

```bash
conda activate newton
python debug/l10_color_debug/render_l10_urdf_colors.py
```
