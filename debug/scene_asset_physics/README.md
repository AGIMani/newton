# Scene and Asset Physics Tools

This directory contains interactive tools for editing or inspecting scene and
asset collision geometry, physical properties, mass, and center of mass.

The generated runtime specifications remain in the `debug` directory so the
existing VR launch scripts continue to load them without configuration changes:

- `debug/dynamic_bottle_body.json`
- `debug/scene_collision_boxes.json`

Run the tools from the repository root in the existing Newton conda environment:

```bash
conda activate newton
python debug/scene_asset_physics/edit_dynamic_bottle_body.py
python debug/scene_asset_physics/edit_scene_collision_box.py
python debug/scene_asset_physics/inspect_dynamic_bottle_body.py --viewer gl
```
