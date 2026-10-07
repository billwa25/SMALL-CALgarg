# Animation tools

Python 3 + numpy + Pillow, no Blender needed. Run from the repository root.

- `omf.py` / `ogf.py` / `skel.py`: readers and writers for the X-Ray OMF animations and the hud OGF meshes, hand-skeleton forward kinematics.
- `anim_report.py [filter]`: every clip with length, loop flag, per-frame steps and loop seam.
- `continuity.py`: how far each clip's first and last frame sit from the idle pose.
- `render.py <key> <clip>[:frames] [--out x.png]`: offline first-person render (weapon mesh + hand sticks), e.g. `python tools/render.py pt25 hand_pt25_sprint:0,7,15,22 --out sprint.png`.
- `build_sprint.py [--dry DIR]`: rebuilds every pistol's sprint loop and idle<->sprint transitions from the Colt Police Positive's running cycle.
- `fix_pops.py [--dry DIR]`: eases single-frame arm jumps in the hands clips.

The hud weapon is attached to the hands' `lead_gun` bone; the item clips' own root key is ignored by the engine.

## Sounds

- `oggx.py`: Ogg Vorbis page tools: read and write the X-Ray sound header (min and max distance, volume, type, AI distance), decode and encode through ffmpeg.
- `build_sounds.py [--out DIR]`: layered gunshots per weapon from the pack's own recordings (declipped, rebalanced, with synthesised body, crack, action and reverb layers; three takes per layer; mono layers for NPCs, stereo for the player), the matching `[smallcal_<w>_snd_shoot]` / `_actor` sections, and relevelled handling sounds. Needs ffmpeg with libvorbis.
