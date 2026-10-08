# Animation tools

Python 3 + numpy + Pillow, no Blender needed. Run from the repository root.

- `omf.py` / `ogf.py` / `skel.py`: readers and writers for the X-Ray OMF animations and the hud OGF meshes, hand-skeleton forward kinematics.
- `anim_report.py [filter]`: every clip with length, loop flag, per-frame steps and loop seam.
- `continuity.py`: how far each clip's first and last frame sit from the idle pose.
- `render.py <key> <clip>[:frames] [--out x.png]`: offline first-person render (weapon mesh + hand sticks), e.g. `python tools/render.py pt25 hand_pt25_sprint:0,7,15,22 --out sprint.png`.
- `build_sprint.py [--dry DIR]`: rebuilds every pistol's sprint loop and idle<->sprint transitions from the Colt Police Positive's running cycle.
- `fix_pops.py [--dry DIR]`: eases single-frame arm jumps in the hands clips.
- `fix_elbows.py [--dry DIR] [--keys a,b]`: left-elbow correction (shoulder kept behind the lens, two-bone IK, natural elbow pole) and easing of hand jumps in the hands clips.
- `fix_loader.py [--keys cpp38,9galo22]`: revolver reloads, hides the speedloader grab below the frame.
- `fix_cylinder.py [--keys cpp38,9galo22] [--dry DIR] [--sheet DIR]`: revolver shots, turns the cylinder one chamber (Colt clockwise, 9 Galo counter-clockwise) over the frames after the hammer drops, ending on the idle pose; `--sheet` writes zoomed contact sheets.

The hud weapon is attached to the hands' `lead_gun` bone; the item clips' own root key is ignored by the engine.

## Sounds

- `oggx.py`: Ogg Vorbis page tools: read and write the X-Ray sound header (min and max distance, volume, type, AI distance), decode and encode through ffmpeg.
- `pocketpops.py`: cleans the pocketpops recordings (pooled noise profiles, spectral gate, brass hits cut, hall wash tamed).
- `mix_ds.py --wire [--keys a,b] [--ds-db -4.5]`: the live gunshot build: cleaned recording on top, a Dark Signal shot aligned underneath, Dark Signal distance layers and echoes as tails, writes the layered sound sections.
- `reference.py` + `reference_target.json`: measurements of the reference packs (onset, band levels, crest, a matcher the live build no longer uses); the mixer borrows its onset and pop helpers.
- `build_sounds.py [--out DIR]`: layered gunshots per weapon from the pack's own recordings (declipped, rebalanced, with synthesised body, crack, action and reverb layers; three takes per layer; mono layers for NPCs, stereo for the player), the matching `[smallcal_<w>_snd_shoot]` / `_actor` sections, and relevelled handling sounds. Needs ffmpeg with libvorbis.
