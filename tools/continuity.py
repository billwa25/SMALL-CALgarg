"""Clip-to-idle continuity of the hands: how far the first / last frame of each clip sits from the idle pose (what pops when the engine switches clips).
python tools/continuity.py"""
import sys, glob, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, omf, skel
from anim_report import qang

ROOT = os.path.join(os.path.dirname(__file__), '..', 'gamedata')
ARM = ['l_clavicle', 'l_upperarm', 'l_forearm', 'l_hand', 'r_clavicle', 'r_upperarm', 'r_forearm', 'r_hand']

for p in sorted(glob.glob(os.path.join(ROOT, 'meshes/anomaly_weapons/hud_hands_animation/*.omf'))):
    o = omf.load(p); names = o.bone_names; par = skel.parents(names)
    idle = [m for m in o.motions if m.name.endswith('_idle')][0]
    Ri, Ti = omf.motion_arrays(idle); Rwi, Pwi = skel.fk(Ri, Ti, par)
    ih = [names.index('r_hand'), names.index('l_hand')]
    print(f"\n=== {os.path.basename(p)}   idle={idle.name}  r_hand@{np.round(Pwi[ih[0],0]*100,1)} l_hand@{np.round(Pwi[ih[1],0]*100,1)} cm")
    print(f"  {'clip':32s} {'len':>4s}  first-frame vs idle: arm-bone deg / hands cm      last-frame vs idle: arm-bone deg / hands cm    (bone with max angle)")
    for m in o.motions:
        R, T = omf.motion_arrays(m); Rw, Pw = skel.fk(R, T, par)
        out = []
        for f in (0, m.length - 1):
            ang = qang(R[:, f], Ri[:, 0]); arm = [names.index(b) for b in ARM]
            amax = ang[arm].max(); bmax = ARM[int(ang[arm].argmax())]
            hcm = np.linalg.norm(Pw[ih, f] - Pwi[ih, 0], axis=-1).max() * 100
            fing = ang[[i for i in range(len(names)) if 'finger' in names[i]]].max()
            out.append(f"{amax:5.1f} / {hcm:5.1f}  (fingers {fing:5.1f}) {bmax:11s}")
        print(f"  {m.name:32s} {m.length:4d}  {out[0]}   {out[1]}")
