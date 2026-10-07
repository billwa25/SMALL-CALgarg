"""X-Ray OGF v4 skinned model reader: bones (names, parents, bind pose) + mesh children (positions, normals, bone links, triangles)."""
import struct
import numpy as np
from omf import Reader, chunks


class Bone:
    __slots__ = ('name', 'parent', 'rot', 'offset')


class Child:
    __slots__ = ('texture', 'tris', 'pos', 'nrm', 'bones', 'weights')


class OGF:
    def __init__(self):
        self.bones = []
        self.children = []
        self.motion_refs = []

    def parent_index(self):
        idx = {b.name.lower(): i for i, b in enumerate(self.bones)}
        return [idx.get(b.parent.lower(), -1) if b.parent else -1 for b in self.bones]


def _euler_zxy_neg(r):
    """bind rotation as blender-xray reads it: Euler(-rot, 'ZXY') -> matrix on column vectors"""
    x, y, z = -r[0], -r[1], -r[2]
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def bind_world(o, euler=_euler_zxy_neg):
    """-> (nb,3,3) rotations, (nb,3) positions of the bind pose in model space"""
    par = o.parent_index()
    nb = len(o.bones)
    Rw = np.zeros((nb, 3, 3)); Pw = np.zeros((nb, 3))
    for i, b in enumerate(o.bones):
        Rl = euler(b.rot); Tl = np.asarray(b.offset)
        p = par[i]
        if p < 0:
            Rw[i] = Rl; Pw[i] = Tl
        else:
            Rw[i] = Rw[p] @ Rl; Pw[i] = Pw[p] + Rw[p] @ Tl
    return Rw, Pw


def _read_vertices(raw):
    r = Reader(raw)
    fmt = r.u32(); n = r.u32()
    links = {0x12071980: 1, 1: 1, 0x240E3300: 2, 2: 2, 0x36154C80: 3, 3: 3, 0x481C6D00: 4, 4: 4}[fmt]
    pos = np.zeros((n, 3)); nrm = np.zeros((n, 3)); bones = np.zeros((n, 4), np.int32); w = np.zeros((n, 4))
    for i in range(n):
        if links == 1:
            v = struct.unpack_from('<14fI', raw, r.p); r.p += 60
            pos[i] = v[0:3]; nrm[i] = v[3:6]; bones[i, 0] = v[14]; w[i, 0] = 1
        else:
            b = struct.unpack_from('<%dH' % links, raw, r.p); r.p += 2 * links
            v = struct.unpack_from('<12f', raw, r.p); r.p += 48
            ws = struct.unpack_from('<%df' % (links - 1), raw, r.p); r.p += 4 * (links - 1)
            r.p += 8   # uv
            pos[i] = v[0:3]; nrm[i] = v[3:6]
            bones[i, :links] = b
            if links == 2:
                w[i, 0] = 1 - ws[0]; w[i, 1] = ws[0]
            else:
                w[i, :links - 1] = ws; w[i, links - 1] = 1 - sum(ws)
    assert r.eof(), ('vertices trailing', len(raw) - r.p)
    return pos, nrm, bones, w


def _read_child(raw):
    c = Child(); c.texture = ''
    sw = None; idx = None
    for cid, data in chunks(raw):
        if cid == 0x2:
            r = Reader(data); c.texture = r.sz()
        elif cid == 0x3:
            c.pos, c.nrm, c.bones, c.weights = _read_vertices(data)
        elif cid == 0x4:
            n = struct.unpack_from('<I', data)[0]
            idx = np.frombuffer(data[4:4 + 2 * n], dtype='<u2').astype(np.int32)
        elif cid == 0x6:
            r = Reader(data); r.p += 16; cnt = r.u32()
            sw = [(r.u32(), r.u16(), r.u16()) for _ in range(cnt)]
    if sw:
        off, ntris, nverts = sw[0]
        idx = idx[off:off + ntris * 3]
    c.tris = idx.reshape(-1, 3)
    return c


def load(path):
    data = open(path, 'rb').read()
    o = OGF()
    for cid, raw in chunks(data):
        if cid == 0x9:
            for _, craw in chunks(raw):
                o.children.append(_read_child(craw))
        elif cid == 0xD:
            r = Reader(raw)
            for _ in range(r.u32()):
                b = Bone(); b.name = r.sz(); b.parent = r.sz(); r.p += 15 * 4
                o.bones.append(b)
        elif cid == 0x10:
            r = Reader(raw)
            for b in o.bones:
                r.u32(); r.sz(); r.u16(); r.u16(); r.p += (15 + 4 + 8) * 4
                r.u32(); r.p += 12 * 4 + 2 * 4; r.u32(); r.p += 3 * 4
                b.rot = r.f3(); b.offset = r.f3(); r.f32(); r.f3()
            assert r.eof(), ('ikdata trailing', len(raw) - r.p)
        elif cid == 0x18:
            r = Reader(raw); o.motion_refs = [r.sz() for _ in range(r.u32())]
        elif cid == 0x13:
            o.motion_refs = raw.rstrip(b'\0').decode('cp1251').split(',')
    return o
