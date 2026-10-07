"""X-Ray OMF (skeletal motion) reader / writer.

OMF = chunked file: S_MOTIONS (sub-chunk 0 = count, sub-chunk i+1 = motion i) + S_SMPARAMS (partitions = bone list, motion defs).
Key formats: rotation = 4 x int16 quaternion (/32767), translation = 3 x int8 or int16 keys scaled by size + init, or one float3 if static.
"""
import struct, zlib
import numpy as np

FL_T_PRESENT = 1
FL_R_ABSENT = 2
FL_T_16BIT = 4
Q = 32767.0


class Reader:
    def __init__(self, data, pos=0):
        self.d = data
        self.p = pos

    def u8(self):
        v = self.d[self.p]; self.p += 1; return v

    def u16(self):
        v = struct.unpack_from('<H', self.d, self.p)[0]; self.p += 2; return v

    def u32(self):
        v = struct.unpack_from('<I', self.d, self.p)[0]; self.p += 4; return v

    def f32(self):
        v = struct.unpack_from('<f', self.d, self.p)[0]; self.p += 4; return v

    def f3(self):
        v = struct.unpack_from('<3f', self.d, self.p); self.p += 12; return v

    def sz(self):
        e = self.d.index(b'\0', self.p)
        s = self.d[self.p:e].decode('cp1251')
        self.p = e + 1
        return s

    def line(self):
        """-> (text, terminator) of a w_string line (the SDK writes \\r\\n, some tools \\n)"""
        e = self.d.index(b'\n', self.p)
        raw = self.d[self.p:e]
        term = b'\r\n' if raw.endswith(b'\r') else b'\n'
        self.p = e + 1
        return raw.rstrip(b'\r').decode('cp1251'), term

    def raw(self, n):
        v = self.d[self.p:self.p + n]; self.p += n; return v

    def eof(self):
        return self.p >= len(self.d)


def chunks(data):
    out = []
    r = Reader(data)
    while not r.eof():
        cid = r.u32() & 0x7FFFFFFF
        size = r.u32()
        out.append((cid, r.raw(size)))
    return out


def pack_chunk(cid, payload):
    return struct.pack('<II', cid, len(payload)) + payload


class BoneTrack:
    """One bone of one motion.  rot: (n,4) int16 (x,y,z,w) or (1,4) if constant.  trans: (n,3) int keys + size/init, or init only."""
    __slots__ = ('flags', 'rot', 'rot_crc', 'tkeys', 'tsize', 'tinit', 'trans_crc')

    def rot_f(self):
        return self.rot.astype(np.float64) / Q

    def trans_f(self, n):
        if self.flags & FL_T_PRESENT:
            return self.tkeys.astype(np.float64) * np.asarray(self.tsize, np.float64) + np.asarray(self.tinit, np.float64)
        return np.tile(np.asarray(self.tinit, np.float64), (n, 1))


class Motion:
    def __init__(self, name, length, bones):
        self.name = name
        self.length = length
        self.bones = bones          # list[BoneTrack], index = bone index of the partition order


class MotionDef:
    __slots__ = ('name', 'flags', 'bone_or_part', 'motion', 'speed', 'power', 'accrue', 'falloff', 'marks')


class OMF:
    def __init__(self):
        self.params_raw = b''
        self.version = 0
        self.partitions = []        # [(name, [(bone_name, bone_index), ...])]
        self.defs = []
        self.motions = []
        self.ids = (0, 0)           # (motions chunk id, params chunk id)

    @property
    def bone_names(self):
        names = {}
        for _, bl in self.partitions:
            for n, i in bl:
                names[i] = n
        return [names[i] for i in range(len(names))]

    def motion(self, name):
        for m in self.motions:
            if m.name == name:
                return m
        raise KeyError(name)


def _parse_params(raw):
    r = Reader(raw)
    o = {}
    ver = r.u16()
    parts = []
    for _ in range(r.u16()):
        pname = r.sz()
        bl = []
        for _ in range(r.u16()):
            if ver == 1:
                bl.append((r.sz(), len(bl)))
            elif ver == 2:
                bl.append(('', r.u32()))
            else:
                bl.append((r.sz(), r.u32()))
        parts.append((pname, bl))
    defs = []
    for _ in range(r.u16()):
        d = MotionDef()
        d.name = r.sz()
        d.flags = r.u32()
        d.bone_or_part = r.u16()
        d.motion = r.u16()
        d.speed = r.f32(); d.power = r.f32(); d.accrue = r.f32(); d.falloff = r.f32()
        d.marks = []
        if ver >= 4:
            for _ in range(r.u32()):
                mname, term = r.line()
                iv = [(r.f32(), r.f32()) for _ in range(r.u32())]
                d.marks.append((mname, iv, term))
        defs.append(d)
    assert r.eof(), ('params trailing bytes', len(raw) - r.p)
    return ver, parts, defs


def _parse_motion(raw, nbones):
    r = Reader(raw)
    name = r.sz()
    n = r.u32()
    bones = []
    for _ in range(nbones):
        b = BoneTrack()
        b.flags = r.u8()
        if b.flags & FL_R_ABSENT:
            b.rot = np.frombuffer(r.raw(8), dtype='<i2').reshape(1, 4).copy()
            b.rot_crc = None
        else:
            b.rot_crc = r.u32()
            b.rot = np.frombuffer(r.raw(8 * n), dtype='<i2').reshape(n, 4).copy()
        if b.flags & FL_T_PRESENT:
            b.trans_crc = r.u32()
            if b.flags & FL_T_16BIT:
                b.tkeys = np.frombuffer(r.raw(6 * n), dtype='<i2').reshape(n, 3).copy()
            else:
                b.tkeys = np.frombuffer(r.raw(3 * n), dtype='<i1').reshape(n, 3).copy()
            b.tsize = r.f3()
            b.tinit = r.f3()
        else:
            b.trans_crc = None
            b.tkeys = None
            b.tsize = None
            b.tinit = r.f3()
        bones.append(b)
    assert r.eof(), ('motion trailing bytes', name, len(raw) - r.p)
    return Motion(name, n, bones)


def load(path):
    data = open(path, 'rb').read()
    o = OMF()
    top = chunks(data)
    params = None
    motions_raw = None
    for cid, raw in top:
        if len(raw) >= 2 and struct.unpack_from('<H', raw)[0] in (1, 2, 3, 4) and params is None and cid != 0x1A and cid != 0xE:
            params = (cid, raw)
        else:
            motions_raw = (cid, raw)
    assert params and motions_raw, [c for c, _ in top]
    o.params_raw = params[1]
    o.version, o.partitions, o.defs = _parse_params(params[1])
    nb = len(o.bone_names)
    sub = chunks(motions_raw[1])
    assert sub[0][0] == 0
    count = struct.unpack('<I', sub[0][1])[0]
    assert len(sub) == count + 1, (len(sub), count)
    for cid, raw in sub[1:]:
        o.motions.append(_parse_motion(raw, nb))
    o.ids = (motions_raw[0], params[0])
    o.top_order = [c for c, _ in top]
    return o


def _pack_motion(m):
    out = bytearray()
    out += m.name.encode('cp1251') + b'\0'
    out += struct.pack('<I', m.length)
    for b in m.bones:
        out.append(b.flags)
        if b.flags & FL_R_ABSENT:
            out += b.rot.astype('<i2').tobytes()
        else:
            kb = b.rot.astype('<i2').tobytes()
            out += struct.pack('<I', zlib.crc32(kb) & 0xFFFFFFFF)
            out += kb
        if b.flags & FL_T_PRESENT:
            kb = b.tkeys.astype('<i2' if b.flags & FL_T_16BIT else '<i1').tobytes()
            out += struct.pack('<I', zlib.crc32(kb) & 0xFFFFFFFF)
            out += kb
            out += struct.pack('<3f', *b.tsize)
            out += struct.pack('<3f', *b.tinit)
        else:
            out += struct.pack('<3f', *b.tinit)
    return bytes(out)


def pack_params(o):
    """serialise partitions + motion defs (version 3 / 4 layout)"""
    out = bytearray()
    out += struct.pack('<H', o.version)
    out += struct.pack('<H', len(o.partitions))
    for pname, bl in o.partitions:
        out += pname.encode('cp1251') + b'\0'
        out += struct.pack('<H', len(bl))
        for bname, bidx in bl:
            if o.version == 1:
                out += bname.encode('cp1251') + b'\0'
            elif o.version == 2:
                out += struct.pack('<I', bidx)
            else:
                out += bname.encode('cp1251') + b'\0' + struct.pack('<I', bidx)
    out += struct.pack('<H', len(o.defs))
    for d in o.defs:
        out += d.name.encode('cp1251') + b'\0'
        out += struct.pack('<IHHffff', d.flags, d.bone_or_part, d.motion, d.speed, d.power, d.accrue, d.falloff)
        if o.version >= 4:
            out += struct.pack('<I', len(d.marks))
            for mname, iv, term in d.marks:
                out += mname.encode('cp1251') + term + struct.pack('<I', len(iv))
                for a, b in iv:
                    out += struct.pack('<ff', a, b)
    return bytes(out)


def add_motion(o, name, template, R, T, flags=None, hq=True):
    """append a motion built from (nb,n,4) rotations and (nb,n,3) translations; the def copies `template`'s def (speed, blend) unless flags given"""
    import copy
    tm = o.motion(template)
    m = Motion(name, R.shape[1], [BoneTrack() for _ in tm.bones])
    for b in m.bones:
        b.flags = 0; b.rot_crc = None; b.trans_crc = None; b.tkeys = None; b.tsize = None; b.tinit = (0.0, 0.0, 0.0); b.rot = np.zeros((1, 4), np.int16)
    set_motion_arrays(m, R, T, hq=hq)
    o.motions.append(m)
    td = next(d for d in o.defs if d.name == template)
    d = MotionDef(); d.name = name; d.flags = td.flags if flags is None else flags
    d.bone_or_part = td.bone_or_part; d.motion = len(o.motions) - 1
    d.speed, d.power, d.accrue, d.falloff = td.speed, td.power, td.accrue, td.falloff
    d.marks = []
    o.defs.append(d)
    return m, d


def save(o, path):
    o.params_raw = pack_params(o)
    sub = pack_chunk(0, struct.pack('<I', len(o.motions)))
    for i, m in enumerate(o.motions):
        sub += pack_chunk(i + 1, _pack_motion(m))
    parts = {o.ids[0]: pack_chunk(o.ids[0], sub), o.ids[1]: pack_chunk(o.ids[1], o.params_raw)}
    data = b''.join(parts[c] for c in o.top_order)
    open(path, 'wb').write(data)


# ---- float <-> quantized -------------------------------------------------------------------------------------------

def quat_continuous(q):
    """flip signs so consecutive quaternions sit in the same hemisphere"""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def set_rot_f(b, qf):
    """qf (n,4) float -> int16 keys; collapses to a constant key if every frame is the same"""
    qf = qf / np.linalg.norm(qf, axis=1, keepdims=True)
    k = np.clip(np.rint(qf * Q), -32767, 32767).astype(np.int16)
    if len(k) > 1 and np.all(k == k[0]):
        b.flags |= FL_R_ABSENT
        b.rot = k[:1].copy()
    else:
        b.flags &= ~FL_R_ABSENT
        b.rot = k


def set_trans_f(b, tf, hq=None):
    """tf (n,3) float -> keys; constant tracks become a single init vector"""
    tf = np.asarray(tf, np.float64)
    lo, hi = tf.min(axis=0), tf.max(axis=0)
    if np.all(hi - lo < 1e-7):
        b.flags &= ~(FL_T_PRESENT | FL_T_16BIT)
        b.tkeys = None; b.tsize = None
        b.tinit = tuple(float(x) for x in tf[0])
        return
    if hq is None:
        hq = bool(b.flags & FL_T_16BIT)
    qmax = 32767.0 if hq else 127.0
    c = (hi + lo) / 2
    half = (hi - lo) / 2
    size = np.where(half > 0, half / qmax, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        keys = np.where(size > 0, (tf - c) / np.where(size > 0, size, 1.0), 0.0)
    keys = np.clip(np.rint(keys), -qmax, qmax)
    b.flags |= FL_T_PRESENT
    if hq:
        b.flags |= FL_T_16BIT
        b.tkeys = keys.astype(np.int16)
    else:
        b.flags &= ~FL_T_16BIT
        b.tkeys = keys.astype(np.int8)
    b.tsize = tuple(float(x) for x in size)
    b.tinit = tuple(float(x) for x in c)


def motion_arrays(m):
    """-> rot (nb, n, 4) float continuous, trans (nb, n, 3) float"""
    n = m.length
    R = np.zeros((len(m.bones), n, 4)); T = np.zeros((len(m.bones), n, 3))
    for i, b in enumerate(m.bones):
        q = b.rot_f()
        if len(q) == 1:
            q = np.tile(q, (n, 1))
        R[i] = quat_continuous(q)
        T[i] = b.trans_f(n)
    return R, T


def set_motion_arrays(m, R, T, hq=None):
    n = R.shape[1]
    m.length = n
    for i, b in enumerate(m.bones):
        set_rot_f(b, R[i])
        set_trans_f(b, T[i], hq=hq)
