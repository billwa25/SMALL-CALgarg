"""Ogg Vorbis page/packet tools: read the X-Ray sound comment, and write one (version 3: min dist, max dist, volume, game type, AI dist)
as the first user comment of an existing .ogg, re-paginating the header pages.  Also decode/encode through ffmpeg."""
import struct, subprocess, os
import numpy as np

SR = 44100


def crc32_ogg(data):
    crc = 0
    for b in data:
        crc ^= b << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) if crc & 0x80000000 else (crc << 1)
            crc &= 0xFFFFFFFF
    return crc


_TABLE = None


def _crc_table():
    global _TABLE
    if _TABLE is None:
        t = []
        for i in range(256):
            r = i << 24
            for _ in range(8):
                r = ((r << 1) ^ 0x04C11DB7) if r & 0x80000000 else (r << 1)
                r &= 0xFFFFFFFF
            t.append(r)
        _TABLE = t
    return _TABLE


def ogg_crc(data):
    t = _crc_table(); crc = 0
    for b in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ t[((crc >> 24) & 0xFF) ^ b]
    return crc


def read_pages(data):
    """-> list of dict(header fields, segments(list of bytes), raw)"""
    pages = []; p = 0
    while p < len(data):
        assert data[p:p + 4] == b'OggS', ('bad page at', p)
        ver, htype = data[p + 4], data[p + 5]
        granule, serial, seq, crc, nseg = struct.unpack_from('<qIIIB', data, p + 6)
        lacing = data[p + 27:p + 27 + nseg]
        q = p + 27 + nseg
        segs = []
        for l in lacing:
            segs.append(data[q:q + l]); q += l
        pages.append(dict(htype=htype, granule=granule, serial=serial, seq=seq, lacing=list(lacing), segs=segs))
        p = q
    return pages


def packets_from_pages(pages):
    """-> list of (packet bytes, index of the page the packet ends on)"""
    out = []; cur = b''
    for pi, pg in enumerate(pages):
        for l, s in zip(pg['lacing'], pg['segs']):
            cur += s
            if l < 255:
                out.append((cur, pi)); cur = b''
    if cur:
        out.append((cur, len(pages) - 1))
    return out


def build_page(packets, htype, granule, serial, seq):
    """one page holding whole packets (each must be < 255*255 bytes)"""
    lacing = []; body = b''
    for pk in packets:
        n = len(pk)
        while n >= 255:
            lacing.append(255); n -= 255
        lacing.append(n)
        body += pk
    assert len(lacing) <= 255
    hdr = b'OggS' + bytes([0, htype]) + struct.pack('<qIIIB', granule, serial, seq, 0, len(lacing)) + bytes(lacing)
    page = bytearray(hdr + body)
    struct.pack_into('<I', page, 22, ogg_crc(bytes(page)))
    return bytes(page)


def xray_blob(min_dist, max_dist, volume, game_type, ai_dist):
    return struct.pack('<IfffIf', 3, min_dist, max_dist, volume, game_type, ai_dist)


def parse_xray(blob):
    if len(blob) < 4:
        return None
    ver = struct.unpack_from('<I', blob)[0]
    if ver == 3 and len(blob) >= 24:
        _, mn, mx, vol, gt, ai = struct.unpack_from('<IfffIf', blob)
        return dict(version=3, min=mn, max=mx, volume=vol, type=gt, ai=ai)
    if ver == 2 and len(blob) >= 20:
        _, mn, mx, gt, ai = struct.unpack_from('<IffIf', blob)
        return dict(version=2, min=mn, max=mx, type=gt, ai=ai)
    if ver == 1 and len(blob) >= 16:
        _, mn, mx, gt = struct.unpack_from('<IffI', blob)
        return dict(version=1, min=mn, max=mx, type=gt)
    return dict(version=ver)


def read_comment(path):
    """-> (vendor, [user comment bytes...]) of the Vorbis comment header"""
    pages = read_pages(open(path, 'rb').read())
    pk = packets_from_pages(pages)
    com = pk[1][0]
    assert com[:7] == b'\x03vorbis', com[:7]
    p = 7
    vl = struct.unpack_from('<I', com, p)[0]; p += 4; vendor = com[p:p + vl]; p += vl
    n = struct.unpack_from('<I', com, p)[0]; p += 4
    ucs = []
    for _ in range(n):
        l = struct.unpack_from('<I', com, p)[0]; p += 4; ucs.append(com[p:p + l]); p += l
    return vendor, ucs


def info(path):
    vendor, ucs = read_comment(path)
    x = parse_xray(ucs[0]) if ucs else None
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=sample_rate,channels,duration', '-of', 'csv=p=0', path], capture_output=True, text=True)
    return dict(xray=x, comments=len(ucs), probe=r.stdout.strip())


def write_comment(path, blob, out_path=None, vendor=b'Xiph.Org libVorbis'):
    data = open(path, 'rb').read()
    pages = read_pages(data)
    pk = packets_from_pages(pages)
    ident, com, setup = pk[0][0], pk[1][0], pk[2][0]
    assert ident[:7] == b'\x01vorbis' and com[:7] == b'\x03vorbis' and setup[:7] == b'\x05vorbis'
    last_hdr_page = pk[2][1]
    serial = pages[0]['serial']
    newcom = b'\x03vorbis' + struct.pack('<I', len(vendor)) + vendor + struct.pack('<I', 1) + struct.pack('<I', len(blob)) + blob + b'\x01'
    out = build_page([ident], 0x02, 0, serial, 0)
    out += build_page([newcom, setup], 0x00, 0, serial, 1)
    seq = 2
    for pg in pages[last_hdr_page + 1:]:
        body = b''.join(pg['segs'])
        hdr = b'OggS' + bytes([0, pg['htype']]) + struct.pack('<qIIIB', pg['granule'], serial, seq, 0, len(pg['lacing'])) + bytes(pg['lacing'])
        page = bytearray(hdr + body); struct.pack_into('<I', page, 22, ogg_crc(bytes(page)))
        out += bytes(page); seq += 1
    open(out_path or path, 'wb').write(out)


def decode(path, channels=None):
    """-> float32 array (n,) mono or (n, ch) at 44.1 kHz"""
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=channels', '-of', 'csv=p=0', path], capture_output=True, text=True)
    ch = int(r.stdout.strip().split(',')[0]) if channels is None else channels
    r = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'f32le', '-ac', str(ch), '-ar', str(SR), '-'], capture_output=True)
    x = np.frombuffer(r.stdout, np.float32)
    return x if ch == 1 else x.reshape(-1, ch)


def encode(x, path, quality=8):
    """float array (n,) or (n, ch) -> .ogg (libvorbis), 44.1 kHz"""
    x = np.asarray(x, np.float32)
    ch = 1 if x.ndim == 1 else x.shape[1]
    r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', str(ch), '-i', '-', '-c:a', 'libvorbis', '-q:a', str(quality), path], input=x.tobytes(), capture_output=True)
    assert r.returncode == 0, r.stderr.decode()[-500:]


def write_ogg(x, path, cm, quality=8):
    encode(x, path, quality)
    write_comment(path, xray_blob(*cm))
    return info(path)
