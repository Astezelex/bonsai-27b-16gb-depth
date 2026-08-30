#!/usr/bin/env python3
"""Dump GGUF metadata keys and scalar values. Minimal reader, no gguf-py dependency.

Purpose: decide whether a kv-mean-center bias file records the -ctk it was calibrated with.
If it does, the loader could reject a mismatch and does not. If it does not, the information
is simply absent and the maintainer's claim cannot be implemented as stated. Those are very
different bug reports.
"""
import struct, sys

T = {0:"u8",1:"i8",2:"u16",3:"i16",4:"u32",5:"i32",6:"f32",7:"bool",8:"str",9:"arr",
     10:"u64",11:"i64",12:"f64"}
FMT = {0:("<B",1),1:("<b",1),2:("<H",2),3:("<h",2),4:("<I",4),5:("<i",4),6:("<f",4),
       7:("<B",1),10:("<Q",8),11:("<q",8),12:("<d",8)}


def rd_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def rd_val(f, t):
    if t == 8:
        return rd_str(f)
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        vals = [rd_val(f, et) for _ in range(min(n, 8))]
        if n > 8:
            for _ in range(n - 8):
                rd_val(f, et)
            vals.append(f"... ({n} total)")
        return vals
    fmt, sz = FMT[t]
    return struct.unpack(fmt, f.read(sz))[0]


for path in sys.argv[1:]:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            print(f"{path}: not a GGUF file (magic {magic!r})"); continue
        ver, ntensor, nkv = struct.unpack("<IQQ", f.read(20))
        print(f"\n=== {path}\n    version {ver}, {ntensor} tensors, {nkv} metadata keys")
        for _ in range(nkv):
            k = rd_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            v = rd_val(f, t)
            s = str(v)
            if len(s) > 110:
                s = s[:107] + "..."
            print(f"    {k:44} {T.get(t,t):4} = {s}")
