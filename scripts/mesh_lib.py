"""Minimal reader for the Abaqus input file of case C6 (nodes, C3D20 elements, element/node sets)."""
import re

import numpy as np


def read_lines(path):
    return open(path, encoding="utf-8", errors="ignore", newline="").read().replace("\r\n", "\n").split("\n")


def parse(path):
    lines = read_lines(path)
    nodes, elems = {}, {}
    sets = {}                      # name -> ("elset"|"nset", list of ids)
    i = 0
    n = len(lines)
    cur = None
    while i < n:
        l = lines[i]
        low = l.lower()
        if low.startswith("*node") and "output" not in low and "print" not in low:
            i += 1
            while i < n and (not lines[i].startswith("*") or lines[i].startswith("**")):
                if lines[i].strip() and not lines[i].startswith("**"):
                    p = [s.strip() for s in lines[i].split(",")]
                    nodes[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
                i += 1
            continue
        if low.startswith("*element") and "output" not in low:
            i += 1
            buf = []
            while i < n and (not lines[i].startswith("*") or lines[i].startswith("**")):
                if lines[i].strip() and not lines[i].startswith("**"):
                    buf += [int(s) for s in lines[i].replace("\n", "").split(",") if s.strip()]
                    if len(buf) >= 21:
                        elems[buf[0]] = buf[1:21]
                        buf = []
                i += 1
            continue
        if low.startswith("*elset") or low.startswith("*nset"):
            kind = "elset" if low.startswith("*elset") else "nset"
            m = re.search(r"(?:elset|nset)\s*=\s*([^,\s]+)", l, flags=re.I)
            name = m.group(1)
            gen = "generate" in low
            i += 1
            ids = []
            while i < n and not lines[i].startswith("*"):
                if lines[i].strip():
                    vals = [int(s) for s in lines[i].split(",") if s.strip()]
                    if gen:
                        a, b, c = (vals + [1])[:3]
                        ids += list(range(a, b + 1, c))
                    else:
                        ids += vals
                i += 1
            sets.setdefault((kind, name), []).extend(ids)
            continue
        i += 1
    return lines, nodes, elems, sets
