#!/usr/bin/env python3
import re
import sys
import h5py
import numpy as np

EDGE_LOCAL = {
    1: (0, 1),  # SPECFEM/Exodus bottom
    2: (1, 2),  # right
    3: (2, 3),  # top
    4: (3, 0),  # left
}

EXPECTED_OUTER_SIDE = {
    "zmin": 1,
    "xmax": 2,
    "zmax": 3,
    "xmin": 4,
}

def decode_names(ds):
    out = []
    for row in ds:
        raw = b"".join(row.tolist()) if hasattr(row, "tolist") else b"".join(row)
        out.append(raw.decode("utf-8", errors="ignore").rstrip("\x00 "))
    return out

def numbered_keys(h5, prefix):
    keys = []
    for k in h5.keys():
        m = re.fullmatch(rf"{re.escape(prefix)}(\d+)", k)
        if m:
            keys.append((int(m.group(1)), k))
    return [k for _, k in sorted(keys)]

def main(path):
    with h5py.File(path, "r") as f:
        cx = np.asarray(f["coordx"][:], float)
        cy = np.asarray(f["coordy"][:], float)

        connect_keys = numbered_keys(f, "connect")
        blocks = [np.asarray(f[k][:], dtype=np.int64) for k in connect_keys]
        sizes = np.asarray([len(b) for b in blocks], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(sizes)))
        all_conn = np.concatenate(blocks, axis=0)

        ss_names = decode_names(f["ss_names"])

        print(f"nodes={len(cx)} elements={len(all_conn)} blocks={len(blocks)}")
        print(f"x range: {cx.min():.9g} .. {cx.max():.9g}")
        print(f"y range: {cy.min():.9g} .. {cy.max():.9g}")
        print("\nSPECFEM LOCAL-SIDE CHECK")
        print("expected: zmin=1, xmax=2, zmax=3, xmin=4")
        print("expected coupling: water=1, sediment=3")

        total_bad = 0

        for i, name in enumerate(ss_names, start=1):
            elems = np.asarray(f[f"elem_ss{i}"][:], dtype=np.int64)
            sides = np.asarray(f[f"side_ss{i}"][:], dtype=np.int64)

            print(f"\n{name}: {len(elems)} element-sides")
            unique_sides = sorted(set(int(x) for x in sides))
            print(f"  side ids: {unique_sides}")

            if name in EXPECTED_OUTER_SIDE:
                expected = EXPECTED_OUTER_SIDE[name]
                bad_sem = sides != expected
                print(
                    f"  semantic side expected={expected}; "
                    f"BAD={int(bad_sem.sum())}/{len(bad_sem)}"
                )
                total_bad += int(bad_sem.sum())

            if "|" in name:
                if len(sides) % 2:
                    print("  ERROR: odd number of coupling entries")
                    total_bad += 1
                    continue
                water_bad = sides[0::2] != 1
                sed_bad = sides[1::2] != 3
                print(
                    f"  water side=1 BAD={int(water_bad.sum())}/{len(water_bad)}; "
                    f"sediment side=3 BAD={int(sed_bad.sum())}/{len(sed_bad)}"
                )
                total_bad += int(water_bad.sum() + sed_bad.sum())

            # Geometric verification of the selected local side.
            q = all_conn[elems - 1]
            n1 = np.empty(len(elems), dtype=np.int64)
            n2 = np.empty(len(elems), dtype=np.int64)
            for side in (1, 2, 3, 4):
                mask = sides == side
                a, b = EDGE_LOCAL[side]
                n1[mask] = q[mask, a]
                n2[mask] = q[mask, b]

            x1, y1 = cx[n1 - 1], cy[n1 - 1]
            x2, y2 = cx[n2 - 1], cy[n2 - 1]
            scale = max(cx.max()-cx.min(), cy.max()-cy.min(), 1.0)
            tol = 1e-9 * scale

            if name == "zmin":
                bad_geo = (
                    (np.abs(y2-y1) > tol)
                    | (np.abs(y1-cy.min()) > tol)
                    | (np.abs(y2-cy.min()) > tol)
                )
            elif name == "zmax":
                bad_geo = (
                    (np.abs(y2-y1) > tol)
                    | (np.abs(y1-cy.max()) > tol)
                    | (np.abs(y2-cy.max()) > tol)
                )
            elif name == "xmin":
                bad_geo = (
                    (np.abs(x2-x1) > tol)
                    | (np.abs(x1-cx.min()) > tol)
                    | (np.abs(x2-cx.min()) > tol)
                )
            elif name == "xmax":
                bad_geo = (
                    (np.abs(x2-x1) > tol)
                    | (np.abs(x1-cx.max()) > tol)
                    | (np.abs(x2-cx.max()) > tol)
                )
            elif "|" in name:
                # Entries are water, sediment, water, sediment...
                c1 = np.sort(np.column_stack((n1[0::2], n2[0::2])), axis=1)
                c2 = np.sort(np.column_stack((n1[1::2], n2[1::2])), axis=1)
                bad_geo = np.any(c1 != c2, axis=1)
            else:
                bad_geo = np.zeros(len(elems), dtype=bool)

            print(f"  geometric BAD={int(bad_geo.sum())}/{len(bad_geo)}")
            total_bad += int(bad_geo.sum())

        print("\nRESULT")
        if total_bad:
            print(f"FAIL: {total_bad} semantic/geometric side-set errors")
            sys.exit(1)
        print("PASS: all side sets use SPECFEM semantic local-edge numbering")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} mesh.e")
        sys.exit(2)
    main(sys.argv[1])
