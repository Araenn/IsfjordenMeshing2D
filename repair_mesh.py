#!/usr/bin/env python3
"""Single Isfjorden mesh checker and optional local repair.

Typical CURRENT situation (reuse the all-quad mesh; do NOT regenerate):
    python3 repair_mesh.py --check \
        archives_maillage/output_hybrid5/Isfjorden_candidate_patch_retile_steiner.msh \
        --reference archives_maillage/output_hybrid5/Isfjorden_candidate_patch_retile.msh

When the check passes, use --adopt to copy the already validated mesh to
output_mesh/Isfjorden.msh (without changing/re-writing node numbering).

If a future .msh still has triangles: pass --repair instead of --adopt.
The same two repair phases run automatically when isfjorden_msh.py generates a new mesh.

Never overwrites input or an existing different canonical mesh. No Exodus steps.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import shutil
from pathlib import Path

import gmsh
import numpy as np

import meshing_functions as msh

ROOT = Path(__file__).resolve().parent
CANONICAL = ROOT / "output_mesh"


def abs_path(path):
    path = Path(path).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def check_loaded_model(reference=None):
    """Geometric check with optional source coordinate cloud and boundary comparison."""
    # First capture the current target and calculate quality. Reference is read
    # second, so that the existing mesh is never silently re-meshed/re-written.
    state = msh.mesh_snapshot()
    if set(state["counts"]) != {3}:
        raise RuntimeError(f"Residual triangles or non-quads: {state['counts']}")
    stats = msh.inspect_quad_quality(0.05, 0.1, label="read-only validation")
    if stats["n_failed"]:
        raise RuntimeError(f"{stats['n_failed']} quads have minSICN <= 0.05")
    msh.audit_global_quad_edges()
    physical_names = [name for (d, _), (name, _) in state["physical"].items() if d == 2]
    if 'socle_1' not in physical_names or 'socle_3' not in physical_names:
        raise RuntimeError('Missing physical water or basement domain')
    if reference is not None:
        gmsh.clear()
        gmsh.open(str(reference))
        source = msh.mesh_snapshot()
        msh.compare_mesh_snapshots(source, state, tol=1e-8)
    print("\nPASS: geometry, 1-D boundary, physical groups and all-quads quality", flush=True)
    return state


def place_canonical(mesh_input, materials, output):
    """Copy a verified .msh without invoking Gmsh's renumber-on-save machinery."""
    output.mkdir(parents=True, exist_ok=True)
    dst = output / "Isfjorden.msh"
    mat = output / "materials_generated.json"
    if dst.exists() and not filecmp.cmp(dst, mesh_input, shallow=False):
        raise RuntimeError(f"A DIFFERENT canonical mesh already exists at {dst}. Refusing overwrite.")
    if mat.exists():
        current = json.loads(mat.read_text(encoding="utf-8"))
        if current != materials:
            raise RuntimeError(f"Existing {mat} has different physical properties; refusing overwrite")
    if not dst.exists():
        temp = output / 'Isfjorden.copying.msh'
        if temp.exists():
            raise RuntimeError(f"Leftover staged copy exists: {temp}")
        shutil.copy2(mesh_input, temp)
        temp.replace(dst)
    if not mat.exists():
        mat.write_text(json.dumps(materials, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"\nPASS: canonical validated .msh: {dst}")
    print(f"      generator-compatible material metadata: {mat}")
    return dst


def run(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh',type=Path,help='Input MSH (unchanged)')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--check',action='store_true',help='Only validate (default)')
    group.add_argument('--adopt',action='store_true',help='Validate and copy to output_mesh/Isfjorden.msh')
    group.add_argument('--repair',action='store_true',help='Repair triangles, validate, then create a new canonical MSH')
    parser.add_argument('--reference',type=Path,
                        help='Original pre-repair MSH to compare physically (independent of node tags)')
    parser.add_argument('--output-dir',type=Path,default=CANONICAL)
    args=parser.parse_args(argv)
    src = abs_path(args.mesh)
    ref = abs_path(args.reference) if args.reference else None
    dst_dir = abs_path(args.output_dir)
    if not src.is_file(): parser.error(f'Input not found: {src}')
    if ref is not None and not ref.is_file(): parser.error(f'Reference not found: {ref}')
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 1)
        gmsh.open(str(src))
        if args.repair:
            if (dst_dir/'Isfjorden.msh').exists() or (dst_dir/'materials_generated.json').exists():
                raise RuntimeError(f'{dst_dir} already has a canonical mesh/materials; refusing overwrite')
            print('\nRepair mode: original file stays untouched', flush=True)
            original = msh.mesh_snapshot()
            msh.ensure_all_quads(allow_repair=True)
            immediate = msh.mesh_snapshot()
            msh.compare_mesh_snapshots(original, immediate)
            out=dst_dir/'Isfjorden.msh'
            if out.exists():
                raise RuntimeError(f'Already exists: {out}; cannot overwrite')
            dst_dir.mkdir(parents=True, exist_ok=True)
            staged=dst_dir/'Isfjorden.incomplete.msh'
            if staged.exists():
                raise RuntimeError(f'Incomplete file exists from earlier run: {staged}')
            gmsh.option.setNumber('Mesh.MshFileVersion',2.2)
            gmsh.option.setNumber('Mesh.Binary',1)
            gmsh.write(str(staged))
            gmsh.clear()
            gmsh.open(str(staged))
            final=msh.mesh_snapshot()
            msh.compare_mesh_snapshots(original, final)
            stats=msh.inspect_quad_quality(0.05,0.1,label='reopened repaired MSH')
            if stats['n_failed'] or set(final['counts'])!={3}:
                raise RuntimeError('Repaired output failed final quality validation')
            msh.audit_global_quad_edges()
            materials=msh.material_data_from_physical_groups()
            staged.replace(out)
            mat_path=dst_dir/'materials_generated.json'
            mat_path.write_text(json.dumps(materials,indent=2)+'\n',encoding='utf-8')
            print(f'\nPASS: repaired .msh and materials written to {dst_dir}',flush=True)
        else:
            # Take material definitions from the verified target (BEFORE a reference is loaded).
            materials=msh.material_data_from_physical_groups()
            check_loaded_model(ref)
            if args.adopt:
                place_canonical(src,materials,dst_dir)
            else:
                print('READ-ONLY PASS: nothing written or moved',flush=True)
    finally:
        gmsh.finalize()
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
