#!/usr/bin/env python3
"""Render 2-D SPECfem-waves XDMF outputs with VisIt 3.5 and FFmpeg.

Place specfem_visit.py and visit_render.py together in the meshing directory.

    cd ~/Desktop/code/meshing
    python3 specfem_visit.py 2715444
    python3 specfem_visit.py 2715444 --encode-only
    python3 specfem_visit.py 2715444 --time 0.25

The original XDMF and binary files are never modified or copied. Converted
XMF files and rendered frames remain inside output_simu/job_<ID>/.
"""

import argparse
import hashlib
import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def local_name(tag):
    return tag.rsplit('}', 1)[-1]


def children(element, name):
    return [] if element is None else [c for c in element if local_name(c.tag) == name]


def child(element, name):
    return next(iter(children(element, name)), None)


def time_step_number(step):
    match = re.search(r'T(\d+)', step.get('Name', ''))
    return int(match.group(1)) if match else None


def get_times(series, steps, solver_dt):
    """Interpret a time series even when the XDMF HyperSlab count is wrong."""
    count = len(steps)
    explicit = [child(step, 'Time') for step in steps]
    if all(t is not None and t.get('Value') is not None for t in explicit):
        return [float(t.get('Value')) for t in explicit], 'explicit time values'

    time = child(series, 'Time')
    item = child(time, 'DataItem')
    if time is not None and item is not None and item.text:
        values = [float(v) for v in item.text.split()]
        kind = (time.get('TimeType') or time.get('Type') or '').lower()
        if kind == 'list' and len(values) == count:
            return values, 'time list'
        if kind == 'hyperslab' and len(values) >= 3:
            start, stride, declared_count = values[:3]
            candidate = [start + i * stride for i in range(count)]
            numbers = [time_step_number(step) for step in steps]
            check_solver = (solver_dt is not None
                            and all(n is not None for n in numbers)
                            and all(math.isclose(t, n * solver_dt, rel_tol=1e-5, abs_tol=1e-7)
                                    for t, n in zip(candidate, numbers)))
            if check_solver or (int(round(declared_count)) == count and stride > 0):
                note = '' if int(round(declared_count)) == count else ' (incorrect count ignored)'
                return candidate, 'time HyperSlab' + note

    numbers = [time_step_number(step) for step in steps]
    if solver_dt is not None and all(n is not None for n in numbers):
        return [n * solver_dt for n in numbers], 'step number x solver dt'
    raise ValueError('Cannot determine physical time stamps; set --dt-solver')


def convert_xdmf(source, solver_dt):
    """Flatten Temporal > Spatial > Uniform to Temporal > Uniform for VisIt."""
    original = ET.parse(source).getroot()
    domain = child(original, 'Domain')
    topology, geometry = child(domain, 'Topology'), child(domain, 'Geometry')
    if topology is None or geometry is None:
        raise ValueError('%s has no shared topology and geometry' % source.name)
    series = next((g for g in children(domain, 'Grid')
                   if g.get('CollectionType', '').lower() == 'temporal'), None)
    if series is None:
        raise ValueError('%s has no temporal collection' % source.name)
    steps = children(series, 'Grid')
    if not steps:
        raise ValueError('%s has no time steps' % source.name)
    times, time_source = get_times(series, steps, solver_dt)
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('%s: non-increasing time stamps: %s' % (source.name, times))

    root = ET.Element('Xdmf', Version=original.get('Version', '2.0'))
    new_domain = ET.SubElement(root, 'Domain')
    new_series = ET.SubElement(new_domain, 'Grid', Name='SPECfem_TimeSeries',
                               GridType='Collection', CollectionType='Temporal')
    fields = {}
    for index, (step, seconds) in enumerate(zip(steps, times)):
        attributes = {}
        grids = children(step, 'Grid') or [step]
        for grid in grids:
            for attribute in children(grid, 'Attribute'):
                name = attribute.get('Name', '')
                if name and name not in attributes:
                    attributes[name] = attribute
                    fields[name] = attribute.get('AttributeType', '')
        if not attributes:
            raise ValueError('%s: no fields at time step %d' % (source.name, index))
        uniform = ET.SubElement(new_series, 'Grid', Name='Step_%04d' % index,
                                 GridType='Uniform')
        ET.SubElement(uniform, 'Time', Value=format(seconds, '.12g'))
        uniform.append(copy.deepcopy(topology))
        uniform.append(copy.deepcopy(geometry))
        for attribute in attributes.values():
            uniform.append(copy.deepcopy(attribute))

    destination = source.with_name(source.stem.replace('_2d', '') + '_timeseries.xmf')
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(destination, encoding='utf-8', xml_declaration=True)
    ET.parse(destination)
    return destination, times, fields, time_source



def binary_items_for_field(xmf, field):
    """Find the raw binary attribute for every time state in a flattened XMF."""
    tree = ET.parse(xmf)
    root = tree.getroot()
    domain = child(root, 'Domain')
    series = next((g for g in children(domain, 'Grid')
                   if g.get('CollectionType', '').lower() == 'temporal'), None)
    if series is None:
        raise ValueError('No temporal collection in %s' % xmf)
    result = []
    target = 'potential' if field == 'potential' else 'displacement'
    for index, step in enumerate(children(series, 'Grid')):
        attribute = next((a for a in children(step, 'Attribute')
                          if a.get('Name') == target), None)
        if attribute is None:
            raise ValueError('%s time %d has no %s field' % (xmf.name, index, target))
        data = child(attribute, 'DataItem')
        if data is None or (data.get('Format') or '').lower() != 'binary':
            raise ValueError('%s state %d: expected raw binary DataItem for %s' %
                             (xmf.name, index, target))
        name = (data.text or '').strip()
        if not name:
            raise ValueError('%s state %d: empty binary path' % (xmf.name, index))
        path = (xmf.parent / name).resolve()
        if not path.is_file():
            raise FileNotFoundError('Missing binary time-step data: %s' % path)
        dims = tuple(int(v) for v in data.get('Dimensions', '').split())
        if not dims:
            raise ValueError('Missing Dimensions in %s at state %d' % (xmf.name, index))
        precision = int(data.get('Precision', '4'))
        kind = (data.get('NumberType') or data.get('DataType') or 'Float').lower()
        endian = (data.get('Endian') or 'Little').lower()
        if kind not in ('float', 'double', 'int', 'uint') or precision not in (4, 8):
            raise ValueError('Unsupported binary type %s/%d in %s' %
                             (kind, precision, path))
        if endian not in ('little', 'big'):
            raise ValueError('Unsupported byte order %s in %s' % (endian, path))
        seek = int(data.get('Seek', '0'))
        count = math.prod(dims)
        required = seek + count * precision
        if path.stat().st_size < required:
            raise ValueError('Truncated binary %s: %d bytes, need at least %d' %
                             (path, path.stat().st_size, required))
        component_count = (dims[-1] if len(dims) >= 2 else 1)
        if target == 'displacement' and component_count < 2:
            raise ValueError('Expected 2 displacement components in %s; dimensions=%r' %
                             (path, dims))
        if target == 'potential' and component_count != 1:
            raise ValueError('Expected scalar potential in %s; dimensions=%r' %
                             (path, dims))
        result.append({'path': path, 'dims': dims, 'precision': precision,
                       'kind': kind, 'endian': endian, 'seek': seek,
                       'component_count': component_count})
    if not result:
        raise ValueError('No time steps found in %s' % xmf)
    return result


def fixed_color_limit(xmf, field, percentile, output_root, recompute=False):
    """Compute one amplitude limit across ALL states, independent of image/view."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Global fixed scales require NumPy: python3 -m pip install numpy') from exc

    items = binary_items_for_field(xmf, field)
    signatures = []
    for info in items:
        path = info['path']
        stat = path.stat()
        signatures.append([str(path), stat.st_size, stat.st_mtime_ns,
                           info['dims'], info['precision'], info['kind'],
                           info['endian'], info['seek']])
    key = hashlib.sha256(json.dumps([field, percentile, signatures],
                                     sort_keys=True).encode('utf-8')).hexdigest()
    cache_path = output_root / ('color_limits_' + xmf.stem + '_' + field + '.json')
    if cache_path.exists() and not recompute:
        try:
            cached = json.loads(cache_path.read_text(encoding='utf-8'))
            if cached.get('signature') == key and math.isfinite(cached['limit']) and cached['limit'] > 0:
                print('[COLOR] Cached global %s (%d states): +/-%.9g (p=%g)' %
                      (field, len(items), cached['limit'], percentile), flush=True)
                return float(cached['limit'])
        except (ValueError, KeyError, TypeError):
            pass

    positive_limit = 0.0
    sampled = []
    nonfinite = 0
    print('[COLOR] Scanning %d %s snapshots to calculate ONE fixed color range...' %
          (len(items), field), flush=True)
    for index, info in enumerate(items):
        type_char = ('f' if info['kind'] in ('float', 'double') else
                     ('i' if info['kind'] == 'int' else 'u'))
        np_type = np.dtype(('<' if info['endian'] == 'little' else '>') +
                           type_char + str(info['precision']))
        ncols = info['component_count']
        nrows = math.prod(info['dims']) // ncols
        source = np.memmap(info['path'], dtype=np_type, mode='r',
                           offset=info['seek'], shape=(nrows, ncols))
        if percentile < 100.0:
            # A reproducible, evenly weighted sample of each simulation state.
            rng = np.random.default_rng(20261106 + index)
            take = min(150_000, nrows)
            edges = np.linspace(0, nrows, take + 1, dtype=np.int64)
            rows = edges[:-1] + np.floor(
                rng.random(take) * (edges[1:] - edges[:-1])
            ).astype(np.int64)
            vals = source[rows]
            if field == 'magnitude':
                amplitudes = np.sqrt(np.sum(vals[:, :2].astype('float64') ** 2, axis=1))
            elif field == 'x':
                amplitudes = np.abs(vals[:, 0].astype('float64'))
            elif field == 'y':
                amplitudes = np.abs(vals[:, 1].astype('float64'))
            else:
                amplitudes = np.abs(vals[:, 0].astype('float64'))
            finite = np.isfinite(amplitudes)
            nonfinite += int(amplitudes.size - np.count_nonzero(finite))
            sampled.append(amplitudes[finite])
        else:
            # Scan in blocks: bounded RAM use even for >100M node grids.
            for begin in range(0, nrows, 2_000_000):
                values = source[begin:begin + 2_000_000]
                if field == 'magnitude':
                    amp = np.hypot(values[:, 0], values[:, 1])
                else:
                    component = (0 if field in ('potential', 'x') else 1)
                    amp = np.abs(values[:, component])
                finite = np.isfinite(amp)
                nonfinite += int(amp.size - np.count_nonzero(finite))
                if np.any(finite):
                    positive_limit = max(positive_limit, float(np.max(amp[finite])))
        del source
        print('[COLOR] %s: state %d/%d processed' % (field, index + 1, len(items)), flush=True)

    if percentile < 100.0:
        if not sampled or not any(v.size for v in sampled):
            raise ValueError('No finite values in %s; cannot compute percentile' % field)
        positive_limit = float(np.percentile(np.concatenate(sampled), percentile))
    if nonfinite:
        raise ValueError('%s contains %d sampled non-finite amplitudes (NaN/Inf); '
                         'fix the simulation data before rendering' % (field, nonfinite))
    if not math.isfinite(positive_limit) or positive_limit <= 0:
        raise ValueError('Global %s limit is %g (all-zero or invalid data). '
                         'Use --solid-limit / --water-limit if appropriate.' %
                         (field, positive_limit))
    cache = {'field': field, 'percentile': percentile,
             'limit': positive_limit, 'states': len(items), 'signature': key,
             'clipped': percentile < 100.0,
             'method': ('full-data global maximum' if percentile == 100.0
                        else 'sampled global percentile')}
    cache_path.write_text(json.dumps(cache, indent=2), encoding='utf-8')
    if field == 'magnitude':
        scale_text = '[0, %.9g]' % positive_limit
    else:
        scale_text = '[-%.9g, +%.9g]' % (positive_limit, positive_limit)
    print('[COLOR] FIXED global %s range: %s (p=%g)' %
          (field, scale_text, percentile), flush=True)
    return positive_limit


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description='SPECfem-waves XDMF/bin -> VisIt -> global + zoom PNG, MP4 and GIF.')
    parser.add_argument('job_id', help='Job number, e.g. 2715444 or job_2715444')
    parser.add_argument('--jobs-root', type=Path, default=root / 'output_simu')
    parser.add_argument('--domain', choices=('both', 'viscoelastic', 'acoustic'), default='both')
    parser.add_argument('--solid-field', choices=('magnitude', 'x', 'y'), default='magnitude')
    parser.add_argument('--water-field', choices=('potential', 'magnitude', 'x', 'y'),
                        default='potential', help='Potential is NOT pressure')
    parser.add_argument('--solid-limit', type=float, default=None,
                        help='Positive color-limit magnitude (signed fields use +/- limit)')
    parser.add_argument('--water-limit', type=float, default=None,
                        help='Positive color-limit magnitude (signed fields use +/- limit)')
    parser.add_argument('--scale-percentile', type=float, default=100.0,
                        help='100 = exact global maximum (default). Values <100 use sampled '
                             'global percentile and clip larger amplitudes.')
    parser.add_argument('--recompute-limits', action='store_true',
                        help='Recalculate global color limits even if the binary cache is valid')
    parser.add_argument('--field', choices=('magnitude', 'x', 'y', 'potential'), default=None,
                        help='Field alias for a single-domain run')
    parser.add_argument('--limit', type=float, default=None,
                        help='Color-limit alias for a single-domain run')
    parser.add_argument('--time', type=float, default=None,
                        help='Render one frame, closest to specified physical time in seconds')
    parser.add_argument('--single-frame', action='store_true',
                        help='Render one PNG per view at 0.25 s by default')
    parser.add_argument('--all-frames', action='store_true',
                        help='Render all time steps, already the default')
    parser.add_argument('--format', choices=('both', 'gif', 'mp4', 'png'), default='both')
    parser.add_argument('--fps', type=int, default=6, help='Animation frame rate (default: 6)')
    parser.add_argument('--gif-width', type=int, default=1100,
                        help='Maximum GIF width (default: 1100 pixels)')
    parser.add_argument('--window', type=float, nargs=4,
                        metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                        default=(2000., 4000., -200., 0.),
                        help='Zoom bounds in meters: xmin xmax ymin ymax')
    parser.add_argument('--views', choices=('both', 'zoom', 'global'), default='both')
    parser.add_argument('--global-window', type=float, nargs=4,
                        metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'), default=None,
                        help='Override automatic global bounds')
    parser.add_argument('--full-domain', action='store_true',
                        help='Legacy alias for --views global')
    parser.add_argument('--dt-solver', type=float, default=5e-6,
                        help='Solver time step used when XDMF timestamps are unusable')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--title', default='Acoustic-elastic wave propagation',
                        help='English title shown above the plot')
    parser.add_argument('--visit', default=None, help='Path to VisIt executable')
    parser.add_argument('--ffmpeg', default=None, help='Path to FFmpeg executable')
    parser.add_argument('--with-window', action='store_true',
                        help='Start VisIt CLI with an interactive rendering window')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Create VisIt-compatible XMF files but do not render')
    parser.add_argument('--encode-only', action='store_true',
                        help='Rebuild MP4/GIF from existing PNGs WITHOUT running VisIt')
    return parser, parser.parse_args()


def confirm_png_sequence(folder, count):
    expected = [folder / ('frame_%04d.png' % i) for i in range(count)]
    missing = [p.name for p in expected if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError('Missing rendered frames in %s: %s' %
                           (folder, ', '.join(missing[:8])))
    return expected


def make_movies(frames_dir, base, fps, fmt, gif_width, ffmpeg):
    """Encode a single view. Each view has a distinct output basename."""
    if fmt == 'png':
        return []
    executable = ffmpeg or shutil.which('ffmpeg')
    if not executable:
        raise RuntimeError('FFmpeg not found. Install it with: sudo apt install ffmpeg')
    source = str(frames_dir / 'frame_%04d.png')
    common = [str(executable), '-hide_banner', '-loglevel', 'error', '-y',
              '-framerate', str(fps), '-start_number', '0', '-i', source]
    outputs = []
    if fmt in ('both', 'mp4'):
        target = Path(str(base) + '.mp4')
        command = common + ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
        print('[FFMPEG] MP4 ->', target, flush=True)
        subprocess.run(command, check=True)
        outputs.append(target)
    if fmt in ('both', 'gif'):
        target = Path(str(base) + '.gif')
        filter_complex = ('[0:v]scale=w=min(%d\\,iw):h=-2:flags=lanczos,'
                          'split[a][b];[a]palettegen=stats_mode=full[p];'
                          '[b][p]paletteuse=dither=bayer:bayer_scale=4' % gif_width)
        command = common + ['-filter_complex', filter_complex, '-loop', '0', str(target)]
        print('[FFMPEG] GIF ->', target, flush=True)
        subprocess.run(command, check=True)
        outputs.append(target)
    for target in outputs:
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError('FFmpeg did not create a nonempty file: ' + str(target))
        print('[OK] %s (%0.1f MiB)' % (target, target.stat().st_size / 1048576), flush=True)
    return outputs


def main():
    parser, args = parse_args()
    root = Path(__file__).resolve().parent
    job_id = args.job_id.removeprefix('job_')
    if not re.fullmatch(r'\d+', job_id):
        parser.error('Invalid job ID: use a number such as 2715444')
    job_dir = args.jobs_root.expanduser().resolve() / ('job_' + job_id)
    if not job_dir.is_dir():
        parser.error('Job directory not found: ' + str(job_dir))
    if args.single_frame and args.all_frames:
        parser.error('--single-frame and --all-frames cannot be used together')
    if args.encode_only and (args.single_frame or args.time is not None or args.prepare_only):
        parser.error('--encode-only cannot be combined with --time, --single-frame or --prepare-only')
    if not (90.0 <= args.scale_percentile <= 100.0):
        parser.error('--scale-percentile must be between 90 and 100')
    if args.solid_field == 'potential':
        parser.error('The viscoelastic domain has no potential field')
    if args.field is not None or args.limit is not None:
        if args.domain == 'both':
            parser.error('Use --solid-field/--water-field and --solid-limit/--water-limit '
                         'when displaying both domains')
        if args.field is not None:
            if args.domain == 'viscoelastic':
                if args.field == 'potential':
                    parser.error('Potential is not present in the viscoelastic domain')
                args.solid_field = args.field
            else:
                args.water_field = args.field
        if args.limit is not None:
            if args.domain == 'viscoelastic':
                args.solid_limit = args.limit
            else:
                args.water_limit = args.limit
    if any(limit is not None and (limit <= 0 or not math.isfinite(limit))
           for limit in (args.solid_limit, args.water_limit)):
        parser.error('Color limits must be positive finite values')
    if args.full_domain:
        args.views = 'global'
    for name, bounds in [('zoom', args.window), ('global', args.global_window)]:
        if bounds is not None and not (bounds[0] < bounds[1] and bounds[2] < bounds[3]):
            parser.error('%s bounds must satisfy xmin<xmax and ymin<ymax' % name)
    if args.fps < 1 or args.width < 400 or args.height < 200 or args.gif_width < 100:
        parser.error('Invalid --fps / --width / --height / --gif-width')

    print('[JOB] %s: %s' % (job_id, job_dir), flush=True)
    results = {}
    for domain in ('viscoelastic', 'acoustic'):
        source = job_dir / (domain + '_2d.xdmf')
        if not source.is_file():
            print('[INFO] Optional XDMF not found:', source.name, flush=True)
            continue
        destination, times, fields, method = convert_xdmf(source, args.dt_solver)
        results[domain] = (destination, times, fields)
        print('[XMF] %s -> %s | %d time steps [%g, %g] s | %s' %
              (source.name, destination.name, len(times), times[0], times[-1], method), flush=True)
        print('[XMF] Fields:', fields, flush=True)

    active = ['viscoelastic', 'acoustic'] if args.domain == 'both' else [args.domain]
    for domain in active:
        if domain not in results:
            parser.error('Required XDMF domain not found: ' + domain)
        field = args.solid_field if domain == 'viscoelastic' else args.water_field
        fields = results[domain][2]
        target = 'potential' if field == 'potential' else 'displacement'
        kind = 'scalar' if target == 'potential' else 'vector'
        if fields.get(target, '').lower() != kind:
            parser.error('%s has no %s %s field' % (domain, kind, target))
    times = results[active[0]][1]
    if len(active) == 2:
        other = results[active[1]][1]
        if len(times) != len(other) or not all(
                math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)
                for a, b in zip(times, other)):
            parser.error('Acoustic and viscoelastic time stamps do not match')

    output_root = job_dir / 'visit_renders'
    output_root.mkdir(exist_ok=True)

    # Do not rescan the 8+ GiB dataset if the user only wants to encode old PNGs.
    if not args.encode_only and not args.prepare_only:
        for domain in active:
            field = args.solid_field if domain == 'viscoelastic' else args.water_field
            limit = args.solid_limit if domain == 'viscoelastic' else args.water_limit
            if limit is None:
                limit = fixed_color_limit(results[domain][0], field,
                                          args.scale_percentile, output_root,
                                          args.recompute_limits)
                if domain == 'viscoelastic':
                    args.solid_limit = limit
                else:
                    args.water_limit = limit
            print('[COLOR] %s / %s: FIXED over all states and views, %s%.9g' %
                  (domain, field, '+/-' if field != 'magnitude' else '0..', limit),
                  flush=True)

    one_frame = args.single_frame or (args.time is not None and not args.all_frames)
    desired_time = 0.25 if args.time is None else args.time
    selected = min(range(len(times)), key=lambda i: abs(times[i] - desired_time))
    frames = [selected] if one_frame else list(range(len(times)))
    tag = ('water_solid_' + args.water_field + '_' + args.solid_field
           if args.domain == 'both'
           else args.domain + '_' + (args.solid_field if args.domain == 'viscoelastic'
                                      else args.water_field))
    view_names = ('global', 'zoom') if args.views == 'both' else (args.views,)
    views = {}
    for view in view_names:
        frames_dir = output_root / (tag + '_' + view + '_frames')
        frames_dir.mkdir(exist_ok=True)
        views[view] = {
            'window': (list(args.window) if view == 'zoom' else
                       (list(args.global_window) if args.global_window is not None else None)),
            'output_dir': str(frames_dir),
        }
    config_file = output_root / ('job_' + job_id + '_' + tag + '_config.json')
    config = {
        'job_id': job_id,
        'domain': args.domain,
        'datasets': {
            domain: {'database': str(results[domain][0]),
                     'field': args.solid_field if domain == 'viscoelastic' else args.water_field,
                     'limit': args.solid_limit if domain == 'viscoelastic' else args.water_limit}
            for domain in active
        },
        'times': times, 'frames': frames, 'views': views,
        'width': args.width, 'height': args.height, 'title': args.title,
        'scale_percentile': args.scale_percentile,
        'fixed_scales': not args.encode_only and not args.prepare_only,
    }
    config_file.write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('[OUTPUT] %s | %d frame(s) | views: %s' %
          (output_root, len(frames), ', '.join(views)), flush=True)

    if args.prepare_only:
        print('[OK] XMF preparation complete; rendering skipped', flush=True)
        return

    if not args.encode_only:
        renderer = root / 'visit_render.py'
        if not renderer.is_file():
            parser.error('Place visit_render.py alongside specfem_visit.py in ' + str(root))
        visit = args.visit or ('/usr/local/visit/bin/visit'
                               if Path('/usr/local/visit/bin/visit').exists()
                               else shutil.which('visit'))
        if not visit:
            parser.error('VisIt executable not found; provide --visit /path/to/visit')
        command = [str(visit), '-cli']
        if not args.with_window:
            command.append('-nowin')
        command += ['-s', str(renderer)]
        environment = os.environ.copy()
        environment['SPECFEM_VISIT_CONFIG'] = str(config_file)
        print('[VISIT]', ' '.join(command), flush=True)
        result = subprocess.run(command, cwd=job_dir, env=environment, check=False)
        if result.returncode != 0:
            raise RuntimeError('VisIt rendering failed (exit code %d)' % result.returncode)
    else:
        print('[ENCODE ONLY] Reusing existing PNGs: original color scales are unchanged!', flush=True)

    movies = []
    for view, view_settings in views.items():
        images_dir = Path(view_settings['output_dir'])
        image_files = confirm_png_sequence(images_dir, len(frames))
        print('[FRAMES] %s: %d PNGs, e.g. %s' %
              (view, len(image_files), image_files[0]), flush=True)
        if not one_frame:
            movie_base = output_root / ('job_' + job_id + '_' + tag + '_' + view)
            movies.extend(make_movies(images_dir, movie_base, args.fps, args.format,
                                      args.gif_width, args.ffmpeg))
    if one_frame:
        print('[INFO] Single-frame mode: no GIF or MP4 requested', flush=True)
    else:
        expected_count = len(views) * (2 if args.format == 'both' else
                                       (0 if args.format == 'png' else 1))
        if len(movies) != expected_count:
            raise RuntimeError('Expected %d encoded animations; created %d' %
                               (expected_count, len(movies)))
    print('[COMPLETE] %d animation file(s) in %s' % (len(movies), output_root), flush=True)
    for file in movies:
        print('  ', file, flush=True)


if __name__ == '__main__':
    main()
