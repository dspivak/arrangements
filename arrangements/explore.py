"""Read-only inspection of morphisms of IMC's Cartesian array subcategory.

Uses the existing polynomial interpretation, not a second dynamics. Retained
composition trees expose the actual factors, values and covectors. An opaque
primitive still exposes its four interface sides and internal configuration.
Inspection never steps an integrator (including stochastic ones).

The browser's version-1 scene protocol also accepts adapters for smooth,
many-round, stochastic and externally executed examples. Such adapters must
provide actual observations and declare unavailable loads as None, never zero.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement
from .interpretation import interpretation, trivial_omega
from .smooth import SmoothArrangement


def inspect(arr: Union[Arrangement, SmoothArrangement], state: Any = None, *, values: Any = None,
            incoming: Any = None, inputs: Any = None, loads: Any = None,
            duration: Optional[float] = None, round_index: int = 0) -> Dict[str, Any]:
    """Observe one interpretation round at ``state`` without advancing it.

    ``state`` is a state of the arrangement's integrator (a pytree of arrays); ``values``, ``incoming``,
    ``inputs`` and ``loads`` are arrays or a one-form for an array arrangement and a plot/point pair, a
    one-form, a tuple and a callback for a smooth one, so they are ``Any``.

    ``values`` are M outputs, ``incoming`` their one-form on M inputs;
    ``inputs`` are N inputs, ``loads`` the covector on N outputs. Defaults are
    zero boundary data. With look-ahead, pass the actual step's duration.
    Configuration inclinations are deliberately distinct from port loads.
    """
    if hasattr(arr, 'M_in'):
        return inspect_smooth(arr, state, values=values, incoming=incoming, inputs=inputs,
                              loads=loads, duration=duration, round_index=round_index)
    state = arr.P.init if state is None else state
    q = _read(arr.P, state, duration, round_index)
    m = jnp.zeros(arr.out_dim_M) if values is None else jnp.asarray(values)
    n = jnp.zeros(arr.in_dim_N) if inputs is None else jnp.asarray(inputs)
    xi = jnp.zeros(arr.out_dim_N) if loads is None else jnp.asarray(loads)
    for x, size, name in [(m, arr.out_dim_M, 'values'), (n, arr.in_dim_N, 'inputs'),
                          (xi, arr.out_dim_N, 'loads')]:
        if x.shape != (size,):
            raise ValueError(f'{name}: expected shape {(size,)}, got {x.shape}')
    incoming = trivial_omega(arr.in_dim_M) if incoming is None else incoming
    return _inspect(arr, q, state, m, incoming, n, xi)


def _read(parameter, state, duration, round_index):
    if hasattr(parameter, 'read1'):
        if round_index not in (0, 1):
            raise ValueError('A two-round integrator has rounds zero and one')
        parameter = parameter.for_duration(duration)
        return (parameter.read1, parameter.read2)[round_index](state)
    if hasattr(parameter, 'reads'):
        if duration is not None:
            if parameter.durational is None:
                raise ValueError('This integrator has no duration law')
            parameter = parameter.durational(duration)
        return parameter.reads[round_index](state)
    if round_index:
        raise ValueError('A one-round integrator has only round zero')
    return parameter.position_at(state, duration)


def _smooth_inputs(fn, q, m, n, results, inward):
    """Trace numeric ports as whole values. A function-valued port stays opaque."""
    from ._explorer_inputs import coordinate_inputs
    parts = (q, *m, *n) if inward else (q, *m)
    ports = [('q', i) for i in range(len(q))]
    ports += [('mo', i) for i in range(len(m))]
    if inward:
        ports += [('ni', i) for i in range(len(n))]
    rows = []
    for i, result in enumerate(results):
        try:
            arrays = tuple(jnp.asarray(v) for v in parts)
            if any(callable(x) for x in jax.tree.leaves(result)):
                raise TypeError('function-valued output')
            source = ports[:len(q)]
            for port, x in zip(ports[len(q):], arrays[1:]):
                source.extend([port] * x.size)
            def one(*args):
                values = tuple(args[1:1+len(m)])
                if inward:
                    return fn(args[0], values, tuple(args[1+len(m):]))[i]
                return fn(args[0], values)[i]
            supports = coordinate_inputs(one, arrays, [np.size(x) for x in jax.tree.leaves(result)])
            inputs = set()
            uncertain = False
            for leaf in supports:
                for indices, bound in leaf:
                    inputs.update(source[j] for j in indices)
                    uncertain |= bound
            rows.append(([port for port in ports if port in inputs], uncertain))
        except (TypeError, ValueError):
            rows.append((ports, True))
    return rows


def inspect_smooth(arr: SmoothArrangement, state: Any = None, *, values: Any = None,
                   incoming: Any = None, inputs: Any = None, loads: Any = None,
                   duration: Optional[float] = None, round_index: int = 0) -> Dict[str, Any]:
    """Observe a smooth-interface arrangement on the plots its runner presents.

    ``values`` is the actual (plot, point) pair, ``incoming`` its 1-form,
    ``inputs`` a tuple, and ``loads`` a callback from the published target
    position to a covector on that position's plot. A field wire has no single
    scalar load: returned arrays are explicitly *components on these plots*.
    Supply samples via a scene adapter for model-specific field animations.
    Opaque smooth primitives are displayed in the internal-state/coupling
    factorization; the display never invents their hidden subarrangements.
    """
    from . import smooth
    if values is None:
        if arr.M_out:
            raise ValueError('Supply the source plot and point for a nonempty smooth interface')
        values = smooth.point(())
    if inputs is None:
        if arr.N_in:
            raise ValueError('Supply inputs for a nonempty smooth interface')
        inputs = ()
    incoming = smooth.zero_form if incoming is None else incoming
    state = arr.P.init if state is None else state
    q = _read(arr.P, state, duration, round_index)
    forward, backward = smooth.interpretation(arr)(q)
    position = forward(values, incoming)
    (out_plot, out_at), returned = position
    xi = jax.tree.map(jnp.zeros_like, out_at) if loads is None else loads(position)
    inclination, source_load, in_m = backward(values, incoming, xi, inputs)
    p, u = values
    # Input-side covectors are measured on the constant point plot unless an
    # adapter supplies a richer plot. These zero-dimensional components are
    # not evidence that the full covector vanishes.
    m, out = p(u), out_plot(out_at)
    dependencies = {'no': _smooth_inputs(arr.out_f, q, m, inputs, out, False),
                    'mi': _smooth_inputs(arr.in_f, q, m, inputs, in_m, True)}
    return dict(transfer_inputs=dependencies, label=arr.label or 'Smooth arrangement', configuration=q, state=state,
                inclination=inclination, m_out=m, m_load=source_load,
                m_in=in_m, m_in_load=None, n_out=out, n_load=xi,
                n_in=inputs, n_in_load=None, children=[], operation='smooth primitive; loads on presented plots')


def _inspect(arr, q, state, m, incoming, n, xi):
    forward, backward = interpretation(arr)(q)
    out, returned = forward(m, incoming)
    inclination, source_load, in_m = backward(m, incoming, xi, n)
    record = dict(label=arr.label or 'Arrangement', configuration=q, state=state,
                  inclination=inclination, m_out=m, m_load=source_load,
                  m_in=in_m, m_in_load=incoming(in_m), n_out=out, n_load=xi,
                  n_in=n, n_in_load=returned(n), children=[], operation='primitive')
    tree = arr.construction
    if not tree:
        record['maps'] = _array_maps(arr, record)
        return record
    operation, a, b = tree
    record['operation'] = operation
    qa, qb = q[:a.P.Q_dim], q[a.P.Q_dim:]
    sa, sb = state
    if operation == 'compose':
        mid, mid_form = interpretation(a)(qa)[0](m, incoming)
        rb = _inspect(b, qb, sb, mid, mid_form, n, xi)
        ra = _inspect(a, qa, sa, m, incoming, rb['m_in'], rb['m_load'])
    elif operation == 'tensor':
        mi, mo, ni, no = a.in_dim_M, a.out_dim_M, a.in_dim_N, a.out_dim_N
        ia = lambda z: incoming(jnp.concatenate([z, in_m[mi:]]))[:mi]
        ib = lambda z: incoming(jnp.concatenate([in_m[:mi], z]))[mi:]
        ra = _inspect(a, qa, sa, m[:mo], ia, n[:ni], xi[:no])
        rb = _inspect(b, qb, sb, m[mo:], ib, n[ni:], xi[no:])
    else:
        raise ValueError(f'Unknown construction {operation}')
    record['children'] = [ra, rb]
    return record


_SIDES = {'mo': ('Internal output', 'm_out', 'm_load'), 'mi': ('Internal input', 'm_in', 'm_in_load'),
          'ni': ('External input', 'n_in', 'n_in_load'), 'no': ('External output', 'n_out', 'n_load'),
          'q': ('Configuration', 'configuration', 'inclination')}


def _coordinate(side, i, size):
    name = _SIDES[side][0]
    return name if size == 1 else f'{name} {i+1}'


def _array_maps(arr, r):
    """Keep the actual pullback on each transfer branch apart from the local one-form."""
    from ._explorer_inputs import coordinate_inputs
    q, m, n = r['configuration'], r['m_out'], r['n_in']
    args = (q, m, n)
    sources = [(side, i) for side, xs in zip(('q','mo','ni'), args) for i in range(xs.size)]
    maps = []
    for side, fn, xs in [('no', arr.out_f, args[:2]), ('mi', arr.in_f, args)]:
        _, key, lk = _SIDES[side]
        size = r[key].size
        if not size:
            continue
        pullback = jax.vjp(fn, *xs)[1] if np.any(r[lk]) else None
        supports, = coordinate_inputs(fn, xs, [size])
        for i, (support, uncertain) in enumerate(supports):
            loads = np.zeros(sum(x.size for x in xs))
            if r[lk][i] != 0:
                seed = jnp.zeros_like(r[lk]).at[i].set(r[lk][i])
                loads = np.concatenate([np.asarray(part) for part in pullback(seed)])
            # Custom pullbacks may act on coordinates unused in the forward calculation.
            inputs = [(sources[j], j in support, float(loads[j])) for j in sorted(set(support) | set(np.flatnonzero(loads)))]
            maps.append(dict(side=side, index=i, inputs=inputs, uncertain=uncertain))
    coeffs = arr.omega(*args)
    supports = coordinate_inputs(arr.omega, args, [x.size for x in args])
    for side, values, rows in zip(('q','mo','ni'), coeffs, supports):
        for i, (support, uncertain) in enumerate(rows):
            value = float(values[i])
            if not support and not uncertain and value == 0:
                continue
            target = sources.index((side,i))
            inputs = [(sources[j], j in support, value if j == target else 0.)
                      for j in sorted(set(support) | {target})]
            maps.append(dict(side=side, index=i, inputs=inputs, uncertain=uncertain, coefficient=value))
    return maps


def scene(record: Dict[str, Any], *, title: Optional[str] = None, time: float = 0) -> Dict[str, Any]:
    """Draw individual transfer results and the local one-form using the measured port loads."""
    nodes, wires = [], []

    def data(value):
        if value is None:
            return None
        if callable(value):
            return 'Function (supply samples through a scene adapter)'
        if isinstance(value, dict):
            return {str(k): data(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [data(v) for v in value]
        return np.asarray(value).tolist()

    def node(id, parent, label, kind, **kw):
        nodes.append(dict(id=id, parent=parent, label=label, kind=kind, **kw))

    def wire(source, target, label, value, load, **kw):
        wires.append(dict(id=f'{source}>{target}:{label}', source=source, target=target, label=label,
                          value=data(value), load=data(load), valueScale=1, loadScale=1, **kw))

    def port(id, side, i):
        return id + ('/state' if side == 'q' else f'/{side}/{i}')

    def visit(r, id, parent):
        node(id, parent, r['label'], 'box', description=r['operation'])
        for side, (name, key, lk) in _SIDES.items():
            if side != 'q':
                for i in range(len(r[key])):
                    node(port(id, side, i), id, _coordinate(side, i, len(r[key])), 'boundary',
                         side='in' if side in ('mo','ni') else 'out')
        if r['children']:
            a, b = r['children']
            visit(a, id+'/a', id)
            visit(b, id+'/b', id)

            def connect(from_id, from_side, to_id, to_side, trace, side, from_offset=0, to_offset=0):
                _, key, lk = _SIDES[side]
                for i, value in enumerate(trace[key]):
                    parent_side = to_side if to_id == id else from_side
                    if to_id == id or from_id == id:
                        offset = to_offset if to_id == id else from_offset
                        label = _coordinate(parent_side, offset+i, len(r[_SIDES[parent_side][1]]))
                    else:
                        label = _coordinate(side, i, len(trace[key]))
                    wire(port(from_id, from_side, from_offset+i), port(to_id, to_side, to_offset+i),
                         label, value, trace[lk][i])

            if r['operation'] == 'compose':
                connect(id+'/a','no',id+'/b','mo',a,'no')
                connect(id+'/b','mi',id+'/a','ni',b,'mi')
                for side, child in [('mo','a'),('mi','a'),('ni','b'),('no','b')]:
                    inner = id+'/'+child
                    src, dst = (id, inner) if side in ('mo','ni') else (inner,id)
                    connect(src,side,dst,side,r,side)
            else:
                for side in ('mo','mi','ni','no'):
                    offset = 0
                    for child, trace in [('a',a),('b',b)]:
                        inner = id+'/'+child
                        if side in ('mo','ni'):
                            connect(id,side,inner,side,trace,side,from_offset=offset)
                        else:
                            connect(inner,side,id,side,trace,side,to_offset=offset)
                        offset += len(trace[_SIDES[side][1]])
            return
        if np.size(r['configuration']):
            node(id+'/state', id, 'Internal state', 'state', value=data(r['state']),
                 configuration=data(r['configuration']), inclination=data(r['inclination']))
        if 'maps' not in r:
            opaque(r, id)
            return
        for row in r['maps']:
            side, i = row['side'], row['index']
            _, key, lk = _SIDES[side]
            label = _coordinate(side, i, len(r[key]))
            field = 'coefficient' in row
            name = ('Inclination on ' if side == 'q' else 'Load on ') + label.lower() if field else label
            op = f'{id}/map/{side}/{i}'
            node(op, id, name, 'field' if field else 'operation', possibleInputs=row['uncertain'],
                 result=row['coefficient'] if field else data(r[key][i]), dependencyView=True,
                 description='Contribution of the local inclination field.' if field else
                             ('Out-transfer.' if side == 'no' else 'In-transfer.'))
            for (src, j), argument, load in row['inputs']:
                values = r[_SIDES[src][1]]
                wire(port(id,src,j), op, _coordinate(src,j,len(values)), values[j], load,
                     argument=argument, inclination=src=='q')
            if not field:
                wire(op, port(id,side,i), label, r[key][i], r[lk][i])

    def opaque(r, id):
        # A smooth output covector is already on its presenting plot. Splitting
        # it among output values would invent coordinate loads. Keep it intact.
        smooth = r['operation'].startswith('smooth')
        plot = id+'/loads'
        description = ('Observed components on the supplied plots. An output load belongs to the whole '
                       'presenting plot.' if smooth else 'Total observed loads for this recorded primitive.')
        node(plot, id, 'Loads on the plots' if smooth else 'Observed loads', 'field', description=description,
             observations=[dict(label=label, value=data(r[key])) for label,key in
                           [('Configuration inclination','inclination'),('Internal output load','m_load'),
                            ('External output load','n_load'),('Internal input load','m_in_load'),
                            ('External input load','n_in_load')]])
        for side in ('no','mi'):
            _, key, lk = _SIDES[side]
            for i, result in enumerate(r[key]):
                label = _coordinate(side,i,len(r[key]))
                op = f'{id}/map/{side}/{i}'
                sides = ('q','mo') if side == 'no' else ('q','mo','ni')
                if 'transfer_inputs' in r:
                    inputs, bound = r['transfer_inputs'][side][i]
                else:
                    inputs = [(src,j) for src in sides for j in range(len(r[_SIDES[src][1]]))]
                    bound = True
                node(op,id,label,'operation',result=data(result),possibleInputs=bound,dependencyView=True,
                     description=('Only possible inputs could be determined.' if bound else
                                  'Coordinate dependencies of the transfer on the presented values.'))
                for src, j in inputs:
                    values = r[_SIDES[src][1]]
                    wire(port(id,src,j), op, _coordinate(src,j,len(values)), values[j], None)
                wire(op, port(id,side,i), label, result, None)

    visit(record, 'root', None)
    return dict(version=1, title=title or record['label'], root='root', time=time, nodes=nodes, wires=wires)


def write_html(frames: Union[Dict[str, Any], Sequence[Dict[str, Any]]], path: Union[str, Path]) -> Path:
    """A portable offline explorer for one snapshot or a recorded run."""
    frames = [frames] if isinstance(frames, dict) else list(frames)
    if not frames:
        raise ValueError('At least one frame is required')
    viewer = Path(__file__).resolve().parents[1] / 'viewer'
    css = (viewer/'explorer.css').read_text()
    core = (viewer/'explorer-core.js').read_text() + '\n' + (viewer/'explorer-layout.js').read_text()
    ui = (viewer/'explorer.js').read_text()
    payload = json.dumps(frames, allow_nan=False).replace('<', '\\u003c')
    html = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Arrangement explorer</title><style>'+css+'</style><main id="explorer"></main>'
            '<script>'+core+'</script><script>'+ui+'</script><script>'
            'const frames='+payload+'; let index=0,elapsed=0; const adapter={snapshot:()=>frames[index],'
            'step:dt=>{elapsed+=dt;while(index<frames.length-1){const gap=frames[index+1].time-frames[index].time;'
            'const duration=gap>0?gap:0.1;if(elapsed<duration)break;elapsed-=duration;index++;}},reset:()=>{index=0;elapsed=0},'
            'ended:()=>index===frames.length-1}; new ArrangementExplorer.View(document.getElementById("explorer"),adapter);'
            '</script>')
    path = Path(path); path.write_text(html); return path
