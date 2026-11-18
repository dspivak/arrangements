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
import jax
import jax.numpy as jnp
import numpy as np

from .interpretation import interpretation, trivial_omega


def inspect(arr, state=None, *, values=None, incoming=None, inputs=None,
            loads=None, duration=None, round_index=0):
    """Observe one interpretation round at ``state`` without advancing it.

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


def inspect_smooth(arr, state=None, *, values=None, incoming=None, inputs=None,
                   loads=None, duration=None, round_index=0):
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
    return dict(label=arr.label or 'Smooth arrangement', configuration=q, state=state,
                inclination=inclination, m_out=p(u), m_load=source_load,
                m_in=in_m, m_in_load=None, n_out=out_plot(out_at), n_load=xi,
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


def scene(record, *, title=None, time=0):
    """Convert a trace to the same version-1 protocol used by live JS adapters.

    Each primitive has unboxed dynamic state and four named boundary sides.
    Composite factors can be entered; interface connections retain both the
    value and its corresponding load. Numeric arrays remain arrays in JSON.
    """
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
    def node(id, parent, label, kind='box', **kw):
        nodes.append(dict(id=id, parent=parent, label=label, kind=kind, **kw))
    def wire(id, source, target, value, load, label, **ports):
        if not isinstance(value, (tuple, list)) or len(value):
            if isinstance(value, (np.ndarray, jax.Array)) and value.size == 0:
                return
            wires.append(dict(id=id, source=source, target=target, label=label,
                              value=data(value), load=data(load), valueScale=1, loadScale=1,
                              **ports, description=('Load components on the supplied plot; not a scalar field-wire load.'
                                           if record['operation'].startswith('smooth') else '')))
    def visit(r, id, parent):
        node(id, parent, r['label'], description=r['operation'])
        for side, label in [('mi', 'Internal inputs'), ('mo', 'Internal outputs'),
                            ('ni', 'External inputs'), ('no', 'External outputs')]:
            node(id+'/'+side, id, label, 'boundary', side='in' if side in ('mo','ni') else 'out')
        if r['children']:
            a, b = r['children']
            visit(a, id+'/a', id); visit(b, id+'/b', id)
            if r['operation'] == 'compose':
                wire(id+'/middle-out', id+'/a/no', id+'/b/mo', a['n_out'], b['m_load'], 'Intermediate outputs')
                wire(id+'/middle-in', id+'/b/mi', id+'/a/ni', b['m_in'], a['n_in_load'], 'Intermediate inputs')
                for side, child, key, lk, rev in [
                    ('mo','a','m_out','m_load',False), ('mi','a','m_in','m_in_load',True),
                    ('ni','b','n_in','n_in_load',False), ('no','b','n_out','n_load',True)]:
                    ends = (id+'/'+side, id+'/'+child+'/'+side)
                    wire(id+'/'+side, *(ends[::-1] if rev else ends), r[key], r[lk], side)
            else:
                for side, key, lk, rev in [('mo','m_out','m_load',False),('mi','m_in','m_in_load',True),
                                          ('ni','n_in','n_in_load',False),('no','n_out','n_load',True)]:
                    # A product's two coordinates are different readouts, not
                    # two copies of one wire. Show the split/join as a static map.
                    operation=id+'/'+side+'-product'
                    node(operation,id,'Assemble product' if rev else 'Read product coordinates','operation')
                    ends=(id+'/'+side,operation)
                    wire(id+'/'+side,*(ends[::-1] if rev else ends),r[key],r[lk],side)
                    for child,rr in [('a',a),('b',b)]:
                        ends=(operation,id+'/'+child+'/'+side)
                        wire(id+'/'+child+'-'+side,*(ends[::-1] if rev else ends),rr[key],rr[lk],
                             side+' · '+child,sourcePort=child if not rev else side)
        else:
            node(id+'/coupling', id, 'Transfers and inclination field', 'field')
            if np.asarray(r['configuration']).size:
                node(id+'/state', id, 'Internal state', 'state', value=data(r['state']),
                     configuration=data(r['configuration']), inclination=data(r['inclination']))
                wire(id+'/readout', id+'/state', id+'/coupling', r['configuration'], r['inclination'], 'Configuration / inclination')
            for side, key, lk, rev in [('mo','m_out','m_load',False),('mi','m_in','m_in_load',True),
                                      ('ni','n_in','n_in_load',False),('no','n_out','n_load',True)]:
                ends = (id+'/'+side, id+'/coupling')
                wire(id+'/'+side, *(ends[::-1] if rev else ends), r[key], r[lk], side)
    visit(record, 'root', None)
    connected = {end for w in wires for end in (w['source'], w['target'])}
    nodes = [n for n in nodes if n['kind'] not in ('boundary','operation') or n['id'] in connected]
    return dict(version=1, title=title or record['label'], root='root', time=time, nodes=nodes, wires=wires)


def write_html(frames, path):
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
