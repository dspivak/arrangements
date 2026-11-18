# Arrangement explorer

Open `feeding-explorer.html` directly in a browser. No server, build step, CDN or
new dependency is needed. Press Play, click Receiver, then Wheels and Dragger.
The path above the picture returns to any enclosing arrangement. Shift-click
parts (or use Select parts), give the group a name, then Put a box around. Open
the new box, Remove this box, or Undo grouping. Grouping is saved locally.
Drag a part to move it; its wires reroute as it moves. A click without a drag
still enters the part. Leave clearance between parts; an overlapping move keeps
the last clear position. Positions are saved separately at each level and
survive reload. Auto arrange clears positions at the current level. Alt + arrow
keys move a focused part. Moving a part changes the diagram only.

Both styles draw individual named ports, inputs on the left and outputs on the
right. Illustrated adds library animations; Wiring shows the same arrangement
with plain glyphs. Click a port to isolate its readout and branches; click a wire
for that branch's value and returning load. Use Zoom, + / −, or Ctrl/⌘ + scroll
to enlarge a dense view (25–400%, or Fit). Pinch/scroll changes size continuously
with gesture distance, with small deltas combined into one render per frame;
it does not jump a preset on every trackpad event. Scroll zoom keeps the point under the
pointer fixed as far as the page's scroll bounds permit. Fork dots mark shared
readouts, with returning loads summed on the trunk.
Forward and backward strokes follow the same complete path to the same port.
Blue marks travel forward; amber/red marks travel backward. A zero load has no
moving marks. Color indicates magnitude relative to the fixed scale supplied
for that wire. Signs and exact numbers appear on selection. An array uses its
Euclidean norm for color, with all entries retained in the inspector. Unknown
loads remain gray and say unavailable; they are never silently zero.

## Connecting a runnable example

The shared view does not know about feeding. Supply an adapter:

```js
const adapter = {
  snapshot() { return observedScene; }, // read-only, synchronous, same instant
  step(seconds) { /* advance the actual model */ },
  reset() { /* restore its initial conditions */ },
  ended() { return false; },            // optional, e.g. end of a recording
  dispose() {},                         // optional cleanup
};
const view = new ArrangementExplorer.View(element, adapter, {
  renderers,                           // optional model-specific drawings
  storageKey: 'unique-example-layout-v1',
});
view.refresh();                        // after an external model update
view.enter('receiver');                // also usable by an existing animation
view.destroy();
```

Load `explorer-core.js`, `explorer-layout.js`, and `explorer.js` in that order,
with `explorer.css`. The layout module is dependency-free and also runs under
Node for geometry tests.

`step` must consume the supplied model duration. Replay adapters should respect
frame timestamps. Multi-round adapters should emit a complete round's
observations, identifying the round in their description. For a live model
that already owns its clock, omit `step` and invoke `refresh` from that clock;
the view disables its playback controls in that mode.

A snapshot uses this versioned protocol:

```js
{
  version: 1, title: 'Example', root: 'root', time: 1.25,
  nodes: [
    { id: 'root', parent: null, label: 'Example', kind: 'box' },
    { id: 'state', parent: 'root', label: 'Temperature', kind: 'state',
      value: 3, configuration: 3, inclination: -0.2 },
    { id: 'reader', parent: 'root', label: 'Reader', kind: 'box' },
    { id: 'field', parent: 'reader', label: 'Inclination field', kind: 'field' },
  ],
  wires: [
    { id: 'temperature-to-reader', source: 'state', target: 'field',
      label: 'Temperature', value: 3, load: -0.2,
      valueScale: 10, loadScale: 1, valueUnit: 'K',
      inclination: true },
  ],
}
```

IDs must be stable, unique within their list, and resolve to a connected rooted
hierarchy. Wires refer to the actual deepest endpoints. `Model.project()`
computes the visible boundary ports when entering or grouping. Shared values
are displayed once at a boundary and their returning contributions add.
Port identity defaults to the original source node and wire label, preserved
through projection. Equal numbers from different readouts remain different
ports. Use optional wire `sourcePort` for a stable output-port name independent
of its display label, and `targetPort` for an explicit input-port name. Otherwise
each distinct incoming readout gets its own port. An array-valued readout is one
port; split its coordinates into separately identified wires when appropriate
to the actual interface. Port identity must not depend on the current value.
`kind` is `box`, `state`, `operation` (static map), `field` (inclination field),
or `boundary`. An internal integrator is drawn as state, without a box or a
fictitious input port. A state-to-coupling observation carries its configuration
and total inclination. `value` and `load` may be scalars, arrays, display text,
or `null` for unavailable; numeric observations must be finite. Functions and
other structured values need a meaningful sample/summary supplied by the model;
the viewer cannot assign them a canonical scalar magnitude.

Optional node fields include `description`, `renderer`, and layout hints
`column` and `row`. Hints are ignored when regrouping would cause overlap.
The shared layout sizes parts for their port counts, aligns boundary ports,
uses horizontal endpoint tangents, and routes around other parts and their
titles. Unboxed states have visible readout marks; wires meet those marks.
An adapter can append application data to a node or snapshot; feeding supplies
its state and individual spring stretches this way. Grouping is a view operation:
it leaves model state, values, loads and dynamics untouched. It neither deletes
parts nor constructs a new physical coupling.
`ArrangementExplorer.layout(project, {placements})` also accepts named `{x,y}`
centers in SVG units. It rejects overlapping parts, expands the drawing as
needed, and preserves port ordering when parts move. The view persists these
placements by scope alongside its grouping, without writing them into the scene.

## Animation hooks

Each renderer is a function receiving `{group, node, scene, svg, width, height}`.
Draw within the supplied local SVG group centered at the origin. The `svg`
helper creates elements with attributes and text safely. Renderers must not
step or mutate the model. Nodes name their renderer with a string, and the
registry can also provide defaults keyed by `kind`. `feeding-animations.js`
is the example registry. Its wheels turn from the actual wheel state, its
springs show their own stretch, and a casing uses its own food and height.
Custom canvas or WebGL drawing can be mounted in an SVG `foreignObject`; the
renderer is recreated on each observation. It may return a cleanup function,
called before the next rendering and when the view is destroyed.
Keep a default icon within about 130 by 100 local units; the host scales and
places this group within the part, reserving its port labels and readout marks.
An animation does not need to implement port positioning or wire geometry.
For persistent canvas animation, keep the canvas in the host and call
`view.enter(id)` on its hit targets, drawing from the same adapter snapshot.

## Python library bridge

```python
from arrangements.explore import inspect, scene, write_html
observation = inspect(arrangement, state, inputs=external_values,
                      loads=external_output_loads, duration=step_duration)
write_html(scene(observation, time=t), 'example-explorer.html')
```

`inspect` uses the existing polynomial interpretation and never advances the
integrator. Array arrangements retain their actual `compose_seq` and tensor
construction trees automatically. Opaque primitives expose their interface
sides, internal state and inclined coupling; internal parts cannot be recovered
from arbitrary Python closures. Add a scene adapter to expose those parts.
All existing constructors still work: the optional `construction` metadata has
no role in their dynamics, equality or printed representation.

For open internal interfaces, pass their `values` and the actual `incoming`
one-form. Omitted Cartesian boundary data means zero. `round_index` observes a
many-round integrator using that round's actual carrier state. Smooth-interface
arrangements use `inspect_smooth` through the same `inspect` entry point: pass
the source `(plot, point)`, its one-form, the external input tuple, and a `loads`
callback that supplies the covector on the published output plot. Loads there
are plot components, not scalar field-wire forces. Unspecified input-side
components are marked unavailable. Model-specific adapters may supply richer
plots and sampled field animations. This is an API for every kind of runnable
example; this run equips feeding and tests the array, multi-round and smooth
bridges, rather than claiming all examples have already been adapted.

`write_html` also takes a list of timestamped scenes and packages the scripts
and styles into one offline file. It refuses nonfinite JSON. The result has the
same navigation, grouping and inspection controls as the live example.

## Feeding model and checks

`arrangements/feeding.py` builds sources and receiver using library tensor and
composition, with source readouts in the order food, opening, access, height,
red, place. The patrol and the attachment memories belong to the receiver;
the ground memories belong to the sources. `feeding-arrangement.js` assembles
the corresponding displayed contributions and integrates their sum. It is
checked against both the independently maintained `feeding-model.js` and the
library's actual `Phi` / `PhiK` steps. The six source-output loads agree at five
points in the 60-unit episode. The generic explorer has no feeding constants.
The mouth contains an opening interaction and a digestion interaction per
casing, as in the feeding note. The names identify the affected casing. This
factorization does not assert a separate anatomical tool for each casing.
The demonstration is the patrol receiver. The main `feeding.html` has since
gained a food-learning receiver in another run and links back to this example.
Its live animation has not yet been connected to this observer; that adapter
must expose its meal and learned spatial memories as well as the mechanics.

Run `node --test viewer/explorer.test.js viewer/explorer-layout.test.js`, `node viewer/feeding-model.test.js`,
and `.venv/bin/python -m pytest tests/test_explore.py -q`. The browser regression
also checks click-through, grouping/undo, both styles, playback and mobile
overflow; the book run records its script and results. Geometry checks visit
every expandable feeding level and nested groups, testing actual port counts,
endpoint continuity, horizontal tangents, obstacle clearance, and identity of
readouts whose values happen to coincide. Browser checks compare rendered SVG
path endpoints with rendered port and fork coordinates in both styles.
They also exercise real pointer dragging, persisted positions after reload,
ordinary click-to-enter after a drag, scroll zoom, and restoration of layout.

The browser reference permits an eye cost of zero. Its chosen eye rate then
persists with zero payment, a limiting convention; the Python rod-segment
integrator requires positive capacity. This check is about the stipulated
mechanical model. It establishes no biological or social claim.

## Next run: apply the explorer to the remaining examples

Inventory runnable examples first. For each, use `inspect` where it supplies
the required observations; supply an adapter for internal factorizations or
special value types it cannot infer. Reuse the shared view and supply a renderer
registry, preserving existing controls and animation. Compare the adapter's
observations and trajectories against the actual library run, including a
nonzero backward load. Check navigation, boundary values, grouping, and an
offline export. Keep unsupported observations explicit; do not fabricate
subarrangements or use state rates in place of port loads.
