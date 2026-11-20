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
Each wire can display two tracks that meet at its ports. A coordinate that only receives a load shows its amber track. Blue arrows point from source
to target. Amber arrows point back toward the source. During playback, marks
travel along each nonzero track in its arrow's direction. The arrows stay
visible when paused and with reduced motion. Zero is dashed and missing numeric
observations are dotted gray. Use **Wires → Loads only** to follow returning
loads in a crowded diagram. Selecting a branch shows its signed reading.
Selecting a fork shows its total and the contributions from its branches.
Color uses magnitude relative to the wire's scale (the largest branch scale
on a shared trunk). Arrays use their Euclidean norm for color and keep all
components in the inspector. A wire that carries a list of numbers is a hose,
several wires bundled: click its mouth, the oval with a number beside it, to
unwrap that end into one port per number, and again to wrap them.

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
the view disables its playback controls in that mode. To use that clock's own
controls, pass `options.playback` with a `controls` element, a `playing` getter
and `setPlaying(on)`. They stay available at every depth.

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
each distinct incoming readout gets its own port. One port carries one number: a
readout of n numbers is n ports. A wire whose `value` is a list of n numbers is
those n wires bundled, a hose (see "Hoses" below); its optional `coordinates`
names them. Port identity must not depend on the current value.
`kind` is `box`, `state`, `operation` (static map), `field` (inclination field),
or `boundary`. An internal integrator is drawn as state, without a box or a
fictitious input port. A state-to-coupling observation carries its configuration
and total inclination. `value` and `load` may be scalars, arrays, display text,
or `null` for unavailable; numeric observations must be finite. Functions and
other structured values need a meaningful sample/summary supplied by the model;
the viewer cannot assign them a canonical scalar magnitude.

Selecting a single-result map shows its inputs. Select an input to inspect its
source, including through intermediate boundary ports. A field with a single
known result can supply `dependencyView: true`, `result`, `resultLabel` and
`resultUnit`. A wire marked `argument: false` carries its returning load without
appearing in the result's argument list. Use `possibleInputs: true` for a
conservative list when coordinate dependencies cannot be determined.

## A field's 1-form

A `field` node may carry `form`, its inclination field written once (`explorer-forms.js`,
loaded before `explorer.js`; a page without it draws as before):

```js
form: {
  letters:   {'Battery': 'b', 'Look distance': 'r'},   // label of an entering wire -> its letter
  constants: {c_see: 0.002},                            // optional numbers, by letter
  where:     [['live', 'θ(b/b_0)']],                    // optional abbreviations, in order
  terms:     [{c: 'c_see*look_r', h: 'r'}],             // the form is the sum of c d(h)
}
```

The form is the sum of the terms `c d(h)`, the shape of `feeding_parts.pushes`: a push is
`{c: 'F', h: 'x'}`, a cost `k` is `{c: '-1', h: k}`, a Petri transition is its rate with
`h` the signed sum of what it gives and takes, and no terms is the zero form. `c`, `h` and
the abbreviations use the arithmetic Python and JavaScript share (`+ - * / **`, calls),
with Python's precedences. A name may be any letters, Greek included; an underscore starts
a subscript. A page adds the functions it needs to `ArrangementForms.functions`.

The view writes each letter after its terminal's name, the form under the mark (cut when
long, whole in the sidebar with its abbreviations and constants), and at every snapshot
evaluates the form at the values on the mark's entering wires and compares each load with
the wire's `load`. A disagreement is drawn red and named in the sidebar, so a written
formula is never trusted. Every entering wire that carries a load needs a letter, and a
lettered wire carries one number. `node --test viewer/explorer-forms.test.js` checks every
written form through a feeding meal and lists the marks that have none.

**A mark is wired to what its formula names.** A coefficient that reads several wires is
not written on the field's mark under one name: it is a static map of its own, outside the
field, and the mark hears the one wire that carries its value. So a motor at a rate has the
form `c u dq − c u² db` and hears `u`; what the rate reads enters the rate's map.

An `operation` node may carry `law`, the static map's formula, with a form's `letters`,
`constants` and `where`; the map's value is the last abbreviation:

```js
law: {
  letters:   {'Battery': 'b', 'Red at gaze': 'seen', 'Look distance': 'r', 'Search pull': 'pull'},
  constants: {b_0: 1, k_τ: 0.16},
  where:     [['live', 'θ(b/b_0)'], ['free', 'θ(1 - 2*seen)'], ['τ_drive', 'live*(k_τ*seen*r + free*pull)']],
}
```

The view writes each letter after its terminal's name and the law in the sidebar, and
compares it at every snapshot with the value on the wire leaving the map
(`ArrangementForms.lawCheck`); the same test file checks every law through the meal.
While the map's circle is the one chosen (clicked), its law is also written on a card
under the circle. A map also sends the load on the wires leaving it back along the wires
entering it, each times its law's derivative there; the view checks those loads too
(`ArrangementForms.pullbackCheck`), so a load drawn on a map's input is never trusted either.

**A spring is drawn as a spring.** A `field` node with `glyph: 'spring'` is the field
minus d of half a stiffness times the squared difference of its two entering wires. It
has no circle and no box: the two wires are joined by a bend (the cap of string
diagrams), the bend's upright stretch drawn as a red resistor, with `stiffness` (a
letter, such as `'κ_w'`) written over the top and its `form`, the cost, under it. The
two loads are equal and opposite, on the wires. A value another part needs from it (the
load it puts on one end) is a static map of its own, with a `law`.

Two more of the kind. A spring with a **third terminal** (`third`, the label of the wire
that sets its stiffness) has that wire on a row between its two ends, meeting the resistor
from the side at an arrowhead; its stiffness is a constant times the value on that wire,
and it pushes back there by minus half the constant times the squared stretch. A spring to
an **anchor** (`anchor: true`) has one wire, and its resistor ends at a hatched bar, the end
held at 0. In feeding, each Contact's springs are drawn so (Feeding, Spatial coupling,
Contact 1, Contact springs): static maps compute the stretch and the gate's four factors
and their product, a tip spring and a damper's spring each take the gate on a third
terminal, and the damper's return spring goes to an anchor. The loads the gripper, the
lever and the latch motor are told stand in a box of their own, since a motor must be sent
the load on its store as a value.

## Hoses

**One port per number; wires that are bundled are drawn as a hose.** A wire whose
`value` is a list of n numbers (n > 1) is n wires, one per number, bundled. It is
drawn as a hose: a ribbed tube around the value track and the load track, which are
colored by the size of the whole list. The hose's **mouth**, an oval with the number
n beside it, is the icon that wraps and unwraps it. Each end of a hose is in one of
two states, and clicking the mouth changes that end to the other:

- **wrapped**: the part has one hose port there, drawn as the mouth;
- **unwrapped**: the part has n ports there, one per number, each named by the
  wire's label and its `coordinates` entry (or its number in the list). Their
  strands, each a wire with its own value and its own returning load, run to the
  mouth, which stands a little way off the part; the hose goes on from the mouth,
  and a shared readout forks just past it.

A list of at most four numbers is unwrapped until the viewer wraps it; a longer one
is wrapped until the viewer unwraps it (the view's `hoseAbove` option, 4 by
default). A hose stays wrapped where it crosses the boundary of the opened box and
at a spring's terminal. The choice is saved with the positions, level by level, and
like them it changes the drawing only: the hose's value is the list of its strands'
values and its load the list of their loads (`ArrangementExplorer.dimension`,
`coordinate`, `strand` in `explorer-core.js`; `hoseEnd` in `explorer-layout.js`).
One port per number is the rule and a hose the exception: a hose is for a list
that is one thing held as several numbers, such as a function on a line held as
its coefficients. The wire's optional `standsFor` says what that thing is
(feeding's stores of 19 numbers give `'ℝ^[−9, 9]'`, the functions on the displayed
line), and the mouth writes it after the count, the part after `^` raised.
A model whose part reads out several numbers either gives each its own wire or
sends the list on one wire; the explorer never joins separate wires into a hose.

Optional node fields include `description`, `renderer`, and layout hints
`column` and `row`. Hints are ignored when regrouping would cause overlap.
Static maps and fields use compact circles. Their labeled terminals fan into the mark. Additional arguments extend the rows vertically. The shared layout aligns boundary ports,
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
In Illustrated view, opening a box also draws its own renderer at the top,
above its contents, from the same snapshot. Breadcrumbs restore the enclosing
box's animation. A scope with no renderer shows its diagram on its own. A page
that already draws a whole-model animation can pass its DOM element as the
`rootAnimation` option. The explorer displays that element at the root in
Illustrated view. Keep shared controls outside this element.
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
construction trees automatically. Every Cartesian transfer result and local
field coefficient has its own mark. Tensor factors connect coordinate by
coordinate. The diagram follows data flow through array arithmetic, slicing
and coordinate routing, so an input remains visible when its present
derivative is zero. Unknown coordinate rules are labeled **Possible inputs**.
The transfer branches carry the actual pullbacks from the model. A field
coefficient can supply a load on a coordinate without reading its value.
Its input list shows the arguments of the coefficient. These marks describe
the computations inside the primitive. Named internal parts require a
construction tree or a scene adapter.
All existing constructors still work: the optional `construction` metadata has
no role in their dynamics, equality or printed representation.

For open internal interfaces, pass their `values` and the actual `incoming`
one-form. Omitted Cartesian boundary data means zero. `round_index` observes a
many-round integrator using that round's actual carrier state. Smooth-interface
arrangements use `inspect_smooth` through the same `inspect` entry point: pass
the source `(plot, point)`, its one-form, the external input tuple, and a `loads`
callback that supplies the covector on the published output plot. Loads there
remain together under **Loads on the plots**. Numeric transfers show their
traced inputs, and function-valued ports show the possible arguments. Unspecified input-side
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
red, place. The receiver owns its eye, wheels and search. Each casing owns its ground damper memory. The
spatial coupling owns the stroke and latch of each contact. It connects the
same seven receiver inputs and seven outputs to any number of six-output
casings. Enter Receiver, then Mouth, to see its one opener and one digestor.
When a second casing comes within reach during a grip, the first contact
remains engaged until release. An exact tie with a free grip waits for a
difference in the contact scores.

`feeding-arrangement.js` assembles the displayed contributions and integrates
their sum for the patrol reference. Its six source-output loads agree with
the library composition at five points in the 60-unit patrol episode.
For the learning cart, the observer reads its live stores and motor rates.
The learning regression compares the displayed field, mouth flows and memory
loads with `food-learning.js` through a meal and a mid-grip casing addition.
The generic explorer has no feeding constants.
Spatial coupling shows the sums of the individual red signals, contact loads,
food within grip, opening rates, and food taken. Each Contact computes its own
contributions. When casings compete, each latch reads its own offer, the grip
and battery, the comparison of contact scores, and the other latches. The
opening and eating contributions have separate inputs. Food memory and
search likewise shows one rate map for each of the nine stores. Select a
map or follow its output port to inspect its immediate inputs. Click an input
in the sidebar to follow it to its source. These wires
describe dependencies for all configurations, including during a pause when
an individual input happens to have no effect.

The outer Illustrated view shares its scene, plots, playback, and editing
controls with `feeding.html` through `feeding-view.js`. Its clock steps the
food-learning receiver. The explorer observes that outing on each display
update and on every edit or pause. Enter Receiver → Food memory and search
to inspect the meal, destination, search clock, spatial stores and their
motors. Enter Receiver → Mouth to read the held casing and the opening and
eating rates at the same instant. The patrol adapter remains available for
the earlier model and its independent field checks.

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
