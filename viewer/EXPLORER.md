# Arrangement explorer

Open `feeding-explorer.html` directly in a browser. No server, build step, CDN or
new dependency is needed. Press Play, click Seeker with a slot, then Body and motors
(`?filler=robot` puts a whole robot in the seeker's slot; `?scope=ID` opens a level).
Since 2026-11-22 the page is read from the library composite: see "Feeding, read from
the composite" below. The earlier hand-typed graph, `feeding-arrangement.js`, and its models
`feeding-model.js` and `food-learning.js` were retired on 2026-11-23; the node tests run on the
composite's scene.
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
to target. Amber arrows point back toward the source, or away from it when the load
is negative (below). Each track has one color,
blue or amber, and carries packets that travel in its arrow's direction during
playback. **A load's size is the packet rate**: packets are always 60 pixels
apart, so the larger the load, the faster they move and the more pass a point per
second. Against the wire's scale s (the largest branch scale on a shared trunk),
each tenfold step in a load doubles its packets' speed, from the slowest at
x ≤ s/1000 to the fastest at x ≥ 100 s: over a feeding run the nonzero loads span
ten decades, and a speed proportional to the load would leave most of them still,
while at 1.5 per decade the meal's load on the battery (about 12) moved hardly
faster than the wheels' drain (about 0.7). **A load's sign is the packets' direction
of travel**: a load on a wire that carries one number is one real number, and a
positive one, which inclines the source's value up, travels back toward the source,
while a negative one travels away from it, and the load's arrowhead turns round
with it. So the meal's load feeds into the battery while the grip's and the wheels'
drains run out of it, and the map from a load to its packets' velocity is odd,
v(−x) = −v(x). A hose's load, a list of numbers, has no one sign: its packets
travel back and show the size of the list, and each strand of an unwrapped hose
shows its own sign. Each track keeps how far its packets have gone, so they
speed up, slow down and turn round in place. Speed and direction are seen only during playback: paused, or with
reduced motion, the packets hold still and a load is read by selecting its
wire. A
load lives in a cotangent space, whose zero is canonical, so a zero load has no
packets on its plain amber track. A wire into a field mark whose written form
reads it and never pushes it (its letter occurs in no h of Σ c dh) has a load
that is zero at every state, and draws no amber track, as a wire whose value
means nothing draws no blue one (`pushes` in `explorer-core.js`). A value's size depends on where a chart puts zero, so
value packets move at one steady speed, and their shade shows the value: all the
beads on a wire share one shade, light at the lowest value the wire has carried
since the start and dark at the highest. That place in the range is unchanged by
a change of units or of zero, which is what a value allows. A hose's value, a
list, takes the middle shade. `packets.valuePackets = 'size'` in
`explorer-core.js` codes values' speed like loads. Missing numeric observations are
dotted gray. Use **Wires → Loads only** to follow returning loads in a crowded
diagram. Selecting a branch shows its signed reading. Selecting a fork shows its
total and the contributions from its branches. Arrays use their Euclidean norm
for the rate and keep all components in the inspector. A wire that carries a list of numbers is a hose,
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

**Everything in a formula is explained** (David, 2026-11-23: "either i can see it coming in from the left
in the diagram … or it needs to be explained in words"). A name is the letter of a wire entering the mark,
or it has `words`, and a click on the mark or on its formula opens the side panel, where a line gives its number
or formula and those words (David, 2026-11-23, of those lines drawn under each formula: "all that text is too much there"); the functions
beyond the standard ones have theirs in `ArrangementForms.functionWords`, added beside the function. A letter
whose wire is a list (`lists`, its length) is written with the index j and the sum over j is written; a form
with no list has no Σ. `Forms.unexplained(form)` lists what is left over, and the feeding test holds it
empty; in the library, `feeding_parts.pushes` refuses a formula with a name that has neither.

The view writes each letter after its terminal's name, the form under the mark (cut when
long, whole in the sidebar), and at every snapshot
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

**A motor is drawn as a motor.** A `box` node with `glyph: 'motor'` is a motor, a stateless
part that pushes a load on a store and charges the battery the work of that push: two
pushes that share a rate. It has no box. It is one mark, a picture of a motor standing
upright, its finned body on the fill of a field's mark and outlined in the load red of the
spring's resistor, with its name over it. The picture says what runs it and what it runs.
The node names two of its wires: `turns`, the label of its store's wire, and `powered`,
the battery's. The first has the top row and ends at the pulley on the motor's shaft; the
second has the bottom row and ends at the plug of the motor's cord; the wires it only
hears stand between and end at the side of its body (`MOTOR` in `explorer-layout.js`). Clicked, it opens into its two pushes, whose count
is written under the circle. In feeding every motor is drawn so: the gripper, the lift,
the look and the scan, the drive, the nine memory motors and each Contact's latch motor
(in the retired hand-typed graph, `motorPart`; on the composite, every `motor_at_rate` part). Each stands in a box of its own (`drivePart`)
with the static map that computes what it is told, the Grip drive holding the grip rate and
the grip motor: the box has no outputs, so a part that needs the same rate, as the memory's
yield rule needs the grip's, reads what that rate reads and computes it (David, 2026-11-22,
779795d2: "you're right, B.").

**A Petri transition is drawn as a machine.** A `field` node with `glyph: 'transition'` is a
Petri transition: one term, a rate times d of a signed sum of stores with fixed weights
(`feeding_parts.term_kind`). It has no circle. It is one mark, a picture of David's account of
it (2026-11-19: "a conveyor belt that takes things of input-types and they go into a machine
and out comes a conveyer belt of things of output type, and the machine runs at some rate"):
a small red box with a gear in it, on the fill of a field's mark, with its name over it and
its `form` under it, clickable as on any field. Each store's wire comes in level from the
left. The wire of a store it takes from stands above the machine, and a red arrow runs from
it into the machine's top; the wire of a store it gives to stands below, and a red arrow runs
from the machine's bottom to it. So an arrow points the way the load's packets travel on its
wire. A weight that is not 1 is written over its arrow. A wire the rate only hears ends at
the machine's side, behind its black wall. A click on the machine is a click on the mark.

The arrows and their weights are not sent: the view reads them off the node's written form
(`transitionArcs` in `explorer.js`, the weight of each lettered wire in `h`), the form it
checks against the model's loads at every snapshot, and the layout gives each wire its row
(`TRANSITION`, `transitionShape` in `explorer-layout.js`). A node whose form is not of that
shape, or that has none, is drawn as a plain field's circle.

**A transition cut in two by a box's boundary is two halves**, each a `field` node drawn as a
machine with only its own arrows; the wire that carries the rate into each half is what joins
them, so nothing more is drawn (David, 2026-11-23, of a dashed arrow with words that marked the
open side: "why an extra wire when we have one already"). In feeding the mouth's taking is cut
so by the seeker's boundary: "Mouth: settle" (in Mouth) gives to the chamber and the husk
collector, takes from the mouth's battery and hears the rate on the wire "Taken"; each
casing's "Push: taking" (in its handling box, in the spatial coupling) takes from that casing's
food at the rate on its wire "Push", and those rates are summed into "Taken". The library says
a part is a half in its meta (`half`, `feeding_composed.TAKING_SETTLES` and `TAKING_REACHES`),
since a half's own form need not be a transition by `term_kind`; the exporter reads it only to
draw the half as a machine. (The viewer still draws an open side, a dashed arrow to a break
with words, for a node that sends `open` and `openWords`; no exporter sends them.)

A bare `push` whose target the wiring computes as the difference of two wires (spilling,
`pile − food`; laying out) is joined to those two wires themselves and written `F d(x − y)`,
so it is a transition by `term_kind` and is drawn as one; one whose target is minus a wire
is written `F d(−x)` (`explore_composite._pushes_side_by_side`). `node --test
viewer/explorer-transition.test.js` checks the arrows against the form, the rows, and a
half's open side.

**A load-free input port carries a black wall.** An input port is load-free when no mark
its wire enters, however deep inside the part, can push it: each such mark has a written
form that reads the wire and never pushes it (`carriesLoad` in `explorer-core.js`, from
`pushes`). There the port's ring is replaced by the book's mark, a short thick black bar
across the wire (Ch 3: "We draw a black wall at each input port where an arrangement is
load-free"), and the wire has no amber track, at every level. So a closed motor shows a
wall at the rate or the torque it is sent and at the other load on its store, and no wall
at its store and the battery, which it pushes; opened, its two pushes show the same walls
at their own terminals, and the opened box's boundary shows none. A static map has no
wall: it sends a load back through its law's derivative. `node --test
viewer/explorer.test.js` checks, for every feeding motor, that the walled terminals are
exactly the ones it only hears.

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
titles. **Two wires meet only where they cross, at a point**, never along a
stretch: each wire is a rounded polyline, and wires that share a channel are
spread into lanes 12 px apart, ordered by which way each turns at its two ends
so that they cross as little as possible; a long stretch at a port keeps only
16 px there, so branches of one fork and wires entering ports in a row part
soon after. Where a channel has too little room, the gaps between rows and
columns, and the margins, grow and the parts are placed again. Before this,
58% of the feeding explorer's wire length lay within 10 px of another wire
running the same way; now about 4% does, mostly at rounded corners.
Unboxed states have visible readout marks; wires meet those marks.
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
registry can also provide defaults keyed by `kind`. `feeding-animations.js`,
the example registry for the retired hand-typed feeding graph, was removed with it on 2026-11-23 (it
is in the library's history: its wheels turned from the actual wheel state, its springs showed their
own stretch, and a casing used its own food and height).
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

## Feeding, read from the composite

`scripts/export_feeding_explorer.py` writes `feeding-explorer-data.js` (and, with `--robot`,
`feeding-explorer-robot-data.js`) from `feeding_composed.feeding_with_slot(..., learning=True)`:
the scene is `arrangements/explore_composite.diagram` of the composite's construction (a box per
fill the code names, a store per integrand, a static map per computed group of numbers, a field
mark per part with a field at the nicest kind of the rubric its `meta` records: a spring as one
spring between the wires its stretch is the difference of, a motor as one motor opened into its
two pushes, any other sum of pushes one mark per term wired only to what its traced code reads,
with its formula in their letters), and `readings(state, places, params)` and `step(...)` are the
library's own code exported by `export_step_js.javascript`. `feeding-explorer-composite.js` only
copies their numbers into the scene. Checks: `tests/test_explore_composite.py` (the marks' forms
add up to the composite's field at every store; each push mark reads what its trace reads; every
formula is the code's) and `node --test viewer/feeding-explorer-composite.test.js` (every drawn
formula gives the loads the library returned; motors' walls; every level lays out). Regenerate
the data files after any change to the feeding parts.

## The feeding page inside the explorer

The outer Illustrated view is `feeding.html`'s own view, `feeding-view.js`, which runs the one model:
the code generated from `feeding_with_slot` (`feeding-slot-N.js`, one file per count of casings, by
`scripts/export_step_js.py --slot --casings N`). On each display update the explorer's adapter takes
the view's state, places, knobs and time (the view's `observe`) and redraws; the view's controls are
the explorer's playback, and the form under the diagram adds a casing to the run. The diagram is drawn
for the data file's three casings; while the run has another count it holds its last numbers and says
so. With `?filler=robot` the explorer runs its own generated step. The page's checks:
`node viewer/feeding-slot.test.js` (whole outings on the generated code), `tests/test_feeding_page.py`
(each committed `feeding-slot-N.js` is the library's step) and `tests/test_feeding_slot_learning.py`.

Run `node --test viewer/explorer.test.js viewer/explorer-layout.test.js viewer/explorer-forms.test.js
viewer/feeding-explorer-composite.test.js` and `.venv/bin/python -m pytest tests/test_explore.py -q`. The browser regression
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
