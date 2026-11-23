"""The explorer's diagram of a composite, read from its construction, and its live readings.

Book P109 (f), act (4) (David, 2026-11-18, 6a32afbb: "the page is hand-written? it's not
derived from an IMC arrangemetn? i thought that was the running rule"). ``diagram(arr)``
flattens ``arr.construction``, the binary tree of ``compose_seq`` and ``tensor_arrangements``,
into what the code names:

* a **box** for every ``fill`` (a ``compose_seq`` whose second factor is a wiring, a
  ``transfer``), named by the wiring's label, or by the composite's own label when it was given
  one; ``send`` and ``discard`` and the tensors are no boxes; a filler put into a part's open
  slot is drawn inside that part;
* a **state** for every integrand (``rod_segment``, ``mass``);
* a **static map** (an ``operation``) for every group of numbers a wiring computes that is not
  a copy of one of its inputs or a constant; a copy is no mark at all, the wire runs on;
* a **field mark** for every part with an inclination field, at the nicest kind of David's
  rubric that the part's ``meta`` (``feeding_parts``) records: a spring as one spring, joined
  to the two wires whose difference its stretch is and to the wire its stiffness is a multiple
  of; a motor as one motor, opened into its two pushes; any other sum of pushes as one mark per
  term, each wired only to the inputs its traced code reads, with its formula in their letters;
  the zero field undrawn.

Every wire is a **branch**: one value read by one mark or map. ``live_function`` returns a
function of the state computing every branch's value and the load that its reader sends back
along it. The loads come from one reverse pass: each branch's value is perturbed by its own
``eps``, the marks' one-forms at the point are the cotangent, and the pullback to the ``eps``
is every branch's load, so the loads on a store's branches add up to the composite's own
inclination there (``tests/test_explore_composite.py``). Nothing in the diagram is typed by
hand; a formula is written only where the part's ``meta`` carries one, and that is checked
against the code (``feeding_parts.check_formulas``).

Built 2026-11-22 (Claude Opus 5.5, subagent of c5f9ecc9).
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from . import feeding_parts as fp
from ._explorer_inputs import coordinate_inputs
from .arrangement import Arrangement, zero_form

EMPTY = jnp.zeros(0)
MARK_PARTS = ("push", "pushes", "spring", "motor")


# ---------------------------------------------------------------------------
# The tree: paths, stores, primitives.
# ---------------------------------------------------------------------------


def _children(a: Arrangement):
    return a.construction[1:] if a.construction else ()


def nodes_by_path(arr: Arrangement) -> Dict[str, Arrangement]:
    """Every node of the construction tree by its path: '' the root, then 'f'/'g' (compose)
    or 'a'/'b' (tensor) at each step. A mark of ``MARK_PARTS`` is a leaf."""
    found = {}

    def walk(a, path):
        found[path] = a
        if a.construction and (a.meta or {}).get("part") not in MARK_PARTS:
            op, f, g = a.construction
            keys = "fg" if op == "compose" else "ab"
            walk(f, path + keys[0])
            walk(g, path + keys[1])
    walk(arr, "")
    return found


def _q_starts(arr: Arrangement) -> Dict[str, int]:
    """Where each node's configuration starts in the composite's: compose and tensor both put
    their first factor's first."""
    starts = {}

    def walk(a, path, at):
        starts[path] = at
        if a.construction:
            _, f, g = a.construction
            keys = "fg" if a.construction[0] == "compose" else "ab"
            walk(f, path + keys[0], at)
            walk(g, path + keys[1], at + f.P.Q_dim)
    walk(arr, "", 0)
    return starts


def _is_wiring(a: Arrangement) -> bool:
    return not a.construction and a.P.Q_dim == 0 and a.omega is zero_form and not (a.meta or {}).get("part")


def _default_compose_label(a: Arrangement) -> bool:
    _, f, g = a.construction
    return a.label == f"({g.label} . {f.label})"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "part"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


# ---------------------------------------------------------------------------
# A wiring's maps, coordinate by coordinate.
# ---------------------------------------------------------------------------


class Fn:
    """One side of a primitive wiring at ``path``: ``out`` is ``out_f(q, m)``, ``in`` is
    ``in_f(q, m, n)``. ``args`` holds a ref for each number of the arguments (q first), and
    ``kinds`` says what each number it computes is: ``('copy', j)``, ``('const',)``,
    ``('diff', j, k)`` (argument j minus argument k), ``('scale', j, c)`` or ``('op', support)``."""

    def __init__(self, fid, path, side, prim, args, sizes, box):
        self.fid, self.path, self.side, self.box = fid, path, side, box
        self.args, self.sizes = args, sizes
        fn = prim.out_f if side == "out" else prim.in_f
        self.size = int(jnp.size(fn(*[jnp.zeros(s) for s in sizes])))
        self.kinds = _classify(fn, sizes, self.size)
        zeros = [jnp.zeros(s) for s in sizes]
        self.constants = np.asarray(fn(*zeros), dtype=float)

    def support(self, i):
        k = self.kinds[i]
        return {"copy": lambda: [k[1]], "const": lambda: [], "diff": lambda: [k[1], k[2]],
                "scale": lambda: [k[1]], "op": lambda: list(k[1])}[k[0]]()


def _classify(fn, sizes, size):
    args = [jnp.zeros(s) for s in sizes]
    rows, = coordinate_inputs(lambda *xs: jnp.ravel(fn(*xs)), tuple(args), [size]) if size else ([],)
    rng = np.random.default_rng(7)
    total = sum(sizes)
    points = [rng.normal(size=total) for _ in range(3)]

    def split(x):
        out, at = [], 0
        for s in sizes:
            out.append(jnp.asarray(x[at:at + s]))
            at += s
        return out

    values = [np.asarray(fn(*split(x)), dtype=float).ravel() for x in points]
    zero = np.asarray(fn(*split(np.zeros(total))), dtype=float).ravel()
    kinds = []
    for i, (support, uncertain) in enumerate(rows):
        y = [v[i] for v in values]
        close = lambda a, b: all(abs(p - q) <= 1e-12 * max(1., abs(p), abs(q)) for p, q in zip(a, b))
        kind = ("op", tuple(support))
        if not uncertain and not support:
            kind = ("const",)
        elif not uncertain and len(support) == 1:
            j = support[0]
            xs = [x[j] for x in points]
            if close(y, xs):
                kind = ("copy", j)
            elif abs(zero[i]) <= 1e-14 and all(abs(x) > 1e-9 for x in xs):
                ratios = [a / b for a, b in zip(y, xs)]
                if close(ratios, [ratios[0]] * 3):
                    kind = ("scale", j, float(ratios[0]))
        elif not uncertain and len(support) == 2:
            j, k = support
            if close(y, [x[j] - x[k] for x in points]):
                kind = ("diff", j, k)
            elif close(y, [x[k] - x[j] for x in points]):
                kind = ("diff", k, j)
        elif uncertain:
            # The trace gave only an upper bound (a gather, say): a constant or a copy is found
            # by its values at three random points.
            # Exactly, not nearly: a rate that is 1e-20 at the sampled points is no constant.
            if all(v == zero[i] for v in y):
                kind = ("const",)
            else:
                for j in support:
                    if close(y, [x[j] for x in points]):
                        kind = ("copy", j)
                        break
        kinds.append(kind)
    return kinds


# ---------------------------------------------------------------------------
# The diagram.
# ---------------------------------------------------------------------------


class Diagram:
    """The flattened diagram of ``arr``: ``nodes`` (the explorer's boxes, states, maps and marks),
    ``signals`` (a state's readout or a map's value, by id), ``branches`` (the wires: one signal,
    or some of its numbers, read by one node, with its slice of ``eps``), ``marks`` (each field
    mark's inputs and how to compute its one-form) and ``fns`` (the wirings' maps)."""

    def __init__(self, arr: Arrangement, title: str = "Feeding"):
        self.arr, self.title = arr, title
        self.tree = nodes_by_path(arr)
        self.qstart = _q_starts(arr)
        self.nodes: List[dict] = []
        self.ids: set = set()
        self.signals: List[dict] = []
        self.branches: List[dict] = []
        self.marks: List[dict] = []
        self.ops: Dict[Tuple, int] = {}
        self.fns: Dict[Tuple[str, str], Fn] = {}
        self.fn_list: List[Fn] = []
        self.hints: Dict[Tuple[int, int], str] = {}
        self.states: Dict[str, int] = {}
        self.eps = 0
        self.boxes: Dict[str, str] = {}
        self.box_paths: set = set()
        root = self.node("root", None, title, "box")
        self._place(arr, "", root)
        self._full(arr, "", [], [])
        self._settle_labels()

    def _settle_labels(self):
        """Name each signal's wires: its own name, unless signals of one name enter one mark,
        where each is named with the box it comes from as well (and numbered if still alike)."""
        names = {sid: s["label"] for sid, s in enumerate(self.signals)}
        by_node = {n["id"]: n for n in self.nodes}

        def chain(sid):
            out, n = [], by_node[by_node[self.signals[sid]["node"]]["parent"]]
            while n is not None:
                out.append(n["label"])
                n = by_node.get(n["parent"])
            return out

        meets = {}
        for x in self.branches:
            meets.setdefault((x["target"], names[x["sid"]]), set()).add(x["sid"])
        clashing = {}
        for (_, name), sids in meets.items():
            if len(sids) > 1:
                clashing.setdefault(name, set()).update(sids)
        for name, sids in clashing.items():
            sids = sorted(sids)
            chains = {sid: chain(sid) for sid in sids}
            for depth in range(max(len(c) for c in chains.values())):
                tags = [chains[sid][depth] if depth < len(chains[sid]) else "" for sid in sids]
                if len(set(tags)) == len(sids):
                    for sid, tag in zip(sids, tags):
                        names[sid] = f"{name} ({tag[:1].lower() + tag[1:]})"
                    break
            else:
                for k, sid in enumerate(sids):
                    names[sid] = f"{name} {k + 1}"
        pattern = re.compile(r"§(\d+)§")
        sub = lambda text: pattern.sub(lambda m: names[int(m.group(1))], text) if isinstance(text, str) else text
        for x in self.branches:
            x["label"] = sub(x["label"])
        for sid, s in enumerate(self.signals):
            s["label"] = names[sid]
        for n in self.nodes:
            for key in ("third", "turns", "powered"):
                if key in n:
                    n[key] = sub(n[key])
            if "form" in n:
                n["form"]["letters"] = {sub(k): v for k, v in n["form"]["letters"].items()}

    # ---- nodes -----------------------------------------------------------
    def node(self, slug, parent, label, kind, **extra) -> str:
        base, k = slug, 2
        while slug in self.ids:
            slug, k = f"{base}-{k}", k + 1
        self.ids.add(slug)
        self.nodes.append(dict(id=slug, parent=parent, label=_cap(label), kind=kind, **extra))
        return slug

    def _place(self, a, path, container):
        """Decide the box every node's contents go into (``self.boxes[path]``)."""
        meta = a.meta or {}
        self.boxes[path] = container
        if not a.construction or meta.get("part") in MARK_PARTS:
            return
        op, f, g = a.construction
        if op == "tensor" or meta.get("wrapper"):
            self._place(f, path + ("a" if op == "tensor" else "f"), container)
            self._place(g, path + ("b" if op == "tensor" else "g"), container)
            return
        named = not _default_compose_label(a)
        if named or _is_wiring(g):
            label = a.label if named else g.label
            outer = self._node(container)["label"] if container else ""
            if outer.lower() == label.lower() and meta.get("part"):
                label = f"{label} {meta['part']}"      # the ground's contact inside the ground
            box = self.node(_slug(label), container, label, "box", **self._describe(meta))
            self.boxes[path] = box
            self.box_paths.add(path)
            self._place(f, path + "f", box)
            self._place(g, path + "g", box)
            return
        # A filler put into an open slot: drawn inside the part whose slot it fills. Parts side
        # by side composed with what couples them are drawn side by side with it.
        self._place(g, path + "g", container)
        slot = (f.in_dim_M == 0 and f.out_dim_M == 0 and (g.in_dim_M or g.out_dim_M)
                and not (f.construction and f.construction[0] == "tensor"))
        inner = self._slot_box(g, path + "g") if slot else None
        self._place(f, path + "f", inner or container)

    def _slot_box(self, a, path):
        """The innermost box of ``a`` whose domain is ``a``'s: where a filler of ``a``'s slot is drawn."""
        if a.construction and not (a.meta or {}).get("part") in MARK_PARTS:
            op, f, g = a.construction
            if op == "compose":
                inner = self._slot_box(f, path + "f")
            else:
                sides = [(x, path + k) for x, k in ((f, "a"), (g, "b")) if x.in_dim_M or x.out_dim_M]
                inner = self._slot_box(*sides[0]) if len(sides) == 1 else None
            if inner:
                return inner
        return self.boxes.get(path) if path in self.box_paths else None

    @staticmethod
    def _describe(meta):
        out = {}
        if meta.get("part"):
            kind = meta.get("kind")
            out["description"] = (f"The note's {meta['part']}" +
                                  (f"; its field's kind in the rubric: {kind} ({fp.RUBRIC.get(kind, '?')})" if kind else "")
                                  + (". Its formulas are under the two marks it opens into" if meta["part"] == "motor" else
                                     f". It is {fp.spelled_out(meta)}" if meta.get("formula") else "") + ".")
            out["part"] = meta["part"]
        return out

    # ---- signals and branches -------------------------------------------
    def signal(self, kind, node, label, width, **spec) -> int:
        self.signals.append(dict(kind=kind, node=node, label=_cap(label), width=width, **spec))
        return len(self.signals) - 1

    def branch(self, sid, indices, target, label=None) -> int:
        for b, x in enumerate(self.branches):
            if x["sid"] == sid and x["indices"] == indices and x["target"] == target:
                return b
        s = self.signals[sid]
        whole = indices == list(range(s["width"]))
        # The signal's name is settled at the end (``_settle_labels``), once it is known which
        # signals of one name meet at one mark; until then it stands as a placeholder.
        name = label or (f"§{sid}§" if whole else f"§{sid}§ " + ",".join(str(i + 1) for i in indices))
        self.branches.append(dict(sid=sid, indices=indices, target=target, label=name,
                                  eps=self.eps, width=len(indices)))
        self.eps += len(indices)
        return len(self.branches) - 1

    def hint(self, refs, name, named=True):
        """Name the maps that compute ``refs`` after the port they feed: a port the part's meta
        names wins over a numbered one, and the first named port wins."""
        self.port_count = getattr(self, "port_count", 0) + 1
        for r in refs:
            key = (r[1], r[2]) if r[0] == "fn" else None
            if key and (key not in self.hints or (named and not self.hints[key][0])):
                self.hints[key] = (named, name, self.port_count)

    def runs(self, refs):
        """Split refs into runs that one branch or one map carries: ('sig', sid, [k...]),
        ('fn', fid, kind, [i...]) of one kind, or ('const', [refs])."""
        out = []
        for r in refs:
            if r[0] == "sig":
                last = out[-1] if out else None
                if last and last[0] == "sig" and last[1] == r[1] and last[2][-1] + 1 == r[2]:
                    last[2].append(r[2])
                else:
                    out.append(["sig", r[1], [r[2]]])
            else:
                fn = self.fn_list[r[1]]
                kind = fn.kinds[r[2]][0]
                kind = "const" if kind == "const" else "op"
                last = out[-1] if out else None
                # Numbers of one map go on one wire only when they feed one named port.
                same_port = last and last[0] == "fn" and \
                    self.hints.get((r[1], last[3][-1])) == self.hints.get((r[1], r[2])) is not None
                if last and last[0] == "fn" and last[1] == r[1] and last[2] == kind and last[3][-1] + 1 == r[2] \
                        and same_port:
                    last[3].append(r[2])
                elif kind == "const" and last and last[0] == "const":
                    last[1].append(r)
                elif kind == "const":
                    out.append(["const", [r]])
                else:
                    out.append(["fn", r[1], kind, [r[2]]])
        return out

    def materialize(self, fid, coords, name) -> int:
        """The static map computing numbers ``coords`` of wiring map ``fid``: one node, its
        input branches, its value a signal."""
        key = (fid, tuple(coords))
        if key in self.ops:
            return self.ops[key]
        fn = self.fn_list[fid]
        label = self.hints.get((fid, coords[0]), (None, name))[1]
        node = self.node(_slug(label), fn.box, label, "operation",
                         description="A static map of the wiring " + repr(self.tree[fn.path].label) +
                                     ": it computes " + ("this value" if len(coords) == 1 else f"{len(coords)} numbers")
                                     + " from the wires that enter it.")
        support = sorted({j for i in coords for j in fn.support(i)})
        items = self.inputs_for(fn, support, node, label)
        sid = self.signal("op", node, label, len(coords), fid=fid, coords=list(coords), items=items)
        self.ops[key] = sid
        return sid

    def inputs_for(self, fn, support, target, name):
        """How a map's arguments are filled when ``target`` evaluates it: each argument in
        ``support`` from a branch into ``target``, each other one by its value."""
        items = [("val", r) for r in fn.args]
        refs = [fn.args[j] for j in support]
        at = 0
        for run in self.runs(refs):
            n = len(run[2]) if run[0] == "sig" else len(run[3]) if run[0] == "fn" else len(run[1])
            positions = support[at:at + n]
            at += n
            if run[0] == "const":
                continue
            if run[0] == "fn":
                sid = self.materialize(run[1], run[3], f"input of {name}")
                b = self.branch(sid, list(range(len(run[3]))), target)
            else:
                b = self.branch(run[1], run[2], target)
            for k, j in enumerate(positions):
                items[j] = ("br", b, k)
        return items

    def port_items(self, refs, target, port_name):
        """The items filling one input port of a mark from ``refs``: branches into ``target``,
        maps materialized where a ref is computed, constants by value. Returns (items, labels)."""
        items, labels = [], []
        for run in self.runs(refs):
            if run[0] == "const":
                items += [("val", r) for r in run[1]]
                continue
            if run[0] == "fn":
                sid = self.materialize(run[1], run[3], port_name)
                b = self.branch(sid, list(range(len(run[3]))), target)
            else:
                b = self.branch(run[1], run[2], target)
            labels.append(self.branches[b]["label"])
            items += [("br", b, k) for k in range(self.branches[b]["width"])]
        return items, labels

    # ---- the walk ---------------------------------------------------------
    def _fn(self, a, path, side, args, sizes):
        key = (path, side)
        if key not in self.fns:
            fn = Fn(len(self.fn_list), path, side, a, args, sizes, self.boxes[path])
            self.fns[key] = fn
            self.fn_list.append(fn)
        fn = self.fns[key]
        refs = []
        for i, kind in enumerate(fn.kinds):
            refs.append(fn.args[kind[1]] if kind[0] == "copy" else ("fn", fn.fid, i))
        return refs

    def _state(self, a, path):
        if path not in self.states:
            width = int(a.out_dim_N)
            node = self.node(_slug(a.label or "store"), self.boxes[path], a.label or "store", "state")
            self.states[path] = self.signal("state", node, a.label or "store", width, path=path)
        return self.states[path]

    def _fwd(self, a, path, mo):
        """The refs of ``a``'s outputs, from the refs of its domain's outputs."""
        meta = a.meta or {}
        if meta.get("part") in MARK_PARTS:
            return []
        if not a.construction:
            if a.P.Q_dim:
                sid = self._state(a, path)
                return [("sig", sid, k) for k in range(a.out_dim_N)]
            return self._fn(a, path, "out", [*mo], [0, a.out_dim_M])
        op, f, g = a.construction
        if op == "compose":
            refs = self._fwd(g, path + "g", self._fwd(f, path + "f", mo))
            named = _outputs_of(a)
            names = named or [(None, f"output {i + 1}", 1) for i in range(a.out_dim_N)]
            owner, at = _part_name(a), 0
            for _, name, width in names:
                self.hint(refs[at:at + width], name if named else f"{name} of {owner}", bool(named))
                # A store read out whole is named after the output it is (the casing's mass is
                # its height, the engagement the grip), once.
                r = refs[at:at + width]
                if named and r and all(x[0] == "sig" for x in r) and len({x[1] for x in r}) == 1:
                    s = self.signals[r[0][1]]
                    if s["kind"] == "state" and s["width"] == width and not s.get("renamed"):
                        s["label"], s["renamed"] = _cap(name), True
                        self._node(s["node"])["label"] = _cap(name)
                at += width
            return refs
        mo_a, mo_b = mo[:f.out_dim_M], mo[f.out_dim_M:]
        return self._fwd(f, path + "a", mo_a) + self._fwd(g, path + "b", mo_b)

    def _full(self, a, path, mo, ni):
        """Walk ``a`` given the refs of its domain's outputs and of its inputs; returns the refs
        of what it sends its domain."""
        meta = a.meta or {}
        part = meta.get("part")
        if part in MARK_PARTS:
            self._mark(a, path, ni)
            return []
        if not a.construction:
            if a.P.Q_dim:
                self._state(a, path)
                return []
            if a.omega is not zero_form:
                raise NotImplementedError(f"a primitive with its own field and no meta: {a.label}")
            return self._fn(a, path, "in", [*mo, *ni], [0, a.out_dim_M, a.in_dim_N])
        op, f, g = a.construction
        if op == "compose":
            mid = self._fwd(f, path + "f", mo)
            g_mi = self._full(g, path + "g", mid, ni)
            self._name_ports(f, path + "f", g_mi)
            return self._full(f, path + "f", mo, g_mi)
        mi_a = self._full(f, path + "a", mo[:f.out_dim_M], ni[:f.in_dim_N])
        mi_b = self._full(g, path + "b", mo[f.out_dim_M:], ni[f.in_dim_N:])
        return mi_a + mi_b

    def _name_ports(self, f, path, refs):
        """Hint the names of the maps a wiring computes for its parts' input ports."""
        if refs is None:
            return
        at = 0
        for part in _factors(f):
            named = _inputs_of(part)
            names = named or [(None, f"input {i + 1}", 1) for i in range(part.in_dim_N)]
            owner = (_part_name(part) or "").removeprefix("push: ")
            for _, name, width in names:
                # A port the code names is named alone; where two such maps meet at one mark,
                # ``_settle_labels`` adds the box each comes from.
                self.hint(refs[at:at + width], name if named or not owner else f"{name} of {owner}", bool(named))
                at += width

    # ---- marks ------------------------------------------------------------
    def _mark(self, a, path, ni):
        meta = a.meta
        container = self.boxes[path]
        if meta["part"] == "spring":
            return self._spring(a, path, ni, container)
        if meta.get("kind") == "zero" or not meta.get("terms"):
            return None
        if meta["part"] == "push" and a.in_dim_N > 2:
            return self._pushes_side_by_side(a, path, ni, container)
        inputs = meta["inputs"]
        slices, at = {}, 0
        for letter, name, width in inputs:
            slices[letter] = (at, width, name)
            at += width
        label = a.construction[2].label if a.construction else a.label
        if meta["part"] == "motor":
            parent = self.node(_slug(label), container, label, "box", glyph="motor", **self._describe(meta))
        elif len(meta["terms"]) > 1:
            parent = self.node(_slug(label), container, label, "box", **self._describe(meta))
        else:
            parent = container
        port_labels = {}
        for i, term in enumerate(meta["terms"]):
            reads = fp.term_reads(meta, i)
            read = [l for l, _, _ in inputs if l in set(reads["c"]) | set(reads["h"])]
            kind = fp.term_kind(meta, i)
            name = label if len(meta["terms"]) == 1 else term["name"]
            node = self.node(_slug(name), parent, name, "field", rubric=kind, rubricRank=fp.RUBRIC[kind],
                             description=f"An inclination field, kind ({fp.RUBRIC[kind]}) of the rubric: a {kind}."
                             + f" It reads {', '.join(slices[l][2] for l in read) or 'nothing'}.")
            items, letters, constants, split = [], {}, dict(meta.get("constants", {})), False
            for letter, name_, width in inputs:
                start = slices[letter][0]
                refs = ni[start:start + width]
                if letter in read:
                    self.hint(refs, name_)
                    port, labels = self.port_items(refs, node, name_)
                    items += port
                    if len(labels) == 1:
                        letters[labels[0]] = letter
                        port_labels.setdefault(letter, labels[0])
                    elif not labels:
                        constants[letter] = self._const_value(refs)
                    else:                  # one input of the part, several wires here: no letter
                        split = True
                else:
                    items += [("val", r) for r in refs]
            mark = dict(node=node, path=path, term=i, items=items, read=read)
            self.marks.append(mark)
            if term.get("c_text") is not None and term.get("h_text") is not None and not split:
                form = dict(letters=letters, constants=_numbers(constants), terms=[dict(c=term["c_text"], h=term["h_text"])])
                if meta.get("where"):
                    form["where"] = [list(w) for w in meta["where"]]
                # What is not on a wire is said in words; a letter sent a constant, too, by its input's name.
                said = {**{l: name_ for l, name_, _ in inputs if l in constants}, **meta.get("words", {})}
                form["words"] = {k: w for k, w in said.items()
                                 if k in constants or k in dict(meta.get("where", ()))}
                lists = {l: width for l, _, width in inputs if width > 1 and l in read and l not in constants}
                if lists:
                    form["lists"] = lists
                self._node(node)["form"] = form
        if meta["part"] == "motor":
            motor = self._node(parent)
            motor["turns"] = port_labels.get(meta.get("turns"))
            motor["powered"] = port_labels.get(meta.get("powered"))

    def _pushes_side_by_side(self, a, path, ni, container):
        """A ``push`` of width n is n pushes side by side: one mark each, ``F_j dx_j``."""
        w = a.in_dim_N // 2
        for j in range(w):
            name = f"{a.label} {j + 1}"
            node = self.node(_slug(name), container, name, "field", rubric="push of everything",
                             rubricRank=fp.RUBRIC["push of everything"],
                             description="A push F dx, kind (4) of the rubric: it pushes F on what it reads at x.")
            items = [("val", r) for r in ni]
            letters = {}
            owner = name.removeprefix("push: ")
            for letter, k in (("F", j), ("x", w + j)):
                port, labels = self.port_items([ni[k]], node, f"{'push' if letter == 'F' else 'pushed'} of {owner}")
                if port and port[0][0] == "br":
                    items[k] = port[0]
                    letters[labels[0]] = letter
            self.marks.append(dict(node=node, path=path, term=0, items=items, read=["F", "x"], mask=[j, w + j]))
            self._node(node)["form"] = dict(letters=letters, constants={}, terms=[dict(c="F", h="x")])

    def _spring(self, a, path, ni, container):
        """A spring as one mark: minus d of half its stiffness times its squared stretch,
        joined to the two wires whose difference the stretch is, and to the wire its stiffness
        is a multiple of (its third terminal), or to a constant stiffness."""
        label = a.construction[2].label
        node = self.node(_slug(label), container, label, "field", glyph="spring", rubric="spring",
                         rubricRank=fp.RUBRIC["spring"],
                         description="A spring, kind (0.5) of the rubric: minus d of half its stiffness "
                                     "times its squared stretch. Its loads on its two ends are equal and opposite.")
        k_ref, u_ref = ni
        letters, constants, items = {}, {}, []
        # stiffness
        if k_ref[0] == "fn" and self.fn_list[k_ref[1]].kinds[k_ref[2]][0] == "const":
            K = "κ"
            constants["κ"] = self._const_value([k_ref])
            items.append(("val", k_ref))
        elif k_ref[0] == "fn" and self.fn_list[k_ref[1]].kinds[k_ref[2]][0] == "scale":
            fn = self.fn_list[k_ref[1]]
            _, j, c = fn.kinds[k_ref[2]]
            third = self._end(fn.args[j], node, f"stiffness of {label}")
            letters[self.branches[third]["label"]] = "g"
            constants["κ"] = c
            K = "κ*g"
            items.append(("fn", fn.fid, k_ref[2], {j: (third, 0)}))
            self._node(node)["third"] = self.branches[third]["label"]
        else:
            third = self._end(k_ref, node, f"stiffness of {label}")
            letters[self.branches[third]["label"]] = "k"
            K = "k"
            items.append(("br", third, 0))
            self._node(node)["third"] = self.branches[third]["label"]
        # stretch
        if u_ref[0] == "fn" and self.fn_list[u_ref[1]].kinds[u_ref[2]][0] == "diff":
            fn = self.fn_list[u_ref[1]]
            _, j, k = fn.kinds[u_ref[2]]
            first = self._end(fn.args[j], node, f"first end of {label}")
            second = self._end(fn.args[k], node, f"second end of {label}")
            letters[self.branches[first]["label"]] = "x"
            letters[self.branches[second]["label"]] = "y"
            stretch = "(x - y)"
            items.append(("fn", fn.fid, u_ref[2], {j: (first, 0), k: (second, 0)}))
        else:
            first = self._end(u_ref, node, f"stretch of {label}")
            letters[self.branches[first]["label"]] = "x"
            stretch = "x"
            items.append(("br", first, 0))
            self._node(node)["anchor"] = True
        words = {} if "κ" not in constants else {"κ": "the spring's stiffness" if K == "κ"
                                                     else "the spring's stiffness where g is 1"}
        self._node(node).update(stiffness=K, form=dict(letters=letters, constants=_numbers(constants), words=words,
                                                       terms=[dict(c="-1", h=f"{K}/2*{stretch}**2")]))
        self.marks.append(dict(node=node, path=path, term=None, items=items, read=list(letters.values())))

    def _end(self, ref, target, name):
        if ref[0] == "sig":
            return self.branch(ref[1], [ref[2]], target)
        self.hint([ref], name)
        sid = self.materialize(ref[1], [ref[2]], name)
        return self.branch(sid, [0], target)

    def _const_value(self, refs):
        values = [float(self.fn_list[r[1]].constants[r[2]]) for r in refs]
        return values[0] if len(set(values)) == 1 else values

    def _node(self, nid):
        for n in self.nodes:
            if n["id"] == nid:
                return n
        raise KeyError(nid)

    # ---- the scene --------------------------------------------------------
    def scene(self, reading: Optional[dict] = None, time: float = 0.) -> dict:
        """The explorer's version-1 scene. ``reading`` (from ``live_function``) fills the values
        and loads; without it they are ``None``."""
        nodes = [dict(n) for n in self.nodes]
        wires = []
        sources = {}
        for b, x in enumerate(self.branches):
            s = self.signals[x["sid"]]
            value = load = None
            if reading is not None:
                v = reading["values"][x["eps"]:x["eps"] + x["width"]]
                l = reading["loads"][x["eps"]:x["eps"] + x["width"]]
                value, load = (float(v[0]), float(l[0])) if x["width"] == 1 else (list(map(float, v)), list(map(float, l)))
            wires.append(dict(id=f"w{b}", source=s["node"], target=x["target"], label=x["label"],
                              value=value, load=load, valueScale=1, loadScale=1))
        if reading is not None:
            by_node = {n["id"]: n for n in nodes}
            stores = [s for s in self.signals if s["kind"] == "state"]
            for s, value in zip(stores, reading["states"]):
                v = np.asarray(value, dtype=float).ravel()
                by_node[s["node"]]["value"] = float(v[0]) if v.size == 1 else v.tolist()
        return dict(version=1, title=self.title, root="root", time=time, nodes=nodes, wires=wires)

    def static(self) -> dict:
        """What a page needs besides the live function: the scene without readings, and where
        each wire's numbers and each store's readout are in the live function's output."""
        return dict(scene=self.scene(), branches=[dict(eps=x["eps"], width=x["width"], indices=x["indices"],
                                                       source=self.signals[x["sid"]]["node"])
                                                  for x in self.branches],
                    states=[dict(node=s["node"], sid=sid, path=s["path"], start=self.qstart[s["path"]],
                                 size=self.tree[s["path"]].P.Q_dim)
                            for sid, s in enumerate(self.signals) if s["kind"] == "state"])


def _inputs_of(a: Arrangement):
    """The names of a part's input wires, from its meta, or from the part a filler fills."""
    named = (a.meta or {}).get("inputs")
    if named:
        return named
    if a.construction and a.construction[0] == "compose" and _default_compose_label(a) \
            and not _is_wiring(a.construction[2]):
        return _inputs_of(a.construction[2])
    return None


def _outputs_of(a: Arrangement):
    named = (a.meta or {}).get("outputs")
    if named:
        return named
    if a.construction and a.construction[0] == "compose" and _default_compose_label(a) \
            and not _is_wiring(a.construction[2]):
        return _outputs_of(a.construction[2])
    return None


def _part_name(a: Arrangement) -> str:
    """What a part is called: its own label, or its wiring's when its label is the default one."""
    if (a.meta or {}).get("wrapper"):
        return _part_name(a.construction[1])
    if a.construction and a.construction[0] == "compose" and _default_compose_label(a):
        inner = a.construction[2]
        return _part_name(inner) if inner.construction else inner.label
    return a.label


def _factors(a: Arrangement) -> List[Arrangement]:
    """The factors of a tensor tree, left to right."""
    if a.construction and a.construction[0] == "tensor" and not (a.meta or {}).get("part"):
        return _factors(a.construction[1]) + _factors(a.construction[2])
    return [a]


def _numbers(d):
    out = {}
    for k, v in d.items():
        try:
            out[k] = float(v) if np.ndim(v) == 0 else [float(x) for x in np.ravel(v)]
        except TypeError:
            pass
    return out


def diagram(arr: Arrangement, title: str = "Feeding") -> Diagram:
    return Diagram(arr, title)


# ---------------------------------------------------------------------------
# The live readings.
# ---------------------------------------------------------------------------


def readings(d: Diagram, arr: Arrangement, q: jax.Array) -> dict:
    """Every branch's value and load, and every store's readout, at the configuration ``q`` of
    ``arr``, which has the construction ``d`` was read from (built again, perhaps with traced
    places or parameters). Returns ``{'values', 'loads', 'states'}``."""
    tree = nodes_by_path(arr)

    def evaluate(eps):
        sig_memo, fn_memo = {}, {}

        def fn_full(fid):
            if fid not in fn_memo:
                fn = d.fn_list[fid]
                prim = tree[fn.path]
                args = [ref_value(r) for r in fn.args]
                fn_memo[fid] = _call(prim, fn.side, fn.sizes, args)
            return fn_memo[fid]

        def ref_value(r):
            if r[0] == "sig":
                return sig(r[1])[r[2]]
            return fn_full(r[1])[r[2]]

        def item_value(item):
            kind = item[0]
            if kind == "val":
                return ref_value(item[1])
            if kind == "br":
                x = d.branches[item[1]]
                return sig(x["sid"])[x["indices"][item[2]]] + eps[x["eps"] + item[2]]
            _, fid, i, perturbed = item
            fn = d.fn_list[fid]
            args = [ref_value(r) for r in fn.args]
            for j, (b, k) in perturbed.items():
                args[j] = item_value(("br", b, k))
            return _call(tree[fn.path], fn.side, fn.sizes, args)[i]

        def sig(sid):
            if sid not in sig_memo:
                s = d.signals[sid]
                if s["kind"] == "state":
                    prim = tree[s["path"]]
                    start = d.qstart[s["path"]]
                    sig_memo[sid] = prim.out_f(q[start:start + prim.P.Q_dim], EMPTY)
                else:
                    fn = d.fn_list[s["fid"]]
                    args = jnp.stack([item_value(it) for it in s["items"]]) if s["items"] else EMPTY
                    sig_memo[sid] = _call(tree[fn.path], fn.side, fn.sizes, list(args))[jnp.asarray(s["coords"])]
            return sig_memo[sid]

        vs = [jnp.stack([item_value(it) for it in m["items"]]) for m in d.marks]
        values = jnp.concatenate([jnp.ravel(sig(x["sid"])[jnp.asarray(x["indices"])]) for x in d.branches]) \
            if d.branches else EMPTY
        states = [sig(sid) for sid, s in enumerate(d.signals) if s["kind"] == "state"]
        return vs, (values, states)

    zero = jnp.zeros(d.eps)
    vs, (values, states) = evaluate(zero)
    cot = [_mark_form(tree, m, v) for m, v in zip(d.marks, vs)]
    _, pull = jax.vjp(lambda e: evaluate(e)[0], zero)
    (loads,) = pull(cot)
    return dict(values=values, loads=loads, states=states)


def live_function(build: Callable[..., Arrangement], example_args: tuple,
                  title: str = "Feeding") -> Tuple["Diagram", Callable[..., Dict[str, Any]], np.ndarray]:
    """The diagram of ``build(*example_args)`` and a function ``observe(flat_state, *args)``
    returning every branch's value and load and the composite's configuration, for
    ``scripts/export_step_js.javascript``: ``build`` is called again inside it, so places and
    parameters may be traced. Returns ``(diagram, observe, flat_initial_state)``."""
    from jax.flatten_util import ravel_pytree
    from .explore import _read
    concrete = build(*example_args)
    d = diagram(concrete, title)
    flat0, unravel = ravel_pytree(concrete.P.init)

    def observe(flat, *args):
        arr = build(*args)
        q = _read(arr.P, unravel(flat), None, 0)
        r = readings(d, arr, q)
        return r["values"], r["loads"], q

    return d, observe, np.asarray(flat0)


def _call(prim, side, sizes, args):
    q = jnp.zeros(0)
    flat = jnp.asarray(args) if len(args) else EMPTY
    m = flat[sizes[0]:sizes[0] + sizes[1]]
    if side == "out":
        return prim.out_f(q, m)
    n = flat[sizes[0] + sizes[1]:]
    return prim.in_f(q, m, n)


def _mark_form(tree, mark, v):
    """A mark's one-form at its inputs ``v``, by the library's own code."""
    part = tree[mark["path"]]
    meta = part.meta
    if mark["term"] is None:                       # a spring: its own field on (k, u)
        return part.omega(jnp.zeros(0), EMPTY, v)[2]
    term = meta["terms"][mark["term"]]
    h = lambda x: jnp.ravel(term["h"](x))
    c = jnp.broadcast_to(jnp.ravel(term["c"](v)), h(v).shape)
    form = jax.vjp(h, v)[1](c)[0]
    if mark.get("mask") is not None:               # one of pushes side by side
        form = jnp.zeros_like(form).at[jnp.asarray(mark["mask"])].set(form[jnp.asarray(mark["mask"])])
    return form
