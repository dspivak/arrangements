# Carving a rest that was not there: predictions fixed before any code

Claude Opus 5 (1M), session 1a70010f, 2026-09-28. Written before `tests/test_carving.py` exists
(`AGENTS.md` in the book repo, the three red flags; the book's plan, open-mathematics item 8a).

**The question, David's** (2026-09-28): "i think a new form can start to carve out a new capacity.
is that not something we've imagined yet? this is how memory works, and how river beds work." The
sustained-note run (`tests/test_sustained_note.py`) held the reader's wells fixed: a passing note
could flip a committed reader between two rests it already had, and nothing carved a rest. The
claim under test is that a form played long enough carves one.

## The arrangement

One reader, state $(q,w)\in\rr^2$, response the diagonal sharp $\operatorname{diag}(\eta_q,\eta_w)$
with $\eta_w\ll\eta_q$, so $q$ is fast and $w$ slow. Cost, with the note $u$ on the exterior input:

$$k(q,w;u)=\tfrac14 q^4-\tfrac12 w\,q^2-u\,q+\tfrac{\gamma}{2}(w-w_0)^2 .$$

$w$ is the fast landscape's curvature at the origin: for $w<0$ the reader has one rest, $q=0$; for
$w>0$ it has two, $q=\pm\sqrt w$. The slow coordinate descends the *same* cost, so
$w'=w+\eta_w(\tfrac12 q^2-\gamma(w-w_0))$: activity deepens, disuse decays. Start at $w_0=-\tfrac12$,
so the reader begins with one rest and no memory. Take $\gamma<\tfrac12$, which is what makes a
carved state hold itself up: the joint rests at $u=0$ solve $q^2=w$ and $w=w_0+q^2/(2\gamma)$, so
$w=w_0/(1-1/(2\gamma))$, positive exactly when $\gamma<\tfrac12$. With $\gamma=\tfrac14$: $w=\tfrac12$,
$q=\pm\sqrt{1/2}$.

## Predictions

1. **A hold threshold.** There is a number of steps $T_*$ such that a note held for $T\ge T_*$
   leaves the reader, after the note stops, at the carved rest $(\pm\sqrt{1/2},\tfrac12)$ forever,
   and a note held for $T<T_*$ returns it to $(0,w_0)$. The threshold is sharp to within a couple of
   steps. **Could fail**: the carved state may be unreachable, or the crossing may be gradual with
   no threshold. This is the paragraph's "played long enough."
2. **A rest that was not there.** After carving, the fast landscape at the settled $w$ has
   $\partial_q^2 k|_{q=0}=-w<0$ and minima at $\pm\sqrt w$, where before the note it had one minimum
   at $0$. A kick that does not cross $q=0$ decays back into the carved well. **Could fail** if $w$
   settles at or below $0$.
3. **Nothing new in the joint state space** (closed form; cannot come out otherwise, so not a
   result). The carved rest is a fixed point of the joint dynamics at $u=0$ and was one before the
   note ever played. So carving is basin selection at the joint level, and what is new is the *fast*
   landscape at the slow coordinate's new value. The book must say it that way.
4. **The accounting survives a moving landscape.** Along every run, with the note on and with it
   off, the joint cost decreases step to step, $k(q',w';u)\le k(q,w;u)$, so dissipation $-\Delta k$
   stays non-negative and "dissipation is the rate the supply is spent" is unharmed by the tuning
   moving. **Could fail** on step size: a step too large overshoots and $k$ rises.
5. **Non-reciprocal tuning breaks it.** Replace the slow rule with one driven by $q$ rather than
   $\tfrac12 q^2$, $w'=w+\eta_w(\lambda q-\gamma(w-w_0))$, keeping $q$'s rule as before. No single
   cost has both as its gradient, so the pair's forcing is non-exact, and I predict a run in which
   the joint cost of the exact model rises. The lesson for the book, if it holds: a tuning that
   carves may be accounted for only when it descends the same cost the fast coordinate does.
   **Could fail**: the mismatch may still descend something, so nothing rises.

## Weirdness to check at each scale when reporting

A society, a body, a cell: whether a self-holding memory that needs no rehearsal (prediction 1's
carved rest) is reasonable at each; whether the decay rate $\gamma$ has a reading; and whether the
threshold's sharpness is an artifact of one dimension.

## Rainfall: two more predictions, fixed before the rainfall code

David, 2026-09-28: "should have been something like a weekly rainfall or something so that it
shouldn't be so drastic. Like if you get to a certain amount, then something happens. And if you
don't get to a certain amount, something else happens. If you're getting rainfall at around once a
week or whatever, some regularity, then it shouldn't be like this." So the note is no longer held
once and switched off: it arrives on a cycle, on for a fraction $f$ of each period $T_{\text{rain}}$.

6. **The deciding quantity is the rate, not the duration.** For a period short against the tuning's
   own time ($10^4$ steps), the outcome depends on $f$ through a threshold $f_*$, and $f_*$ is the
   same at $T_{\text{rain}}=200$ and at $T_{\text{rain}}=100$, within one grid step. The knife-edge
   does not go away — that is what two self-holding states mean — but it moves from a step count to a
   rate, which is the honest way to say "played long enough."
7. **Whether burstiness matters at equal average, and which way.** Two effects pull opposite ways:
   the deepening is $q^2/2$, convex, so the same water delivered in bursts deepens *more*; but
   between bursts the tuning decays and the fast coordinate can fall back across the divide, which
   deepens *less*. Prediction: bursty wins at periods short against the tuning's time, drizzle wins
   at periods long against it, so at fixed $f$ near the threshold there is a crossover period.
   **I am not sure of this one**, which is why it is worth the run (David: "If you're not sure if it
   would work, then let's try it").
