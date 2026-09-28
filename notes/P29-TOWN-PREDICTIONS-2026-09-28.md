# P29: the town's mathematician — design fixed before construction code

Codex (GPT-6), session 0c750220, 2026-09-28. Status: proposed model and predictions.
David's story: book plan P29; intro notes section 1.6, “Her journey, David's version”.
David's answer about the plant: “i'm not sure. it's trying to maximize profits.”

## Objection and scope

Profit maximization alone does not imply waiting for a paid lawsuit. A plant with a correct
forecast prices future damages now. The experiment compares information available to otherwise
identical profit maximizers, and includes a plant informed from the start. It models learning a
harm coefficient and acting through an insurance quote, not understanding a book or inventing the
wiring. The court, enforceability, monetary valuation of harm, and willingness to use her estimate
are stipulated. No claim about actual law, insurance policies, or epidemiology is made.

The insurer and the insured use own-coordinate responses with one-way records. Non-exactness
rules out a potential whose differential equals this forcing; it does not rule out a Lyapunov
function, convergence, or a shared ethical objective. The smoke test detects asymmetric coupling
at a specified state; it does not uniquely identify a missing institution.

## Boxes, information, and money

All symbols below are local code names; no new book notation is bound. Each row is a 0-ary
Arrangement publishing its configuration, placed in a static outer wiring by tensor and
compose_seq (inside-to-outside substitution). Every smooth map uses its ordinary derivative;
one-way behavior is written explicitly as zero input components of the forcing. The integrator
of each existing box stays fixed as later stages add boxes.

| Box | Cost or forcing, observations, integrator |
| --- | --- |
| Plant | Output `output`; revenue `benefit*output`, production cost `output**2/2`, expected loss or premium `price*output**2/2`. Conditional cost is their negative profit: `(1+price)*output**2/2-benefit*output`. Own-coordinate descent, Euler duration `dt`: forcing `benefit-(1+price)*output`. Price is a held quoted coefficient, read one-way. This is a quadratic premium schedule, not a constant bill, and its marginal price changes with output. |
| Residents | Illness burden `health`, forcing `beta*output**2/2-recovery*health`. A conditional cost is `recovery*health**2/2-health*beta*output**2/2`, differentiated only in health. Euler `dt`. New illness is valued at one money unit per unit of illness for the liability model. Recovery does not erase accrued claims. |
| Observation memory | Publishes last step's output and illness burden. One-way register update, a CotCoalg realized by unit-step displacement `incoming-memory`. The observed incident rate is `(health-health_previous)/dt+recovery*health_previous`; its exposure is `output_previous**2/2`. At initialization previous state equals current state, so the first pair is uninformative at initial health zero. Neither learner gets `beta`. |
| Court delay | Two parallel finite chains of registers, carrying dated exposure and observed incident rate for `delay_steps` steps. Each register copies its predecessor; all are record wires. This is an exact discrete delay, not an exponential smoothing approximation. Zero initial buffers mean no prehistory. Court payouts are the delayed incident rate times `dt`. |
| Liability assessor | Estimate `assessed`; conditional prediction cost `(assessed*delayed_exposure-delayed_incidents)**2/2`. Own-coordinate gradient, rate `learn_rate`, Euler `dt`. Under law alone the plant uses this estimate to price expected future damages, while paying the actual settled claims from its treasury. This extra estimator, not supplied in P29's prose, makes the profit calculation causal and explicit. |
| Insurer | Quote coefficient `premium`; conditional cost `(premium-target)**2/2`, Euler rate `quote_rate`. Without her, target is the assessor's estimate from settled claims; with her, target is her estimate. This is a forecast-based quote, not the fund-feedback oscillator of test_insurance.py. Its claims lag here comes from a literal delay; the old test's quarter-turn is not evidence for it. |
| Mathematician | Estimate `estimate`; conditional cost `(estimate*observed_exposure-observed_incidents)**2/2`, own-coordinate gradient at the same `learn_rate` as the assessor. Publishes her estimate as data, with no input covector sent back to the town. Her forecast is checked against observations each step; the insurer uses her published value. |
| Payment records | Treasury, insurer fund (insured stages), and residents' compensation. Under law: treasury gains operating profit minus settled claims, compensation gains claims. Insured: treasury gains operating profit minus premium receipts; insurer gains premium receipts minus settled claims; compensation gains claims. Euler `dt`; no cost minimized on these records. Total money changes only by the explicitly external operating profit. No forecast term enters a balance directly. Unlimited credit is stipulated; negative balances are reported, not called solvent. |

Wire kinds: plant→residents one-way; plant and residents→observation memory record; dated
observations→delay→assessor one-way records and learning; assessor→plant one-way under law;
assessor→insurer→plant one-way under historical insurance; observations→mathematician→insurer
one-way with her. Within each box the scalar conditional cost yields its own gradient; dropping
input components makes the full open forcing generally non-exact. Premium and claims transfers
are records with equal opposite entries. No exact bilateral smoke coupling is silently added.

## Four experiments and predictions, fixed in advance

Default dimensionless constants: benefit=1, beta=2, recovery=1, dt=.02, initial output=.1,
initial health=0, all estimates and quote=0; learn_rate=40, quote_rate=2, delay_steps=300
(six time units), horizon=30. Initial balances zero. Each test file has a 180-second cap;
estimated runtime under 30 seconds each. Inspect the first completed tests and split if the
projection approaches the cap. No full-library run.

1. **No wire back.** The production update is exactly the scalar recurrence toward output 1,
independent of beta. The illness forcing has output derivative beta*output while the production
forcing has health derivative zero, so the cross-partial mismatch is nonzero away from output 0.
The composite must reproduce this and the analytic production trajectory. Able to fail: output
responds to residents through accidental retroactivity, or the measured mismatch is zero at
positive output. Run the same closed composite through Phi for a few steps as an independent
composition check.

2. **The law, late.** Registers reproduce a known dated sample exactly after their declared
number of steps, with no early access. Compared with a zero-delay assessor, the six-unit delay
raises cumulative incident harm over the fixed 30-unit horizon and raises peak output. This is
transient overproduction above the correctly priced output, not a prediction of oscillatory
instability. The stronger old guess that delay necessarily causes oscillation is not assumed.
Able to fail: either difference is nonpositive. The informed-from-start control uses beta as a
fixed initial estimate and should reduce harm despite equally delayed cash payouts. Payment
records must balance every step.

3. **Insurance without her.** Premium updates only once past settlements change the assessor's
estimate; it cannot anticipate the first claim. With a step in the supplied estimate it follows
the geometric smoothing response at the declared quote rate. In the town, time to premium=1
should be later than time to assessed=1. Insurer fund increments must equal actual premiums
minus claims, including when negative. Able to fail: quote leads its only information source,
or record composition creates money. No prediction that this insurer improves on law alone.

4. **Her account.** She sees dated health observations earlier than the court does; her estimate
should reach 1 before the first nonzero settled claim, and so should the premium reading it.
Compared at the same horizon and parameters with historical insurance, cumulative incident
harm should fall by at least 10%. Disconnecting her report must recover the historical-insurance
trajectory for all preexisting coordinates; a wrong report is not granted correctness. At zero
exposure her learning forcing is zero even for a wrong estimate, demonstrating a possible stale
fit. Test ordinary one-way report reading against an exact downstream prediction cost, whose
pullback should disturb her estimate. Able to fail: early inference does not occur, improvement
is below 10%, her report affects money without a payment, or merely adding the disconnected
learner changes the original town. Repeat the main comparison at half dt with the same physical
delay and duration; require the effect's sign and threshold to persist.

## Interpretation and placement, proposed

Reasonable at a town's scale: uncertain liability, delayed evidence, an estimate affecting a
priced decision, and money paid only through transfers. Artificial: noiseless health measurements,
known recovery, one unknown coefficient, one plant, no coverage exclusions or competition, a
quadratic premium schedule, unlimited credit, and an insurer that adopts her estimate without
negotiation. A lower harm total is not evidence that total social welfare increased: less output
also lowers operating benefit. No body/cell interpretation of courts or money is asserted; only
the abstract delayed observations and feedback mechanism transfers to such examples.

Candidate for Chapter 1, “Accounting for work at every scale”: her observation changes what
insurance prices, and that changes a profit maximizer's production. Chapter 5 should give the
explicit construction and the informed-plant control. Placement remains David's decision. The
model cannot show the town's people are valued by a nation, or that learning causes care.

## Implementation clarification and outcome (after tests; predictions above unchanged)

The observation memory additionally stores an availability coordinate, initially zero and
then one, so the first sample is (0,0). This corrects the implementation's original phantom
zero-harm observation. The informed-estimate invariant tests the correction. All four
predictions held; 11 checks pass. Full measurements, scope and book proposal are in
P29-TOWN-RESULTS.md; the recovery log is in the book's notes/P29-RUNNING-LOG.md.
