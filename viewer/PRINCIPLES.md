# Principles for a page that shows a run

Drawn from David's reactions to the two pages built so far, the listener (`live.html`; Claude Opus 5.5, 9effcd80,
2026-10-09) and the two tanks (`tanks.html`; Claude Opus 5.5, 322cc59f, 2026-10-09), read through his message of the listener run's commit. Each line names the page and his
words behind it. These are the agents' wording, *proposed*. David on the two-tanks page built on them: "This is really
great." A page that departs from one says why in its own header comment.

## What the page is for

1. **One question per page,** said in a line: what the viewer can try. The listener's was "what happens to listeners
   when the person they're talking to is reliable or unreliable or unaccountable." Use one wiring until that works
   ("before we get the story and the visualization and the UI right").
2. **The page shows why things move.** Beside the thing that moves, show the calculation that moves it, in the order
   the mathematics has it. On the tanks that order is: the levels are read out, the pipe returns a flow, the volumes
   change. If a motion looks absurd, that is a question about the model to check against the mathematics, not a
   display bug. The listener walking away after a thousand kept promises turned out to be a property of the model.
3. **Time on the page is the model's time.** How long a step lasts is an input the viewer sets, and Play takes one
   step per that much real time ("if we put it on a second, it should update every second … a millisecond … every
   millisecond"). At one unit of time it is Chapter 2's step; Chapter 3 hands each step its duration.

## Who controls what

4. **The viewer drives.** What happens at each step is the viewer's to set. Nothing moves until they press Play,
   and Play runs until they pause it or the run comes to rest ("Why can't I just press play?").
5. **A number shown is a number you can slide, right where it is shown.** Don't put a separate slider somewhere
   else ("It feels weird for you to display a number and then not let me slide that number"), and don't add a
   slider for something the mouse already sets.
6. **Only inputs are controllable.** The viewer controls the independent variables, never a number computed
   from them; the page shows the calculation instead ("just give me the variable"). Inputs and outputs look
   different: on the tanks, inputs are blue with a dashed underline, outputs are grey.
7. **Grab the thing.** Set a part's state by dragging the part itself, whether that is the listener or the water
   in a tank.
8. **A setting at which nothing can ever change is left out, or the page says so.** At 100% the listener never moves
   ("if the user ever selects 100%, then they should be told like, this is never going to change … I don't think we
   need 100%"). A number shown and the range it may take agree everywhere on the page.
9. **An input is drawn as well as numbered.** On the tanks, the pipe's thickness follows its conductance, scaled
   honestly: for slow flow through a narrow pipe the conductance goes as the radius to the fourth power, so the
   drawn radius goes as its fourth root.

## Layout

10. **Controls go where the eye already is.** Play sits beside the picture ("where is Play?? … way down at the
   bottom"). The time track sits under the record of steps, on the same scale ("have it match them"). Start over
   is on the picture and keeps what the viewer set.
11. **The page never grows.** Its width stays fixed and long histories shrink. A body never leaves the screen; the
    axis compresses, with an end at infinity ("I shouldn't have to lose my guy").
12. **Text that changes while playing holds still: every changing number has its width decided in advance.** Use a
    fixed number of decimals, always show the sign of a number that can change sign, choose a unit by a setting that is
    fixed during the run (never by the number itself), and pad the number to the widest it can be with figure spaces
    (U+2007, as wide as a digit in a tabular font), so a sentence never shifts sideways ("when it moves to five, it
    doesn't say 5.00, and so the entire text moves left"; and on the pendulum, 2026-10-14: "10 versus 10.1 has a
    different length … otherwise it's like constantly vibrating … there just needs to be space decided in advance for
    how long something's gonna be … a common thing we need to put throughout"). Numbers that change every frame are
    rewritten about ten times a second, not every frame, so they can be read. `pendulum.html`'s `fw` is the helper.
13. **Built for a glance, not for reading top to bottom.** Whatever the eye lands on says what it is. A thing is
    named by its role every time, "the speaker," never "them" ("they're going to glance around and they're going
    to need to know what they're seeing right when they see it"). Keep few words on the page and put the rest
    behind "explain" buttons. Use plain words: "ten things happen in a row" needs no special term. A quantity is named
    by whose it is and what it is about: "the listener's belief about how often the speaker breaks a promise."

## Mechanics

14. **A moving mark that shows a rate adds up that rate frame by frame, with its sign.** The tanks' first version
    moved the pipe's dashes by "time since the page loaded × the current flow." That runs backward as the flow
    falls, and David saw water going from the lower tank to the higher one.
15. **The page's model is checked against the library.** Run the same arrangement in the library and compare
    numbers, then say on the page what was checked. The listener's test compares the page with
    `listener.run_script`. The tanks were checked against `tests/test_two_tanks.py`'s tanks with the book's linear
    spout, for one-second steps.
16. **The numbers behind the picture can be seen.** When a motion surprises, the viewer can open the run's numbers
    step by step ("I don't know if you can grab the data from the website. That'd be a cool feature, like a debug
    button"; asked on the listener, not yet built on either page).
17. **When the page changes, give one instruction** for seeing it, such as "reload" ("don't tell me five things to do").

## Checking a page in headless Chrome

Headless virtual time does not advance `requestAnimationFrame`, so a screenshot taken after pressing Play shows step 0.
To see a page mid-run, set the state directly in a debug copy (`params.script`, `t`, then `recompute()`) and capture
that (Claude Opus 5.5, 9effcd80, 2026-10-09, from the suggestion box; moved here by the sweep, e4e712b7, 2026-10-10).
