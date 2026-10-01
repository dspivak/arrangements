You are an ordinary adult with no training in physics, biology or mathematics. You will be told about
seven small "things" and four ways of joining two of them. For each of twelve joined pairs, say what you
expect to happen over a while, from everyday intuition. Do not calculate; guess the way a curious person
would. Every thing reports one number, and the joints act on those numbers.

THE THINGS (and what each does on its own):
- weight: a weight bouncing on a spring. Alone it swings up and down between -1 and 1 forever. Its number is its height.
- tank: a leaky tank under a running tap. Alone it fills from empty to a level of 1 and stays there. Its number is its water level.
- rabbits: a rabbit population. Alone it grows from 0.1 up to what the land can hold, 1, and stays there. Its number is how many rabbits.
- foxes: a fox population with nothing to eat. Alone it dies out, from 1 toward 0. Its number is how many foxes.
- metronome: a metronome. Alone it ticks steadily; its number is its running beat count, rising by 1 per unit of time forever.
- chooser: a person choosing between two options, leaning 60/40 toward option 1 at the start. Alone they make up their mind for option 1. Its number is how much they favor option 1, from 0 (option 2) to 1 (option 1).
- needle: a needle on a dial. It moves only when something pushes it. Alone it stays at 0. Its number is where it points.

THE JOINTS (always between a FIRST and a SECOND thing):
- band: a rubber band between their two numbers; it pulls the two numbers toward each other.
- eats: the second eats the first. When they meet (more often when both are plentiful), the first loses and the second gains.
- in_step: each nudges the other to keep in step, like two metronomes on one shelf.
- watches: the first watches the second and drifts toward the second's number; the second is not affected at all.

THE TWELVE PAIRS (first, second, joint):
1 metronome, tank, eats
2 rabbits, needle, eats
3 rabbits, needle, band
4 rabbits, tank, eats
5 weight, rabbits, band
6 tank, foxes, eats
7 rabbits, chooser, band
8 tank, weight, watches
9 foxes, chooser, in_step
10 metronome, rabbits, in_step
11 metronome, tank, watches
12 chooser, rabbits, eats

For each pair answer, for EACH of the two things: (a) compared with how it behaves alone, does its number
end up higher, lower, about the same, or doing something else (say what); (b) does it end up steady,
swinging, growing without limit, or dying out; and (c) one plain sentence on what happens to the pair.
Reply with only a JSON list of twelve objects:
{"pair": 1, "first": {"vs_alone": "...", "kind": "..."}, "second": {"vs_alone": "...", "kind": "..."}, "story": "..."}
