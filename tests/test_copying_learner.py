"""Copying as a learner: a voice that sings along takes on a source's pitch and keeps it (book P27).

The arrangement is ``copying.singing_along``: a voice with a fast part (its last two values) and a slow part (its
pitch), and an ear whose cost is the mismatch between what the voice sings and what arrives. The source is the
environment's schedule through the external port, ``(s, K)``. Predictions were written first, in the book repo's
``notes/runs/p27-copying-dfb252e2/PREDICTIONS.md``; verdicts are in ``RESULTS.md`` beside it. Each test says whether
it could have come out otherwise.

Settings, shared by every test: source pitch 0.30 rad per step, the voice's own 0.22, ``eta_w = 0.2``.
"""

import jax
import numpy as np

from arrangements.copying import driven_step, loudness, pitch, run, schedule, singing_along, tone, voice_q

jax.config.update("jax_enable_x64", True)

W_A, W_B, ETA_W = 0.30, 0.22, 0.2
CHORD_2 = float(np.arccos(2 * np.cos(0.30) - np.cos(0.25)))  # equal tones at 0.25 and this: mean cos = cos 0.30


def _chord(t0=0):
    return lambda t: (np.sin(0.25 * (t0 + t)) + np.sin(CHORD_2 * (t0 + t))) / np.sqrt(2)


def _voice(eta_theta, eta_w=ETA_W, own=W_B):
    W = singing_along(eta_w, eta_theta, np.array([0.0, -np.sin(own), 2 * np.cos(own)]))
    return W, driven_step(W, check_inputs=schedule(tone(W_A), 3))


def _learn_then_silence(eta_theta, on=4000, off=3000):
    W, step = _voice(eta_theta)
    s, qs, ys, sc = run(W, step, W.P.init, schedule(tone(W_A), on))
    s_on = s
    s, qs2, ys2, _ = run(W, step, s, schedule(tone(W_A), off, K=0.0))
    return W, step, s_on, s, (sc[:, 0] - ys), ys2


def _crossing_pitch(y):
    z = np.where(np.diff(np.sign(y)) != 0)[0]
    return np.pi / np.mean(np.diff(z))


def test_copying_and_what_persists():
    """Predictions 1 and 2. Could fail: the pitch could stop short, or drift once the source stops."""
    _, _, s_on, s_off, e, ys2 = _learn_then_silence(0.01)
    q_on, q_off = voice_q(s_on), voice_q(s_off)
    print("pitch at the stop", pitch(q_on[2]), "mismatch last 200", np.abs(e[-200:]).max())
    print("after 3000 silent steps: pitch", pitch(q_off[2]), "by crossings", _crossing_pitch(ys2[100:]),
          "loudness", loudness(q_off))
    assert abs(pitch(q_on[2]) - W_A) < 1e-6 and np.abs(e[-200:]).max() < 1e-9
    assert abs(pitch(q_off[2]) - W_A) < 1e-6 and abs(_crossing_pitch(ys2[100:]) - W_A) < 1e-3
    assert abs(loudness(q_off) - 1.0) < 1e-6


def test_without_the_slow_part_the_note_is_held_only_while_it_sounds():
    """Prediction 3. Could fail: the fast part alone could follow fully, or keep the source's pitch after it stops."""
    _, _, s_on, s_off, e, ys2 = _learn_then_silence(0.0)
    print("mismatch last 200 while sounding", np.abs(e[-200:]).max())
    print("after the stop: pitch by crossings", _crossing_pitch(ys2[100:]), "loudness", loudness(voice_q(s_off)))
    assert np.abs(e[-200:]).max() > 0.05
    assert abs(_crossing_pitch(ys2[100:]) - W_B) < 1e-3


def _switch(eta_theta, source, steps=6000):
    W, step, _, s, _, _ = _learn_then_silence(eta_theta)
    th0 = voice_q(s)[2]
    s2, qs, ys, sc = run(W, step, s, schedule(source, steps))
    return th0, qs[:, 2], voice_q(s2)[2], sc[:, 0] - ys


def test_same_pitch_moves_the_fast_part_and_not_the_slow():
    """Prediction 4(a). Could fail: the transient push on the pitch could leave a displacement that does not shrink."""
    peaks = []
    for eta_theta in (0.01, 0.005, 0.0025):
        th0, th, th_end, e = _switch(eta_theta, tone(W_A, 0.5, 2.0, t0=7000))
        peaks.append(np.abs(th - th0).max())
        print("eta_theta", eta_theta, "first mismatch", abs(e[0]), "peak push on theta", peaks[-1],
              "net after 6000", th_end - th0)
    assert abs(_switch(0.01, tone(W_A, 0.5, 2.0, t0=7000))[2] - _switch(0.01, tone(W_A))[0]) < 1e-6
    assert peaks[1] < 0.6 * peaks[0] and peaks[2] < 0.6 * peaks[1]


def test_another_pitch_and_the_octave_move_the_slow_part_all_the_way():
    """Predictions 4(b), 4(c). Could fail only by a wrong build; the octave is a different pitch for this ear."""
    for w in (0.36, 0.60):
        _, _, th_end, e = _switch(0.01, tone(w, t0=7000))
        print("source", w, "pitch end", pitch(th_end), "mismatch last 500", np.abs(e[-500:]).max())
        assert abs(pitch(th_end) - w) < 1e-6


def test_a_chord_is_heard_near_its_mean_pitch():
    """Prediction 4(d). Could fail, and partly did: the pitch heard is biased by the follow, the bias shrinking as
    the fast part follows harder (eta_w up), and the mismatch is far smaller than the chord."""
    biases = []
    for eta_w in (0.1, 0.2, 0.3):
        W, step = _voice(0.005, eta_w=eta_w, own=W_A)
        s, qs, ys, sc = run(W, step, W.P.init, schedule(_chord(), 20000))
        heard = np.mean([pitch(x) for x in qs[-5000:, 2]])
        rms_e, rms_s = np.sqrt(np.mean((sc[-5000:, 0] - ys[-5000:]) ** 2)), np.sqrt(np.mean(sc[-5000:, 0] ** 2))
        biases.append(heard - W_A)
        print("eta_w", eta_w, "heard pitch", heard, "rms mismatch", rms_e, "rms chord", rms_s)
        assert 0 < rms_e < 0.2 * rms_s
    assert 0 < biases[2] < biases[1] < biases[0] < 0.01
