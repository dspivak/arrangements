"""Externally positioned bodies in the smooth-interface arrangement category.

The external input carries one (x,y,angle) configuration per held cart. Its
pushback is force and torque. Optional internal cart boxes carry the other
positions, so holding one body and letting another move is also a composition.
No force law is duplicated: both use magnets._received.

GPT-6, d5e53279, 2026-11-11; book plan P76 (j).
"""
from .magnets import FIELDS, _carts, _frame, _received
from .smooth import R, SmoothArrangement, compose_seq, tensor, zero_form
from .wiring import trivial_parameter


def open_field(n, cart_of=None, held=None):
    """Static arrangement on bodies and free carts, with held carts as inputs.

    ``held=None`` holds all carts. Otherwise ``held`` lists cart indices in the
    order of the external R(3) input factors. Free cart output factors follow
    the bodies' fields in cart-index order. An empty ``held`` gives magnets.field.
    """
    cart_of, c = _carts(cart_of, n)
    held = list(range(c)) if held is None else list(held)
    if len(set(held)) != len(held) or any(i < 0 or i >= c for i in held):
        raise ValueError("held must list distinct cart indices")
    free = [i for i in range(c) if i not in held]

    def in_f(q, m, positions):
        carts = {i: positions[k] for k, i in enumerate(held)}
        carts.update({i: m[2*n+k] for k,i in enumerate(free)})
        return _received([_frame(carts[cart_of[i]]) for i in range(n)],m[:2*n])

    return SmoothArrangement(
        P=trivial_parameter(), M_in=FIELDS*n, M_out=FIELDS*n+(R(3),)*len(free),
        N_in=(R(3),)*len(held), N_out=(), out_f=lambda q,m: (),
        in_f=in_f, omega=zero_form, label=f"openField({c},held={held})")


def open_system(bodies, cart_of=None, held=None, carts=()):
    """Fill open_field with bodies and any free-cart parts in cart-index order."""
    return compose_seq(tensor(list(bodies)+list(carts)),open_field(len(bodies),cart_of,held))
