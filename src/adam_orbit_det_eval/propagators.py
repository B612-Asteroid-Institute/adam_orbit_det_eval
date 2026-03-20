"""
Propagator classes for use with the LOOO pipeline.

These are importable by fully-qualified name so they can be serialised
across multiprocessing worker boundaries (the pipeline passes propagator
class FQNs as strings and re-imports them in each worker).
"""

from adam_core.dynamics.propagation import propagate_2body
from adam_core.propagator.propagator import Propagator


class TwoBodyPropagator(Propagator):
    """
    Fast 2-body (Keplerian) propagator wrapping adam_core.

    Suitable for development and testing.  For production use the
    ASSIST N-body propagator (adam_assist.ASSISTPropagator).
    """

    def _propagate_orbits(self, orbits, times, max_iter=1000, tol=1e-14, **kwargs):
        return propagate_2body(orbits, times, max_iter=max_iter, tol=tol)
