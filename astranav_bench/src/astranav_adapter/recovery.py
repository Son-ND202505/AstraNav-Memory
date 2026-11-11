# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
from mujoco_g1_nav.expert import STOP
from .controller import MAX_DECISIONS
from .frontiers import frontier_waypoints
IDLE_STOP = 3

class RecoveryMixin:

    def __init__(self, *a, recover_sweeps: int=0, **kw):
        super().__init__(*a, **kw)
        self.recover_sweeps, self.sweeps = (int(recover_sweeps), 0)

    def _frontier_counts(self, x, y) -> dict:
        raw = frontier_waypoints(self.fog, self.nav.free, (x, y), exclude=(), max_n=10 ** 6)
        radius = frontier_waypoints(self.fog, self.nav.free, (x, y), exclude=self.visited, max_n=10 ** 6)
        seen = {(round(float(v[0]), 1), round(float(v[1]), 1)) for v in self.visited}
        exact = [p for p in raw if (round(float(p[0]), 1), round(float(p[1]), 1)) not in seen]
        try:
            frac = round(float(self.fog.fraction()), 3)
        except Exception:
            frac = None
        return dict(n_raw=len(raw), n_radius=len(radius), n_exact=len(exact), n_visited=len(self.visited), fog_frac=frac)

    def _decide(self, x, y, yaw) -> int:
        try:
            diag = self._frontier_counts(x, y)
        except Exception as e:
            diag = dict(error=f'{type(e).__name__}: {e}')
        n0 = len(self.records)
        act = super()._decide(x, y, yaw)
        if len(self.records) > n0:
            self.records[-1]['diag'] = diag
        dead_end = act == STOP and self.phase == 'done' and (self.idle >= IDLE_STOP) and (len(self.records) < MAX_DECISIONS)
        if dead_end and self.sweeps < self.recover_sweeps:
            return self._restart_sweep(x, y, yaw)
        return act

    def _restart_sweep(self, x, y, yaw) -> int:
        self.sweeps += 1
        self.fog.reset()
        self.visited.clear()
        self.idle = 0
        self.phase = 'spin'
        if self.records:
            self.records[-1]['recovery_sweep'] = self.sweeps
        return self._spin(x, y, yaw)

    def summary(self) -> dict:
        s = super().summary()
        s.update(recover_sweeps=self.recover_sweeps, sweeps=self.sweeps)
        return s

def with_recovery(cls):
    return type('Recovery' + cls.__name__, (RecoveryMixin, cls), {})
