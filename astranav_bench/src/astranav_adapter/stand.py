# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
import numpy as np
from mujoco_g1_nav.expert import LEFT, RIGHT, STOP
from mujoco_g1_nav.nav import wrap
from scipy import ndimage
from .confirm import ConfirmAstraNavController

class FaceBlobConfirmController(ConfirmAstraNavController):

    def __init__(self, *a, face_blob: bool=True, blob_radius_m: float=2.0, **kw):
        super().__init__(*a, **kw)
        self.face_blob, self.blob_radius_m = (face_blob, blob_radius_m)
        self.face_target = None

    def _blob_target(self, x, y):
        nav = self.nav
        px, py = self.found_xy
        r = int(self.blob_radius_m / nav.res)
        i0, j0 = nav.cell(px, py)
        i1, i2, j1, j2 = (max(0, i0 - r), min(nav.ny, i0 + r + 1), max(0, j0 - r), min(nav.nx, j0 + r + 1))
        occ = nav.furn_d[i1:i2, j1:j2] <= 0.5 * nav.res
        if not occ.any():
            return None
        lab, _ = ndimage.label(occ)
        ii, jj = np.nonzero(occ)
        d = np.hypot(nav.xs[jj + j1] - px, nav.ys[ii + i1] - py)
        k = lab[ii, jj][int(np.argmin(d))]
        si, sj = np.nonzero(lab == k)
        m = int(np.argmin(np.hypot(nav.xs[sj + j1] - x, nav.ys[si + i1] - y)))
        return (float(nav.xs[sj[m] + j1]), float(nav.ys[si[m] + i1]))

    def _face(self, x, y, yaw) -> int:
        tgt = self._blob_target(x, y) if self.face_blob else None
        self.face_target = tgt or self.found_xy
        err = wrap(math.atan2(self.face_target[1] - y, self.face_target[0] - x) - yaw)
        if abs(err) <= math.radians(15.0) or self.face_steps >= 8:
            self.phase = 'done'
            return STOP
        self.face_steps += 1
        return LEFT if err > 0 else RIGHT

    def summary(self) -> dict:
        s = super().summary()
        s.update(face_blob=self.face_blob, face_target=None if self.face_target is None else [round(float(v), 2) for v in self.face_target])
        return s
