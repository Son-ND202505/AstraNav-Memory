# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
import numpy as np
from scipy import ndimage
MIN_UNEXPLORED_M2 = 0.45
MIN_CELLS = 2

def frontier_waypoints(fog, reachable: np.ndarray, robot_xy, exclude=(), exclude_radius: float=0.8, min_unexplored_m2: float=MIN_UNEXPLORED_M2, max_n: int=16) -> list:
    fr = fog.frontier_cells(reachable, min_unexplored_m2)
    if not fr.any():
        return []
    lab, n = ndimage.label(fr, structure=np.ones((3, 3), bool))
    wm = fog.wm
    pts = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        ii, jj = np.nonzero(lab[sl] == k)
        if len(ii) < MIN_CELLS:
            continue
        ii, jj = (ii + sl[0].start, jj + sl[1].start)
        m = int(np.argmin((ii - ii.mean()) ** 2 + (jj - jj.mean()) ** 2))
        pts.append((float(wm.xs[jj[m]]), float(wm.ys[ii[m]])))
    pts = [p for p in pts if all((math.hypot(p[0] - e[0], p[1] - e[1]) > exclude_radius for e in exclude))]
    pts.sort(key=lambda p: math.hypot(p[0] - robot_xy[0], p[1] - robot_xy[1]))
    return pts[:max_n]
