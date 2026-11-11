# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
from collections import namedtuple
import numpy as np
from scipy.spatial.transform import Rotation as R
Quat = namedtuple('Quat', 'w x y z')
State = namedtuple('State', 'position rotation')
_F = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
_M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

def mj_position(x: float, y: float, z: float=0.0) -> np.ndarray:
    return np.array([-y, z, -x], dtype=float)

def mj_quat(yaw: float) -> Quat:
    return Quat(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)

def mj_state(x: float, y: float, yaw: float) -> State:
    return State(mj_position(x, y), mj_quat(yaw))

def habitat_to_mujoco_xy(p) -> tuple:
    return (float(-p[2]), float(-p[0]))

def _rot(q: Quat) -> R:
    return R.from_quat([q.x, q.y, q.z, q.w])

def _matrix(state: State) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rot(state.rotation).as_matrix() @ _F @ _M
    T[:3, 3] = np.asarray(state.position, dtype=float)
    return T

def relative_pose(state: State, ref: State) -> tuple:
    rel = np.linalg.inv(_matrix(ref)) @ _matrix(state)
    return (float(rel[0, 3]), float(rel[1, 3]), float(math.atan2(-rel[0, 1], rel[1, 1])))

def to_local(point, ref: State) -> list:
    rel = np.asarray(point, dtype=float) - np.asarray(ref.position, dtype=float)
    local = np.round(_rot(ref.rotation).apply(rel, inverse=True), 2).tolist()
    return [local[0], local[1] * -1, local[2] * -1]

def from_local(local, ref: State) -> np.ndarray:
    lp = np.array([local[0], local[1] * -1, local[2] * -1], dtype=float)
    return _rot(ref.rotation).apply(lp) + np.asarray(ref.position, dtype=float)
