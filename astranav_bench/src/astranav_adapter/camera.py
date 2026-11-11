# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import contextlib
import mujoco
import numpy as np
from mujoco_g1_nav.scene import CEILING_GROUP, pitch_quat, vfov_from_hfov
HFOV_DEG, WIDTH, HEIGHT, PITCH_DEG = (110.0, 720, 640, 0.0)
GEOM_TYPE = int(mujoco.mjtObj.mjOBJ_GEOM)

class AstraCamera:

    def __init__(self, ctx):
        self.ctx = ctx
        self.model = ctx.scene.model
        self.cid = ctx.scene.ego_cam_id
        self.vfov = vfov_from_hfov(HFOV_DEG, WIDTH, HEIGHT)
        self.renderer = mujoco.Renderer(self.model, HEIGHT, WIDTH)
        self._seg = None
        self._view = mujoco.MjData(self.model)

    def _view_data(self, data):
        v, m = (self._view, self.model)
        v.qpos[:] = data.qpos
        if m.nmocap:
            v.mocap_pos[:] = data.mocap_pos
            v.mocap_quat[:] = data.mocap_quat
        mujoco.mj_kinematics(m, v)
        mujoco.mj_camlight(m, v)
        return v

    def pitch_deg(self, data) -> float:
        with self._astra_view():
            v = self._view_data(data)
            forward = -v.cam_xmat[self.cid].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arcsin(np.clip(forward[2], -1.0, 1.0))))

    @contextlib.contextmanager
    def _astra_view(self):
        m, cid = (self.model, self.cid)
        saved_quat, saved_fovy = (m.cam_quat[cid].copy(), float(m.cam_fovy[cid]))
        m.cam_quat[cid] = pitch_quat(PITCH_DEG)
        m.cam_fovy[cid] = self.vfov
        try:
            yield
        finally:
            m.cam_quat[cid] = saved_quat
            m.cam_fovy[cid] = saved_fovy

    def _light(self, data):
        if hasattr(self.ctx.cam, '_lit'):
            self.ctx.cam._lit(data)

    def rgb(self, data) -> np.ndarray:
        self._light(data)
        with self._astra_view():
            self.renderer.update_scene(self._view_data(data), camera=self.cid, scene_option=self.ctx.cam.opt)
            return self.renderer.render().copy()

    def seg_ids(self, data) -> np.ndarray:
        if self._seg is None:
            quality = self.model.vis.quality
            samples = int(quality.offsamples)
            quality.offsamples = 0
            try:
                self._seg = mujoco.Renderer(self.model, HEIGHT, WIDTH)
            finally:
                quality.offsamples = samples
            self._seg.enable_segmentation_rendering()
        self._light(data)
        with self._astra_view():
            self._seg.update_scene(self._view_data(data), camera=self.cid, scene_option=self.ctx.cam.opt)
            seg = self._seg.render().copy()
        return np.where(seg[..., 1] == GEOM_TYPE, seg[..., 0], -1)

    def close(self):
        self.renderer.close()
        if self._seg is not None:
            self._seg.close()
