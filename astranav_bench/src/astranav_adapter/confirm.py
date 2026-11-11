# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
import numpy as np
from mujoco_g1_nav.expert import STOP
from mujoco_g1_nav.house_check import CAMERA_H
from .controller import MAX_DECISIONS, STUCK_MOVE_M, AstraNavController
from .frontiers import frontier_waypoints

class ConfirmAstraNavController(AstraNavController):

    def __init__(self, *a, far_m: float=3.5, stand_far_m: float=3.0, stand_near_m: float=1.6, near_ok_m: float=2.2, max_confirms: int=3, max_rejects: int=2, near_look: bool=True, ray_stand: bool=False, look_los: bool=False, chain_confirms: bool=False, max_total_confirms: int=6, los_min_left: float=1.0, los_target_h: float=0.5, los_clear_m: float=1.0, snap_path: bool=False, snap_detour_m: float=3.0, snap_w: float=3.0, snap_radius_m: float=2.0, consistency_m: float=0.0, max_consistency_looks: int=2, guard_confirmation_cap: bool=False, **kw):
        super().__init__(*a, **kw)
        self.far_m, self.stand_far_m, self.stand_near_m, self.near_ok_m = (far_m, stand_far_m, stand_near_m, near_ok_m)
        self.max_confirms, self.max_rejects, self.near_look, self.ray_stand = (max_confirms, max_rejects, near_look, ray_stand)
        self.look_los, self.chain_confirms, self.max_total_confirms = (look_los, chain_confirms, max_total_confirms)
        self.los_min_left, self.los_target_h, self.los_clear_m = (los_min_left, los_target_h, los_clear_m)
        self.snap_path, self.snap_detour_m, self.snap_w, self.snap_radius_m = (snap_path, snap_detour_m, snap_w, snap_radius_m)
        self.snaps_moved = 0
        self.pending = None
        self.confirms = self.rejects = 0
        self.chain = 0
        self.blind_looks = 0
        self.rejected = []
        self.stand_m = None
        self.consistency_m = consistency_m
        self.max_consistency_looks = max_consistency_looks
        self.consistency_looks = 0
        self.guard_confirmation_cap = guard_confirmation_cap
        self.cap_rejections = 0

    def _cap_reached(self):
        used = self.chain if self.chain_confirms else self.confirms
        return used >= self.max_confirms or self.rejects >= self.max_rejects or (self.chain_confirms and self.confirms >= self.max_total_confirms)

    def _reject_far_at_cap(self, point, x, y):
        return self.guard_confirmation_cap and self._cap_reached() and (math.hypot(point[0] - x, point[1] - y) > self.near_ok_m)

    def _stand_for(self, announced_m: float, left_m: float=math.inf):
        used = self.chain if self.chain_confirms else self.confirms
        if used >= self.max_confirms or self.rejects >= self.max_rejects or (self.chain_confirms and self.confirms >= self.max_total_confirms):
            return None
        if announced_m > self.far_m:
            stand = self.stand_far_m
        elif self.near_look and announced_m > self.near_ok_m:
            stand = self.stand_near_m
        else:
            return None
        if left_m <= stand + 0.3:
            if stand == self.stand_far_m and self.near_look and (announced_m > self.near_ok_m) and (left_m > self.stand_near_m + 0.3):
                return self.stand_near_m
            return None
        return stand

    def _snap_found(self, x, y, obj_xy):
        snapped = super()._snap_found(x, y, obj_xy)
        if not self.snap_path or snapped is None:
            return snapped
        nav = self.nav
        start = nav.nearest_free(x, y, radius=1.0)
        if start is None:
            return snapped
        d0 = nav.field([(float(nav.xs[start[1]]), float(nav.ys[start[0]]))]).dist
        path_nearest = float(d0[nav.cell(*snapped)])
        if not math.isfinite(path_nearest) or path_nearest <= math.hypot(snapped[0] - x, snapped[1] - y) + self.snap_detour_m:
            return snapped
        ii, jj = np.nonzero(np.isfinite(d0) & nav.free)
        off = np.hypot(nav.xs[jj] - obj_xy[0], nav.ys[ii] - obj_xy[1])
        ok = off <= self.snap_radius_m
        if not ok.any():
            return snapped
        ii, jj, off = (ii[ok], jj[ok], off[ok])
        cost = d0[ii, jj] + self.snap_w * off
        k = int(np.argmin(cost))
        cand = (float(nav.xs[jj[k]]), float(nav.ys[ii[k]]))
        if cost[k] >= path_nearest + self.snap_w * math.hypot(snapped[0] - obj_xy[0], snapped[1] - obj_xy[1]):
            return snapped
        self.snaps_moved += 1
        return cand

    def _sees(self, x, y, point) -> bool:
        dx, dy = (point[0] - x, point[1] - y)
        dist = math.hypot(dx, dy)
        wm = self.fog.wm
        d = np.arange(wm.res, dist - self.los_clear_m, wm.res)
        if len(d) == 0:
            return True
        i = np.clip(((y + dy / dist * d - wm.y0) // wm.res).astype(np.int64), 0, wm.ny - 1)
        j = np.clip(((x + dx / dist * d - wm.x0) // wm.res).astype(np.int64), 0, wm.nx - 1)
        line = CAMERA_H - (CAMERA_H - self.los_target_h) * d / dist
        return not bool(np.any(self.fog.h[i, j] > line))

    def _ray_stand(self, x, y, point, snapped):
        dx, dy = (point[0] - x, point[1] - y)
        dist = math.hypot(dx, dy)
        if dist < 0.5:
            return None
        ux, uy = (dx / dist, dy / dist)
        last, hit, s = (None, None, 0.0)
        while s <= dist + 0.5:
            qx, qy = (x + s * ux, y + s * uy)
            if not self.nav.is_free(qx, qy):
                hit = s
                break
            last = (qx, qy)
            s += 0.05
        if hit is None or last is None or (not dist - 2.0 <= hit <= dist + 0.5):
            return None
        if math.hypot(last[0] - snapped[0], last[1] - snapped[1]) > 2.5:
            return None
        left_snapped = float(self.nav.value(self.nav.field([snapped]), x, y))
        left_ray = float(self.nav.value(self.nav.field([last]), x, y))
        if not math.isfinite(left_ray) or (math.isfinite(left_snapped) and left_ray > left_snapped + 3.0):
            return None
        return (float(last[0]), float(last[1]))

    def _decide(self, x, y, yaw) -> int:
        ref = (x, y, yaw)
        fr = frontier_waypoints(self.fog, self.nav.free, (x, y), exclude=self.visited, max_n=self.max_frontiers)
        views, self.views = (self.views, [])
        d = self.decider.decide(views, ref, fr, self.phrase)
        self.model_seconds += float(d.get('seconds') or 0.0)
        seen_px = max((v.cat_px for v in views))
        if self.debug_dir is not None:
            self._dump(views, d, fr, len(self.records))
        field = self.world.field(self.cat)
        costs = [float(self.nav.value(field, fx, fy)) if field is not None else math.nan for fx, fy in fr]
        rec = dict(t=self.t, pose=[round(x, 2), round(y, 2), round(yaw, 3)], n_frontiers=len(fr), kind=d['kind'], text=d['text'], seen_px=seen_px, seconds=round(float(d.get('seconds') or 0.0), 2), fr_cost=[round(c, 2) if math.isfinite(c) else None for c in costs], fr_index=d.get('frontier_index'), stage='confirm' if self.pending is not None else 'explore', confirms=self.confirms)
        rec.update({k: d[k] for k in ('n_images', 'n_history', 'n_tokens', 'vision_encoded_images', 'vision_total_images') if d.get(k) is not None})
        if self.pending is not None and self.look_los:
            rec['los'] = bool(self._sees(x, y, self.pending['point']))
        for k in ('p_found', 'first_token_logprobs'):
            if d.get(k) is not None:
                rec[k] = d[k]
        if len(self.records) >= MAX_DECISIONS:
            self.records.append(rec)
            self.phase = 'done'
            return STOP
        if d['kind'] == 'found' and self._reject_far_at_cap(d['target'], x, y):
            rec.update(cap_rejected=True, rejection_reason='confirmation_budget_exhausted_far', predicted_xy=list(d['target']), announced_m=round(math.hypot(d['target'][0] - x, d['target'][1] - y), 3))
            self.cap_rejections += 1
            self.found_xy = None
            d = dict(d, kind='none')
        if d['kind'] == 'found':
            snapped = self._snap_found(x, y, d['target'])
            if snapped is not None:
                point = (float(d['target'][0]), float(d['target'][1]))
                self.found_xy = point
                announced = math.hypot(point[0] - x, point[1] - y)
                rec.update(found_xy=[round(v, 2) for v in point], found_err_m=round(min((math.hypot(point[0] - it.x, point[1] - it.y) for it in self.items)), 2), snapped=[round(snapped[0], 2), round(snapped[1], 2)], announced_m=round(announced, 2))
                stand = self._stand_for(announced, float(self.nav.value(self.nav.field([snapped]), x, y)))
                if self.consistency_m > 0 and self.pending is not None:
                    shift = math.hypot(point[0] - self.pending['point'][0], point[1] - self.pending['point'][1])
                    rec['confirmation_shift_m'] = round(shift, 3)
                    if stand is None and shift > self.consistency_m and (self.consistency_looks < self.max_consistency_looks):
                        self.consistency_looks += 1
                        rec['consistency_reobserve'] = True
                        stand = min(self.stand_near_m, announced)
                self.records.append(rec)
                if stand is None:
                    self.pending = None
                    rec['accepted'] = True
                    rec['accept_reason'] = 'near_prediction' if announced <= self.near_ok_m else 'confirmation_cap' if self._cap_reached() else 'path_stage_reached'
                    if self.ray_stand:
                        cand = self._ray_stand(x, y, point, snapped)
                        if cand is not None:
                            rec['ray_stand'] = [round(cand[0], 2), round(cand[1], 2)]
                            snapped = cand
                    return self._start_follow(x, y, yaw, snapped, 'found')
                self.pending, self.stand_m = (dict(point=point), stand)
                self.confirms += 1
                self.chain += 1
                rec['confirm_at_path_m'] = stand
                return self._start_follow(x, y, yaw, snapped, 'confirm')
            rec['kind'] = 'found_unreachable'
            d = dict(d, kind='none')
        if self.pending is not None:
            self.rejects += 1
            self.chain = 0
            self.rejected.append(self.pending['point'])
            rec['rejected'] = [round(v, 2) for v in self.pending['point']]
            self.pending = self.found_xy = None
        if d['kind'] == 'frontier' and d['frontier_index'] is not None:
            tgt = fr[d['frontier_index']]
            self.visited.append(tgt)
            rec['frontier'] = [round(tgt[0], 2), round(tgt[1], 2)]
            self.records.append(rec)
            return self._start_follow(x, y, yaw, tgt, 'frontier')
        self.fallbacks += 1
        rec['fallback'] = True
        self.records.append(rec)
        if fr:
            tgt = fr[0]
            self.visited.append(tgt)
            return self._start_follow(x, y, yaw, tgt, 'frontier')
        self.idle += 1
        if self.idle >= 3:
            self.phase = 'done'
            return STOP
        return self._spin(x, y, yaw)

    def _follow(self, x, y, yaw) -> int:
        if self.mode != 'confirm':
            return super()._follow(x, y, yaw)
        self.history.append((x, y))
        stuck = len(self.history) == self.history.maxlen and math.hypot(self.history[0][0] - x, self.history[0][1] - y) < STUCK_MOVE_M
        left = float(self.nav.value(self.field, x, y))
        look = math.isfinite(left) and left <= self.stand_m
        if look and self.look_los and (left > self.los_min_left) and (self.pending is not None) and (not self._sees(x, y, self.pending['point'])):
            look = False
        if look or stuck:
            return self._arrive(x, y, yaw, stuck=stuck)
        wp = self.nav.lookahead(self.field, x, y, 0.6)
        if wp is None:
            return self._arrive(x, y, yaw, stuck=True)
        return self._walk_or_turn(x, y, yaw, math.atan2(wp[1] - y, wp[0] - x))

    def summary(self) -> dict:
        s = super().summary()
        s.update(controller='confirm', confirms=self.confirms, rejects=self.rejects, rejected=[[round(v, 2) for v in p] for p in self.rejected], params=dict(far_m=self.far_m, stand_far_m=self.stand_far_m, stand_near_m=self.stand_near_m, near_ok_m=self.near_ok_m, max_confirms=self.max_confirms, max_rejects=self.max_rejects, near_look=self.near_look, ray_stand=self.ray_stand, look_los=self.look_los, chain_confirms=self.chain_confirms, max_total_confirms=self.max_total_confirms, los_min_left=self.los_min_left, los_target_h=self.los_target_h, los_clear_m=self.los_clear_m, snap_path=self.snap_path, snap_detour_m=self.snap_detour_m, snap_w=self.snap_w, snap_radius_m=self.snap_radius_m, consistency_m=self.consistency_m, max_consistency_looks=self.max_consistency_looks, guard_confirmation_cap=self.guard_confirmation_cap), snaps_moved=self.snaps_moved, consistency_looks=self.consistency_looks, cap_rejections=self.cap_rejections)
        return s
