# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
import time
from collections import deque, namedtuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from mujoco_g1_nav.category_teacher import HeightFog, instance_pixels
from mujoco_g1_nav.expert import FORWARD, LEFT, RIGHT, STOP
from mujoco_g1_nav.house_eval import category_mask
from mujoco_g1_nav.house_map import height_map
from mujoco_g1_nav.nav import rect_distance, wrap
from .camera import HFOV_DEG, AstraCamera
from .frontiers import frontier_waypoints
SPIN_TURNS = 12
CAPTURE_AT = (3, 6, 9, 12)
FLOOR_H = 0.05
FOG_RANGE_M = 10.0
GOAL_REACHED_M = 0.7
FOUND_REACHED_M = 0.3
STUCK_WINDOW, STUCK_MOVE_M = (16, 0.3)
MAX_DECISIONS = 60
View = namedtuple('View', 'rgb pose cat_px inst_px')

@dataclass
class SceneMemory:
    nav: object = None
    fog: object = None
    visited: list = field(default_factory=list)
    started: bool = False

class NullClient:

    def reset(self, *a, **k):
        pass

    def step(self, rgb, gt224=None, force=None, peek=False):
        return (1, dict(probs=[0.0] * 6, mask_px=-1))

    def override(self, action):
        pass

    def close(self):
        pass

class RemoteDecider:
    name = 'model'

    def __init__(self, client, image_length: int=50, color_order: str='rgb'):
        self.client, self.image_length, self.color_order = (client, image_length, color_order)

    def reset(self, goal: str):
        self.client.reset(goal, self.image_length, self.color_order)

    def set_goal(self, goal: str):
        self.client.set_goal(goal)

    def decide(self, views, ref, frontiers, goal: str) -> dict:
        t0 = time.time()
        r = self.client.decide([v.rgb for v in views], [v.pose for v in views], ref, frontiers, goal)
        kind = 'found' if r['found'] and r['target_xy'] is not None else 'frontier' if r['frontier_index'] is not None else 'none'
        return dict(kind=kind, target=r['target_xy'], frontier_index=r['frontier_index'], text=r['text'], seconds=time.time() - t0, n_tokens=r.get('n_tokens'), n_images=r.get('n_images'), n_history=r.get('n_history'), vision_encoded_images=r.get('vision_encoded_images'), vision_total_images=r.get('vision_total_images'))

class OracleDecider:
    name = 'oracle'

    def __init__(self, ctx, world, cat: str, see_px: int=120):
        self.ctx, self.world, self.cat, self.see_px = (ctx, world, cat, see_px)

    def reset(self, goal: str):
        pass

    def _found(self, views):
        best = None
        for v in views:
            for iid, px in v.inst_px.items():
                if px >= self.see_px and (best is None or px > best[1]):
                    best = (iid, px)
        if best is None:
            return None
        it = self.ctx.house.layout.by_id[best[0]]
        return dict(kind='found', target=(float(it.x), float(it.y)), frontier_index=None, text=f'{self.name}: found {it.id} ({best[1]} px)', seconds=0.0, n_tokens=None)

    def decide(self, views, ref, frontiers, goal: str) -> dict:
        found = self._found(views)
        if found is not None:
            return found
        if frontiers:
            field = self.world.field(self.cat)
            nav = self.ctx.navmap
            costs = [nav.value(field, fx, fy) if field is not None else math.hypot(fx - ref[0], fy - ref[1]) for fx, fy in frontiers]
            k = int(np.argmin(costs))
            return dict(kind='frontier', target=None, frontier_index=k, text=f'oracle: frontier {k} (cost {costs[k]:.1f} m)', seconds=0.0, n_tokens=None)
        return dict(kind='none', target=None, frontier_index=None, text='oracle: nothing to do', seconds=0.0, n_tokens=None)

class BlindDecider(OracleDecider):
    name = 'blind'

    def decide(self, views, ref, frontiers, goal: str) -> dict:
        found = self._found(views)
        if found is not None:
            return found
        if frontiers:
            d = [math.hypot(fx - ref[0], fy - ref[1]) for fx, fy in frontiers]
            k = int(np.argmin(d))
            return dict(kind='frontier', target=None, frontier_index=k, text=f'blind: nearest frontier {k} ({d[k]:.1f} m)', seconds=0.0, n_tokens=None)
        return dict(kind='none', target=None, frontier_index=None, text='blind: nothing to do', seconds=0.0, n_tokens=None)

class AstraNavController:

    def __init__(self, ctx, world, cat: str, phrase: str, decider, face: bool=False, cam: AstraCamera=None, max_frontiers: int=16, debug_dir=None, scene_memory: SceneMemory=None, reset_visual_memory: bool=False):
        self.ctx, self.world, self.cat, self.phrase, self.decider, self.face = (ctx, world, cat, phrase, decider, face)
        self.cfg, self.nav = (ctx.cfg, ctx.navmap)
        self.step_m = float(self.cfg['navigation']['forward_step_m'])
        self.turn_tol = math.radians(float(self.cfg['expert']['turn_tol_deg']))
        self.own_cam = cam is None
        self.cam = cam or AstraCamera(ctx)
        cfg_fog = {**self.cfg, 'camera': {**self.cfg['camera'], 'hfov_deg': HFOV_DEG}}
        layout = ctx.house.layout
        self.items = layout.of_cat(cat)
        if scene_memory is not None and scene_memory.nav is not None and (scene_memory.nav is not self.nav):
            raise ValueError('SceneMemory belongs to another scene; create a new SceneMemory after changing the layout')
        if scene_memory is None or scene_memory.fog is None:
            self.fog = HeightFog(self.nav, cfg_fog, height_map(layout, self.nav.X, self.nav.Y, self.nav.res), FLOOR_H, FOG_RANGE_M, rays=160)
            if scene_memory is not None:
                scene_memory.nav, scene_memory.fog = (self.nav, self.fog)
        else:
            self.fog = scene_memory.fog
        self.max_frontiers = max_frontiers
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.handover_step = 0
        self.phase, self.k, self.views, self.visited = ('spin', 0, [], [])
        if scene_memory is not None:
            self.visited = scene_memory.visited
        self.target = self.field = self.mode = self.found_xy = None
        self.history = deque(maxlen=STUCK_WINDOW)
        self.face_steps = self.idle = 0
        self.records = []
        self.model_seconds = 0.0
        self.fallbacks = 0
        if scene_memory is not None and scene_memory.started and (not reset_visual_memory):
            getattr(decider, 'set_goal', decider.reset)(phrase)
        else:
            decider.reset(phrase)
        if scene_memory is not None:
            scene_memory.started = True

    def __call__(self, o, t: int) -> int:
        x, y, yaw = (float(o.x), float(o.y), float(o.yaw))
        self.t = t
        self.fog.update(x, y, yaw)
        if self.phase == 'spin':
            return self._spin(x, y, yaw)
        if self.phase == 'follow':
            return self._follow(x, y, yaw)
        if self.phase == 'face':
            return self._face(x, y, yaw)
        return STOP

    def status(self) -> str:
        if self.phase == 'spin':
            return f'AstraNav SPIN {self.k}/{SPIN_TURNS}'
        if self.phase == 'follow':
            return f'AstraNav -> {self.mode} ({self.target[0]:.1f}, {self.target[1]:.1f})'
        return f'AstraNav {self.phase}'

    def close(self):
        if self.own_cam:
            self.cam.close()

    def _spin(self, x, y, yaw) -> int:
        if self.k in CAPTURE_AT:
            self._capture(x, y, yaw)
        if self.k < SPIN_TURNS:
            self.k += 1
            return LEFT
        self.k = 0
        return self._decide(x, y, yaw)

    def _capture(self, x, y, yaw):
        data = self.ctx.sim
        rgb = self.cam.rgb(data)
        seg = self.cam.seg_ids(data)
        cm = category_mask(self.world, seg, self.cat)
        self.views.append(View(rgb, (x, y, yaw), int(cm.sum()), instance_pixels(self.ctx, seg, self.items)))

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
        rec = dict(t=self.t, pose=[round(x, 2), round(y, 2), round(yaw, 3)], n_frontiers=len(fr), kind=d['kind'], text=d['text'], seen_px=seen_px, seconds=round(float(d.get('seconds') or 0.0), 2), fr_cost=[round(c, 2) if math.isfinite(c) else None for c in costs], fr_index=d.get('frontier_index'))
        rec.update({k: d[k] for k in ('n_images', 'n_history', 'n_tokens', 'vision_encoded_images', 'vision_total_images') if d.get(k) is not None})
        if len(self.records) >= MAX_DECISIONS:
            self.records.append(rec)
            self.phase = 'done'
            return STOP
        if d['kind'] == 'found':
            snapped = self._snap_found(x, y, d['target'])
            if snapped is not None:
                self.found_xy = (float(d['target'][0]), float(d['target'][1]))
                rec['found_xy'] = [round(v, 2) for v in self.found_xy]
                rec['found_err_m'] = round(min((math.hypot(self.found_xy[0] - it.x, self.found_xy[1] - it.y) for it in self.items)), 2)
                rec['snapped'] = [round(snapped[0], 2), round(snapped[1], 2)]
                self.records.append(rec)
                return self._start_follow(x, y, yaw, snapped, 'found')
            rec['kind'] = 'found_unreachable'
            d = dict(d, kind='none')
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

    def _dump(self, views, d, frontiers, n):
        import cv2
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        tiles = []
        for name, v in zip(('+90', '+180', '+270', '0 (ref)'), views):
            img = cv2.resize(v.rgb, (360, 320))
            cv2.putText(img, f'{name} deg  cat px {v.cat_px}', (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            tiles.append(img)
        grid = np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0)
        bar = np.zeros((60, grid.shape[1], 3), np.uint8)
        cv2.putText(bar, f't={self.t} goal={self.phrase} frontiers={len(frontiers)}', (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, str(d['text'])[:110], (6, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 120), 1, cv2.LINE_AA)
        cv2.imwrite(str(self.debug_dir / f'decision_{n:02d}_t{self.t:03d}.png'), np.concatenate([grid, bar], axis=0)[..., ::-1])

    def _snap_found(self, x, y, obj_xy):
        nav = self.nav
        start = nav.nearest_free(x, y, radius=1.0)
        if start is None:
            return None
        d0 = nav.field([(float(nav.xs[start[1]]), float(nav.ys[start[0]]))])
        ii, jj = np.nonzero(np.isfinite(d0.dist) & nav.free)
        if len(ii) == 0:
            return None
        k = int(np.argmin(np.hypot(nav.xs[jj] - obj_xy[0], nav.ys[ii] - obj_xy[1])))
        return (float(nav.xs[jj[k]]), float(nav.ys[ii[k]]))

    def _start_follow(self, x, y, yaw, target, mode) -> int:
        self.target, self.mode = ((float(target[0]), float(target[1])), mode)
        self.field = self.nav.field([self.target])
        self.history.clear()
        if not math.isfinite(self.nav.value(self.field, x, y)) and self.nav.nearest_free(x, y, radius=1.0) is None:
            return self._arrive(x, y, yaw, stuck=True)
        self.phase = 'follow'
        return self._follow(x, y, yaw)

    def _follow(self, x, y, yaw) -> int:
        tx, ty = self.target
        dist = math.hypot(tx - x, ty - y)
        self.history.append((x, y))
        stuck = len(self.history) == self.history.maxlen and math.hypot(self.history[0][0] - x, self.history[0][1] - y) < STUCK_MOVE_M
        if dist < (FOUND_REACHED_M if self.mode == 'found' else GOAL_REACHED_M) or stuck:
            return self._arrive(x, y, yaw, stuck=stuck)
        wp = self.nav.lookahead(self.field, x, y, 0.6)
        if wp is None:
            return self._arrive(x, y, yaw, stuck=True)
        return self._walk_or_turn(x, y, yaw, math.atan2(wp[1] - y, wp[0] - x))

    def _arrive(self, x, y, yaw, stuck=False) -> int:
        if stuck and self.records:
            self.records[-1]['stuck'] = True
        if self.mode == 'found':
            if self.face and self.found_xy is not None:
                self.phase, self.face_steps = ('face', 0)
                return self._face(x, y, yaw)
            self.phase = 'done'
            return STOP
        self.phase, self.k = ('spin', 0)
        return self._spin(x, y, yaw)

    def _face(self, x, y, yaw) -> int:
        err = wrap(math.atan2(self.found_xy[1] - y, self.found_xy[0] - x) - yaw)
        if abs(err) <= math.radians(15.0) or self.face_steps >= 8:
            self.phase = 'done'
            return STOP
        self.face_steps += 1
        return LEFT if err > 0 else RIGHT

    def _walk_or_turn(self, x, y, yaw, desired) -> int:
        err = wrap(desired - yaw)
        if abs(err) > self.turn_tol:
            return LEFT if err > 0 else RIGHT
        e = self.cfg['expert']
        nx, ny = (x + self.step_m * math.cos(yaw), y + self.step_m * math.sin(yaw))
        if self.nav.wall_clearance(nx, ny) >= e['wall_margin_m'] and float(rect_distance(nx, ny, self.cfg)) >= e['step_clearance_min_m']:
            return FORWARD
        turn = math.radians(self.cfg['navigation']['turn_step_deg'])
        room = {}
        for act, sgn in ((LEFT, 1), (RIGHT, -1)):
            px, py = (x + self.step_m * math.cos(yaw + sgn * turn), y + self.step_m * math.sin(yaw + sgn * turn))
            room[act] = min(self.nav.wall_clearance(px, py), float(rect_distance(px, py, self.cfg)))
        return max(room, key=room.get)

    def summary(self) -> dict:
        found = [r for r in self.records if r['kind'] == 'found']
        missed = [r for r in self.records if r['kind'] != 'found' and r['seen_px'] >= getattr(self.decider, 'see_px', 120)]
        return dict(decider=self.decider.name, decisions=len(self.records), found_decisions=len(found), frontier_decisions=sum((r['kind'] == 'frontier' for r in self.records)), fallbacks=self.fallbacks, missed_visible=len(missed), model_seconds=round(self.model_seconds, 2), face=self.face, found_err_m=[r.get('found_err_m') for r in found], found_seen_px=[r['seen_px'] for r in found], records=self.records)
