# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
"""MuJoCo G1 object-goal navigation demo with AstraNav-Memory (the model decides at every 360-degree spin; the controller walks).   (MuJoCo env, run from this repo's root)

  python astranav_bench/scripts/play_house_astranav.py                         # fine-tuned weights, house home25, MuJoCo viewer + a window with what the model sees
  python astranav_bench/scripts/play_house_astranav.py --house l28             # the L-shaped house
  python astranav_bench/scripts/play_house_astranav.py --base                  # the original AstraNav-Memory weights (ds16_context100)
  python astranav_bench/scripts/play_house_astranav.py --no-viewer --speed 0 --cmd "sofa" "bed"    # headless, scripted

Needs mujoco_g1_nav of the OVSegDT fork: a symlink ./mujoco_g1_nav in this repo's root, or G1NAV_ROOT=/path/to/OVSegDT/mujoco_g1_nav.
The model server is started here (one GPU, ~9 GB). Goals of the same scene are chained and the memory (image history, explored map) is kept across them.

Commands:  <goal text> | new | again | objects | where <goal> | text canonical|raw | face on|off | frontiers N | speed X | video on|off | colours on|off | fpv on|off|size X | help | quit"""
import argparse
import contextlib
import ctypes
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parent
G1NAV = Path(os.environ.get('G1NAV_ROOT', REPO / 'mujoco_g1_nav'))
sys.path.insert(0, str(G1NAV / 'scripts'))
sys.path.insert(0, str(G1NAV / 'src'))
sys.path.insert(0, str(BENCH / 'src'))
os.environ.setdefault('MUJOCO_GL', 'egl')
HELP = __doc__.split('Commands:')[1].strip()
ACT = ['STOP', 'F', 'L', 'R', 'u', 'd']
OUT = BENCH / 'outputs' / 'play_astranav'

def _die_with_parent():
    ctypes.CDLL('libc.so.6').prctl(1, signal.SIGTERM)

def start_server(a, log_path: Path):
    env = dict(os.environ)
    env['PYTHONPATH'] = ':'.join([str(BENCH / 'pylibs'), str(REPO / 'train_code' / 'transformers-4.57.1' / 'src'), env.get('PYTHONPATH', '')])
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if a.base:
        cmd = [sys.executable, str(BENCH / 'server' / 'astranav_server.py'), '--port', str(a.astra_port), '--weights', a.weights]
    else:
        cmd = [sys.executable, str(BENCH / 'server' / 'astranav_server_ft.py'), '--port', str(a.astra_port), '--weights', a.weights, '--ft', a.ft]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, env=env, stdout=open(log_path, 'w'), stderr=subprocess.STDOUT, preexec_fn=_die_with_parent)
    return proc

def wait_ready(proc, log_path: Path, timeout: float=900.0):
    t0 = time.time()
    while 'READY' not in log_path.read_text():
        if proc.poll() is not None:
            raise RuntimeError(f'the model server died, see {log_path}')
        if time.time() - t0 > timeout:
            raise TimeoutError('the model server did not become ready')
        time.sleep(1.0)

class _NoPolicy:

    def __init__(self, *a, **k):
        pass

    def close(self):
        pass

class _NoField:

    def field(self, cat):
        return None

class _Quiet:

    def __init__(self, out):
        self.out, self.skip_newline = (out, False)

    def write(self, s):
        if 'starting the policy server' in s:
            self.skip_newline = True
            return len(s)
        if self.skip_newline and s == '\n':
            self.skip_newline = False
            return 1
        return self.out.write(s)

    def flush(self):
        self.out.flush()

    def __getattr__(self, k):
        return getattr(self.out, k)

class _FakeViewer:
    cam = None

    def is_running(self):
        return True

    def sync(self):
        pass

    def clear_images(self):
        pass

    def lock(self):
        return contextlib.nullcontext()

class _DumpWindow:
    alive = True

    def __init__(self, folder: Path, every: int=25):
        self.folder, self.every, self.n = (Path(folder), every, 0)
        self.folder.mkdir(parents=True, exist_ok=True)

    def show(self, rgb):
        import cv2
        self.n += 1
        if self.n % self.every == 0:
            cv2.imwrite(str(self.folder / f'panel_{self.n:05d}.png'), rgb[..., ::-1])
        return True

    def close(self):
        pass

def make_object_controller(ctx, obj_xy, phrase, decider, **kw):
    from astranav_adapter.controller import AstraNavController, View

    class ObjectGoalController(AstraNavController):

        def _capture(self, x, y, yaw):
            self.views.append(View(self.cam.rgb(self.ctx.sim), (x, y, yaw), 0, {}))
    ctrl = ObjectGoalController(ctx, _NoField(), 'object', phrase, decider, **kw)
    ctrl.items = [SimpleNamespace(x=float(obj_xy[0]), y=float(obj_xy[1]), id='object')]
    return ctrl

def _load_play_house():
    import play_house
    return play_house.PlayHouse
PlayHouse = _load_play_house()

class PlayAstra(PlayHouse):
    _acam = None

    def new_scene(self, keep_seed=False):
        from astranav_adapter.controller import SceneMemory
        self.scene_memory = SceneMemory()
        return super().new_scene(keep_seed=keep_seed)

    def __init__(self, a):
        import mujoco_g1_nav.policy_client as pc
        pc.PolicyClient = _NoPolicy
        self.server_proc = None
        log = OUT / 'server.log'
        if not a.server:
            self.server_proc = start_server(a, log)
        with contextlib.redirect_stdout(_Quiet(sys.stdout)):
            super().__init__(a)
        from astranav_adapter.controller import RemoteDecider
        from astranav_adapter.protocol import AstraNavClient
        if a.server:
            host, port = a.server.rsplit(':', 1)
        else:
            host, port = ('127.0.0.1', str(a.astra_port))
            print(f"waiting for the AstraNav server ({('original weights' if a.base else Path(a.ft).name)}); log {log} ...", flush=True)
            wait_ready(self.server_proc, log)
        self.astra = AstraNavClient(host, int(port))
        print('model server:', self.astra.hello(), flush=True)
        image_length = a.image_length or (100 if 'context100' in a.weights else 50)
        self.decider = RemoteDecider(self.astra, image_length, 'rgb')
        self.face, self.max_frontiers, self.controller = (not a.no_face, a.max_frontiers, a.controller)
        self.reset_memory_per_goal = a.reset_memory_per_goal
        self.guard_confirmation_cap = a.guard_confirmation_cap
        self.recover_sweeps, self.diag = (a.recover_sweeps, a.diag)
        self.look_los, self.chain_confirms, self.snap_path, self.face_blob = (True, True, True, True) if a.best else (a.look_los, a.chain_confirms, a.snap_path, a.face_blob)
        if a.dump_fpv:
            self._fpv_shut()
            self.viewer, self.fpv, self.fpv_mode, self.fpv_win = (_FakeViewer(), True, 'window', _DumpWindow(a.dump_fpv))
        print('\nready: type a goal (see `help`), e.g.  coke   bed   sofa   something to sit on', flush=True)

    @property
    def acam(self):
        if self._acam is None:
            from astranav_adapter.camera import AstraCamera
            self._acam = AstraCamera(self.ctx)
        return self._acam

    def _fpv_open(self):
        from astranav_adapter.camera import HEIGHT, WIDTH
        from mujoco_g1_nav.fpv_window import FpvWindow
        self.fpv_win = FpvWindow(aspect=WIDTH / HEIGHT, rel_height=self.fpv_size if self.fpv_size else 0.6, title='AstraNav camera (what the model sees)')
        if not self.fpv_win.alive:
            print(f'(cannot open the first-person window: {self.fpv_win.status}; continuing without it)', flush=True)
            self.fpv_win, self.fpv = (None, False)
            return False
        return True

    def _hud_lines(self):
        h = self.hud
        if not h or 'cat' not in h:
            return ['type a goal in the terminal (coke, bed, sofa ...)']
        goal = f"goal: {str(h['cat']).replace('_', ' ')}   step {h.get('t', 0)}"
        if h.get('verdict'):
            return [goal, h['verdict'], f"nearest {h['final_d']:.2f} m" if h.get('final_d') is not None else '']
        d = h.get('d')
        return [goal, h.get('status', ''), (h.get('last', '') + (f'   d={d:.1f} m' if d is not None and d == d else '')).strip()]

    def _fpv_draw(self, force=False):
        if self.viewer is None or not self.fpv or self.fpv_win is None:
            return
        now = time.time()
        if not force and now - self._fpv_t < 1.0 / 15:
            return
        self._fpv_t = now
        try:
            if not self.fpv_win.alive:
                self._fpv_shut()
                self.fpv = False
                print('\n(the first-person window was closed - type `fpv on` to open it again)', flush=True)
                return
            from astranav_adapter.camera import HEIGHT, WIDTH
            from mujoco_g1_nav.fpv_overlay import ACCENT, panel_image
            rgb = self.acam.rgb(self.ctx.sim)
            self.fpv_win.show(panel_image(rgb, WIDTH, HEIGHT, None, 'ASTRANAV CAMERA (level, 110 deg: what the model sees)', self._hud_lines(), ACCENT.get(self.hud.get('verdict_kind'), ACCENT[None])))
        except Exception as e:
            self.fpv = False
            print(f'\n(first-person panel switched off: {type(e).__name__}: {e})', flush=True)

    def fpv_command(self, args):
        try:
            if args[0] == 'on':
                self.fpv = True
                if self.viewer is not None and (self.fpv_win is None or not self.fpv_win.alive):
                    self._fpv_shut()
                    self._fpv_open()
            elif args[0] == 'off':
                self.fpv = False
                self._fpv_shut()
            elif args[0] == 'size' and len(args) > 1:
                self.fpv_size = float(args[1])
                if self.viewer is not None and self.fpv:
                    self._fpv_shut()
                    self._fpv_open()
            else:
                raise ValueError
        except ValueError:
            print('usage: fpv on|off | fpv size 0.1-0.95')
            return
        print(f"first-person view: {('on' if self.fpv else 'off')}, size {self.fpv_size:g}" + ('' if self.viewer is not None else '   (no MuJoCo window open)'))
        self._fpv_draw(force=True)

    def go(self, text):
        from mujoco_g1_nav.furniture import CATEGORIES
        goal = self.interpret(text)
        if goal is None:
            return
        if self.text_mode == 'raw':
            phrase = text
        elif goal.kind == 'category':
            phrase = CATEGORIES[goal.key].prompts[0]
        else:
            phrase = goal.canonical
        print(f"the model is told: '{phrase}'", flush=True)
        self._go_category(goal.key, phrase) if goal.kind == 'category' else self._go_object(goal, phrase)

    def _announce(self, ctrl, label, n):
        while n < len(ctrl.records):
            r = ctrl.records[n]
            n += 1
            raw = str(r.get('text', '')).replace('<coordinate>', '').replace('</coordinate>', '').strip()
            print()
            if r['kind'] == 'found':
                fx, fy = r['found_xy']
                err = r.get('found_err_m')
                print(f'''  decision {n} (step {r['t']}): model: "{raw}"  -> FOUND, world ({fx:+.1f}, {fy:+.1f})''' + (f', {err:.2f} m from the nearest real {label}' if err is not None else ''), flush=True)
                self.hud['last'] = f'decision {n}: FOUND ({fx:+.1f}, {fy:+.1f})'
            elif r['kind'] == 'frontier' and 'frontier' in r:
                fx, fy = r['frontier']
                k = r.get('fr_index')
                print(f'''  decision {n} (step {r['t']}): model: "{raw}"  -> not seen, frontier {(k + 1 if k is not None else '?')}/{r['n_frontiers']} at ({fx:+.1f}, {fy:+.1f})''', flush=True)
                self.hud['last'] = f"decision {n}: frontier {(k + 1 if k is not None else '?')}/{r['n_frontiers']}"
            else:
                print(f'''  decision {n} (step {r['t']}): model: "{raw}"  -> no usable answer ({r['kind']}): fallback''', flush=True)
                self.hud['last'] = f'decision {n}: no usable answer'
        return n

    def _video(self, cat, ctrl, world):
        if not self.video:
            return None
        from mujoco_g1_nav.house_video import PolicyVideo
        OUT.mkdir(parents=True, exist_ok=True)

        class AstraVideo(PolicyVideo):
            driver = property(lambda self: ctrl.status(), lambda self, v: None)
        return AstraVideo(OUT / f"{time.strftime('%H%M%S')}_seed{self.seed}_{cat}.mp4", self.ctx, cat, world)

    def _go_category(self, cat, phrase):
        from astranav_adapter.controller import AstraNavController, NullClient
        from mujoco_g1_nav.house_eval import run_category_episode
        world = self.sess.world
        shortest = world.path_length(cat, self.cur[0], self.cur[1])
        if self.controller == 'confirm' and self.face_blob:
            from astranav_adapter.stand import FaceBlobConfirmController as Ctl
        elif self.controller == 'confirm':
            from astranav_adapter.confirm import ConfirmAstraNavController as Ctl
        else:
            Ctl = AstraNavController
        kw = dict(look_los=self.look_los, chain_confirms=self.chain_confirms, snap_path=self.snap_path, guard_confirmation_cap=self.guard_confirmation_cap) if self.controller == 'confirm' else {}
        if self.recover_sweeps or self.diag:
            from astranav_adapter.recovery import with_recovery
            Ctl, kw = (with_recovery(Ctl), dict(kw, recover_sweeps=self.recover_sweeps))
        ctrl = Ctl(self.ctx, world, cat, phrase, self.decider, face=self.face, cam=self.acam, max_frontiers=self.max_frontiers, scene_memory=None if self.reset_memory_per_goal else self.scene_memory, **kw)
        vid = self._video(cat, ctrl, world)
        self.hud = dict(cat=cat, t=0, status=ctrl.status(), d=shortest)
        seen = [0]

        def hook(ctx, t, idx, info, m):
            self.hud.update(t=t + 1, status=ctrl.status(), d=float(m['d']))
            seen[0] = self._announce(ctrl, cat, seen[0])
            if vid is not None:
                vid(ctx, t, idx, info, m)
            print(f"\r  step {t + 1:3d}: {ACT[idx]:4s} {ctrl.status():36s} d={m['d']:.2f} m   ", end='', flush=True)
        hook.driver = 'astranav'
        res = run_category_episode(world, NullClient(), None, phrase, cat, (self.cur[0], self.cur[1], self.cur[2], shortest), 'astranav', self.max_steps, True, self.seed, None, on_step=hook, controller=ctrl)
        self.cur = (*(float(v) for v in self.ctx.loco.xy()), self.ctx.loco.yaw())
        if vid is not None:
            vid.close()
        print()
        self.last = res
        f = res['final']
        verdict = 'CORRECT - stopped in front of it' if res['success_front'] else 'NEAR - at the right place, not in front / a bit off' if res['success'] else 'NOT THERE' if not res['called_stop'] or not res['success_loc'] else 'stopped near it but outside the success region'
        print(f"result: {verdict}  | called stop: {('yes' if res['called_stop'] else 'no (out of steps)')} | {res['steps']} steps | path {res['path_m']:.1f} m (shortest {res['shortest_m']:.1f}) | contacts: {res['collision_steps']} steps")
        print(f"   nearest {f['item']}: {f['obj_dist']:.2f} m away, heading error {f['yaw_err_deg']:.0f} deg, {f['px']} px in the image, in its front sector: {('yes' if f['front'] else 'no')}")
        self._summary(ctrl)
        short = 'CORRECT: stopped in front' if res['success_front'] else 'NEAR: not in front / a bit off' if res['success'] else 'NOT THERE' if not res['called_stop'] or not res['success_loc'] else 'STOPPED OUTSIDE THE ZONE'
        self.hud.update(verdict=short, verdict_kind='correct' if res['success_front'] else 'near' if res['success'] else 'fail', final_d=f['obj_dist'])
        self._fpv_draw(force=True)

    def _summary(self, ctrl):
        s = ctrl.summary()
        print(f"   model: {s['decisions']} decisions ({s['found_decisions']} found, {s['frontier_decisions']} frontier, {s['fallbacks']} fallback), {s['model_seconds'] / max(s['decisions'], 1):.1f} s per decision")

    def _go_object(self, goal, phrase):
        from astranav_adapter.controller import NullClient
        from mujoco_g1_nav.front import frontality
        from mujoco_g1_nav.goal import object_xy
        from mujoco_g1_nav.policy_eval import VideoRecorder, run_policy_episode
        from mujoco_g1_nav.scenario import Scenario, building_episode_setup
        ctx, key = (self.ctx, goal.key)
        if key not in ctx.scene.obj_body:
            print(f"'{goal.label}' is registered as a goal but this scene has no such object on the table")
            return
        print('(a table object: NEW for the model - it was fine-tuned on furniture only; the success flags are those of play_house: distance, clearance from the table edge, heading)')
        x0, y0, yaw0 = self.cur
        scn = Scenario(building=self.env.building.name, seed=int(self.seed), scenario='S1a', table_room=self.env.building.origin, table_xy=tuple((float(v) for v in self.env.cfg['table']['center_xy'])), perm=tuple(self.drawn['perm']), mirror=tuple(self.drawn['mirror']), target=key, start_xy=(float(x0), float(y0)), start_yaw=float(yaw0), start_kind='play')
        holder, rec = ({}, None)

        def factory(c, lay, app):
            from astranav_adapter.controller import SceneMemory
            holder['ctrl'] = make_object_controller(c, object_xy(lay, key), phrase, self.decider, face=self.face, cam=self.acam, max_frontiers=self.max_frontiers, scene_memory=None if self.reset_memory_per_goal else SceneMemory(started=self.scene_memory.started))
            return holder['ctrl']
        if self.video:
            from mujoco_g1_nav.building_viz import TopDownRenderer
            OUT.mkdir(parents=True, exist_ok=True)
            sx, sy = self.env.cfg['table']['size_xy']
            rec = VideoRecorder(OUT / f"{time.strftime('%H%M%S')}_seed{self.seed}_{key}.mp4", phrase, topdown=TopDownRenderer(self.env.building, layout=self.env.layout), table=(*scn.table_xy, sx, sy), start=scn.start_xy)
        self.hud = dict(cat=goal.label, t=0, status='AstraNav starting', d=math.nan)
        seen = [0]

        def hook(c, o, t, idx, info):
            ctrl = holder['ctrl']
            ox, oy = object_xy(c.last_setup['layout'], key)
            d = math.hypot(ox - o.x, oy - o.y)
            self.hud.update(t=t + 1, status=ctrl.status(), d=d)
            seen[0] = self._announce(ctrl, goal.label, seen[0])
            if rec is not None:
                rec(c, o, t, idx, info)
            print(f'\r  step {t + 1:3d}: {ACT[idx]:4s} {ctrl.status():36s} d={d:.2f} m   ', end='', flush=True)
        try:
            r = run_policy_episode(ctx, NullClient(), {key: None}, self.seed, 'random', key, 'g1', 'astranav', self.max_steps, True, phrase=phrase, on_step=hook, setup=lambda: building_episode_setup(ctx, scn), controller_factory=factory)
        finally:
            if rec is not None:
                rec.close()
            self.sess.set_variant(self.sess.variant.index)
        print()
        if not r.get('valid'):
            print('no valid approach pose for this goal in this scene (type `new` for another scene)')
            return
        ls = ctx.last_setup
        x, y = (float(v) for v in ctx.loco.xy())
        fr = frontality(ctx.cfg, object_xy(ls['layout'], key), (x, y))
        r['success_front'] = bool(r['success_close'] and fr['front'])
        self.cur = (x, y, ctx.loco.yaw())
        self.last = r
        f = r['final']
        verdict = 'CORRECT - stopped in front of it' if r['success_front'] else 'NEAR - at the table, close but not in front / a bit off' if r['success'] else 'NOT THERE' if not r['called_stop'] or not r['success_loc'] else 'stopped near it but outside the success region'
        print(f"result: {verdict}  | called stop: {('yes' if r['called_stop'] else 'no (out of steps)')} | {r['steps']} steps | path {r['path_m']:.1f} m (shortest {r['shortest_m']:.1f}) | contacts: {r['collision_steps']} steps")
        print(f"   {goal.label}: {f['obj_dist']:.2f} m away, {f['clearance']:.2f} m from the table edge (success needs 0.28-0.65), heading error {f['yaw_err_deg']:.0f} deg, {f['px']} px in the image, in front of it: {('yes' if fr['front'] else 'no')}")
        self._summary(holder['ctrl'])
        short = 'CORRECT: stopped in front' if r['success_front'] else 'NEAR: not in front / a bit off' if r['success'] else 'NOT THERE' if not r['called_stop'] or not r['success_loc'] else 'STOPPED OUTSIDE THE ZONE'
        self.hud.update(verdict=short, verdict_kind='correct' if r['success_front'] else 'near' if r['success'] else 'fail', final_d=f['obj_dist'])
        self._fpv_draw(force=True)

    def close(self):
        try:
            if getattr(self, 'astra', None) is not None:
                self.astra.close()
            if self._acam is not None:
                self._acam.close()
            if self.server_proc is not None:
                self.server_proc.terminate()
        finally:
            super().close()

def build_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--house', default='home25', choices=['home25', 'l28'], help='home25 (default) or l28')
    ap.add_argument('--ft', default=str(BENCH / 'weights' / 'astranav_g1_ft.pt'), help='fine-tune checkpoint (LoRA + compressor + merger; astranav_bench/scripts/download_weights.sh)')
    ap.add_argument('--look-los', action='store_true', help='confirm controller: no look again until the announced point is visible (walls in between make the look blind)')
    ap.add_argument('--face-blob', action='store_true', help='confirm controller: at the end turn to the nearest part of the furniture piece next to the announced point')
    ap.add_argument('--best', action='store_true', help='= --look-los --chain-confirms --snap-path --face-blob')
    ap.add_argument('--snap-path', action='store_true', help="confirm controller: announced point behind a wall -> walk to a cell on the robot's side")
    ap.add_argument('--chain-confirms', action='store_true', help='confirm controller: the cap of 3 confirmations counts per chain of founds')
    ap.add_argument('--controller', default='confirm', choices=['original', 'confirm'], help='approach-then-confirm layer (default) or the original commit-at-first-found controller')
    ap.add_argument('--recover-sweeps', type=int, default=0, help='no frontier left: forget the explored map and sweep again, at most K times (0 = stop, as the original)')
    ap.add_argument('--reset-memory-per-goal', action='store_true', help='discard the memory at every goal (per-episode benchmark protocol; default: keep it within a scene)')
    ap.add_argument('--base', action='store_true', help='the ORIGINAL AstraNav-Memory weights instead of the fine-tuned model')
    ap.add_argument('--weights', default=str(BENCH / 'weights' / 'ds16_context100'))
    ap.add_argument('--image-length', type=int, default=None, help='images of the memory window (default 100 for ds16_context100)')
    ap.add_argument('--server', default=None, help='host:port of a running AstraNav server (default: one is started)')
    ap.add_argument('--astra-port', type=int, default=8302, help='port of the server that is started here (not 8300 / 8301 / 8075 / 8076)')
    ap.add_argument('--no-face', action='store_true', help='do not turn towards the announced target before stopping')
    ap.add_argument('--max-frontiers', type=int, default=4)
    ap.add_argument('--colours', action='store_true')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--speed', type=float, default=2.0, help='viewer playback speed (1 = real time, 0 = as fast as possible)')
    ap.add_argument('--max-steps', type=int, default=None, help='step budget of an episode (default: twice the evaluation budget, 900)')
    ap.add_argument('--no-viewer', action='store_true')
    ap.add_argument('--no-fpv', action='store_true')
    ap.add_argument('--fpv-size', type=float, default=None, help='height of the first-person window as a fraction of the screen height (default 0.6)')
    ap.add_argument('--text-mode', default='canonical', choices=['canonical', 'raw'], help='what the model is told: the category word of the evaluation (default) or your own words')
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--cmd', nargs='*', default=None, help='headless scripted goals (then exit)')
    ap.add_argument('--dump-fpv', default=None, help='headless self-test: save the first-person panel as PNG into this folder')
    ap.set_defaults(guard_confirmation_cap=False, diag=False, pool=None, variant='random', ckpt=None, port=0, perception='gt', e2e=False, fpv_mode='window', fpv_corner='tr', fpv_mask=False, skin=None, skin_text='new', gate_m=0.0, no_escape=True)
    a = ap.parse_args()
    if a.pool is None and a.variant == 'random':
        a.variant = 0
    if a.dump_fpv:
        a.no_viewer = True
    return a

def main():
    a = build_args()
    p = PlayAstra(a)
    try:
        if a.cmd:
            for c in a.cmd:
                print(f'\n> {c}')
                if c.lower() == 'new':
                    p.new_scene()
                elif c.lower() == 'again':
                    p.cur = p.start
                    p.ctx.loco.reset((p.start[0], p.start[1]), p.start[2])
                    p.ctx.executor.settle()
                else:
                    p.go(c)
            return
        while True:
            try:
                line = input('\n> ').strip()
            except EOFError:
                break
            if not line:
                continue
            w = line.split()
            cmd = w[0].lower()
            if cmd in ('quit', 'exit', 'q'):
                break
            elif cmd == 'help':
                print(HELP)
            elif cmd == 'new':
                p.new_scene()
            elif cmd == 'again':
                p.cur = p.start
                p.ctx.loco.reset((p.start[0], p.start[1]), p.start[2])
                p.ctx.executor.settle()
                p.hud = {}
                p._fpv_draw(force=True)
                print('back to the start pose (same scene); type a goal')
            elif cmd == 'objects':
                p.objects()
            elif cmd == 'where' and len(w) > 1:
                p.where(' '.join(w[1:]))
            elif cmd == 'face' and len(w) > 1 and (w[1] in ('on', 'off')):
                p.face = w[1] == 'on'
                print(f"turn towards the announced target before stopping: {('on' if p.face else 'off')}")
            elif cmd == 'frontiers' and len(w) > 1 and w[1].isdigit():
                p.max_frontiers = max(1, int(w[1]))
                print(f'frontiers offered per decision: {p.max_frontiers}')
            elif cmd == 'colours' and len(w) > 1:
                p.colours = w[1] == 'on'
                p.new_scene(keep_seed=True)
            elif cmd == 'speed' and len(w) > 1:
                p.speed = float(w[1])
            elif cmd == 'video' and len(w) > 1:
                p.video = w[1] == 'on'
            elif cmd == 'fpv' and len(w) > 1:
                p.fpv_command(w[1:])
            elif cmd == 'text' and len(w) > 1 and (w[1] in ('canonical', 'raw')):
                p.text_mode = w[1]
                print(f"the model is told {('the category word of the evaluation' if p.text_mode == 'canonical' else 'your own words')}")
            else:
                p.go(line)
    finally:
        p.close()
if __name__ == '__main__':
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
