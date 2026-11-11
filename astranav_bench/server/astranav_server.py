# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
"""AstraNav-Memory model server (original weights). One process holds the model on the GPU; every client connection has its own memory bank. Prints READY once it listens."""
import argparse
import re
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH / 'src'))
import numpy as np
from astranav_adapter import pose, prompt, protocol

class StubRunner:
    name, compression_times, stub = ('stub', 2, True)

    def __init__(self, mode: str='first'):
        self.mode = mode

    def first_token_logprobs(self, pr, words=('the', 'The', 'From')):
        return {w: -1.0 for w in words}

    def generate(self, pr):
        m = re.search('explorable frontiers are: \\[(-?[\\d.]+), (-?[\\d.]+)\\]', pr.text)
        if self.mode == 'found' or m is None:
            return ('the target has been found at coordinate <coordinate>[0.00, 1.00]</coordinate>.', dict(n_tokens=0))
        return (f'go to the frontier at <coordinate>[{m.group(1)}, {m.group(2)}]</coordinate>.', dict(n_tokens=0))

def make_inputs(processor, pr, compression_times: int, dino_size: str):
    from PIL import Image
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': pr.text}]}]
    owner = processor if getattr(processor, 'chat_template', None) else processor.tokenizer
    text = owner.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = text.replace('<|vision_start|><|image_pad|><|vision_end|>', '')
    text = text.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
    pil = [Image.fromarray(np.ascontiguousarray(im)).resize((720, 640)) for im in pr.images]
    inputs = processor(text=text, images=pil, padding=True, padding_side='left', return_tensors='pt', use_compression=True, compression_times=compression_times, dino_size=dino_size)
    return (inputs, text)

class ModelRunner:
    stub = False

    def __init__(self, weights: str, attn: str='sdpa', device: str='cuda:0'):
        sys.path.insert(0, str(BENCH / 'pylibs'))
        sys.path.insert(0, str(BENCH.parent / 'train_code' / 'transformers-4.57.1' / 'src'))
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as M

        def load_dino(self, device=None, dtype=torch.bfloat16):
            self.dino.to(device=device, dtype=dtype)
            for p in self.dino.parameters():
                p.requires_grad = False
            self.dino.eval()
        M.Qwen2_5_VisionTransformerPretrainedModel.load_dino = load_dino
        self.torch, self.device, self.weights = (torch, device, weights)
        t0 = time.time()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(weights, dtype=torch.bfloat16, attn_implementation=attn).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(weights, min_pixels=56 * 56, max_pixels=4480 * 4480)
        self.check_not_broken()
        cfg = self.model.config
        self.compression_times = int(getattr(cfg, 'compression_times', None) or cfg.vision_config.compression_times)
        self.dino_size = str(getattr(cfg, 'dino_size', None) or cfg.vision_config.dino_size)
        self.name = f'{Path(weights).name} (compression_times {self.compression_times}, {attn})'
        print(f'model loaded in {time.time() - t0:.0f} s, VRAM {torch.cuda.memory_allocated() / 1000000000.0:.2f} GB', flush=True)

    def check_not_broken(self):
        lm = self.model.model.language_model
        zero = [name for name, p in [('model.norm', lm.norm.weight)] + [(f'layers.{i}.mlp.down_proj', lm.layers[i].mlp.down_proj.weight) for i in (-3, -2, -1)] if float(p.detach().abs().max()) == 0.0]
        if zero:
            raise RuntimeError(f'broken checkpoint {self.weights}: all-zero tensors {zero} (see scripts/scan_zero_tensors.py); use another checkpoint')

    def generate(self, pr):
        torch = self.torch
        inputs, _ = make_inputs(self.processor, pr, self.compression_times, self.dino_size)
        inputs = inputs.to(self.device)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=128)
        new = out[:, inputs.input_ids.shape[1]:]
        answer = self.processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        info = dict(n_tokens=int(inputs.input_ids.shape[1]), n_new=int(new.shape[1]), vram_peak_gb=round(torch.cuda.max_memory_allocated() / 1000000000.0, 2))
        if not answer.strip():
            info['raw_answer'] = self.processor.batch_decode(new, skip_special_tokens=False)[0][:200]
        return (answer, info)

    def first_token_logprobs(self, pr, words=('the', 'The', 'From')):
        torch = self.torch
        inputs, _ = make_inputs(self.processor, pr, self.compression_times, self.dino_size)
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        out = {}
        for w in words:
            ids = self.processor.tokenizer.encode(w, add_special_tokens=False)
            out[w] = float(lp[ids[0]]) if len(ids) == 1 else None
        return out

class Session:

    def __init__(self):
        self.bank, self.goal, self.image_length, self.color_order = (prompt.Bank(), '', 50, 'rgb')
        self.vision_cache = {}

class Server:

    def __init__(self, runner):
        self.runner, self.lock = (runner, threading.Lock())

    def handle(self, conn):
        sess = Session()
        try:
            while True:
                head, payload = protocol.receive(conn)
                op = head.get('op')
                try:
                    if op == 'hello':
                        reply = dict(ok=True, model=self.runner.name, stub=bool(self.runner.stub), compression_times=self.runner.compression_times, checkpoint_sha256=getattr(self.runner, 'checkpoint_sha256', None), image_embedding_cache=bool(getattr(self.runner, 'image_embedding_cache', False)))
                    elif op == 'reset':
                        sess = Session()
                        sess.goal, sess.image_length, sess.color_order = (head['goal'], int(head.get('image_length', 50)), head.get('color_order', 'rgb'))
                        sess.bank = prompt.Bank(keep=sess.image_length + 8)
                        reply = dict(ok=True)
                    elif op == 'set_goal':
                        sess.goal = head['goal']
                        reply = dict(ok=True, n_images=len(sess.bank))
                    elif op == 'decide':
                        reply = self.decide(sess, head, payload)
                    elif op == 'quit':
                        return
                    else:
                        reply = dict(error=f'unknown op {op!r}')
                except Exception as exc:
                    traceback.print_exc()
                    reply = dict(error=f'{type(exc).__name__}: {exc}')
                protocol.send(conn, reply)
        except (ConnectionError, OSError):
            pass
        finally:
            conn.close()

    def decide(self, sess, head, payload):
        n, h, w = (int(head['n']), int(head['h']), int(head['w']))
        frames = np.frombuffer(payload, dtype=np.uint8).reshape(n, h, w, 3)
        if sess.color_order == 'bgr':
            frames = frames[..., ::-1]
        views = [np.ascontiguousarray(frames[i]) for i in range(n)]
        sess.bank.add(views, [pose.mj_state(*p) for p in head['poses']])
        ref = pose.mj_state(*head['ref'])
        frontiers = [pose.mj_position(fx, fy) for fx, fy in head['frontiers']]
        pr = prompt.build_prompt(sess.bank.images, sess.bank.states, ref, frontiers, head.get('goal') or sess.goal, sess.image_length)
        pr.vision_cache = sess.vision_cache
        t0 = time.time()
        with self.lock:
            answer, info = self.runner.generate(pr)
        res = prompt.resolve(answer, pr, ref)
        extra = {}
        if head.get('score'):
            with self.lock:
                extra['first_token_logprobs'] = self.runner.first_token_logprobs(pr)
        return dict(text=answer, **extra, n_images=len(pr.images), n_history=pr.n_history, seconds=round(time.time() - t0, 3), color_order=sess.color_order, view0_mean_rgb=[round(float(v), 1) for v in pr.images[-4].reshape(-1, 3).mean(0)], **info, **res)

    def serve(self, port: int, host: str='127.0.0.1', ready=None):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(8)
        print(f'READY port {port} model {self.runner.name}', flush=True)
        if ready is not None:
            ready.set()
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', default=str(BENCH / 'weights' / 'ds16_context100'))
    ap.add_argument('--port', type=int, default=8300)
    ap.add_argument('--attn', default='sdpa')
    ap.add_argument('--stub', nargs='?', const='first', default=None, help="no model: 'first' (default) or 'found'")
    a = ap.parse_args()
    runner = StubRunner(a.stub) if a.stub else ModelRunner(a.weights, a.attn)
    Server(runner).serve(a.port)
if __name__ == '__main__':
    main()
