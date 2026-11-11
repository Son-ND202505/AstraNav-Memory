# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
"""AstraNav-Memory model server with a fine-tune checkpoint (LoRA + compressor + merger loaded on top of the base weights). Prints READY once it listens."""
import argparse
import hashlib
import sys
from pathlib import Path
BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH / 'server'))
sys.path.insert(0, str(BENCH / 'src'))
import astranav_server as S
from astranav_adapter import lora as L

class FtRunner(S.ModelRunner):

    def __init__(self, weights: str, ft: str, attn: str='sdpa', device: str='cuda:0'):
        super().__init__(weights, attn, device)
        torch = self.torch
        ck = torch.load(ft, map_location='cpu')
        digest = hashlib.sha256()
        with open(ft, 'rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(block)
        self.checkpoint_sha256 = digest.hexdigest()
        if 'lora' not in ck:
            ck = ck['ft']
        meta = ck.get('meta', {})
        L.apply_lora(self.model, int(meta.get('rank', 32)), float(meta.get('alpha', 64.0)), 0.0)
        L.enable_vision_training(self.model)
        L.load_ft_into_wrapped(self.model, ck)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.name = f"{self.name} + fine-tune {Path(ft).parent.parent.name}/{Path(ft).name} (step {meta.get('step')}, {meta.get('samples')} samples)"
        print('fine-tune loaded:', ft, {k: meta.get(k) for k in ('step', 'samples', 'rank', 'alpha')}, flush=True)

    def generate(self, pr):
        with self.torch.autocast('cuda', dtype=self.torch.bfloat16):
            return super().generate(pr)

    def first_token_logprobs(self, pr, words=('the', 'The', 'From')):
        with self.torch.autocast('cuda', dtype=self.torch.bfloat16):
            return super().first_token_logprobs(pr, words)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', default=str(BENCH / 'weights' / 'ds16_context100'))
    ap.add_argument('--ft', required=True)
    ap.add_argument('--port', type=int, default=8301)
    ap.add_argument('--attn', default='sdpa')
    a = ap.parse_args()
    S.Server(FtRunner(a.weights, a.ft, a.attn)).serve(a.port)
if __name__ == '__main__':
    main()
