# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import json
import re
from dataclasses import dataclass, field
import numpy as np
from .pose import from_local, habitat_to_mujoco_xy, relative_pose, to_local
FIRST_OBSERVATION = 'This is the first observation point, there are no historical images.'
PANORAMA = 'Then, I have four images that form a 360-degree panoramic view around a central location.\nEach image has a field of view (FOV) of 110 degrees and is taken at 90-degree intervals. The fourth image defines the reference orientation (its forward direction is 0 degrees).The first three images are:<image><image><image>\nThe fourth reference image is:<image>'
NO_FRONTIER = 'No explorable frontiers found. Directly identify the target object or location from the user instruction and output the target'
_COORD = re.compile('<coordinate>\\s*(\\[.*?\\])\\s*</coordinate>')

class Bank:

    def __init__(self, keep: int=64):
        self.keep, self.images, self.states = (keep, [], [])

    def add(self, images, states):
        self.images.extend(images)
        self.states.extend(states)
        if len(self.images) > self.keep:
            del self.images[:-self.keep]
            del self.states[:-self.keep]

    def clear(self):
        self.images, self.states = ([], [])

    def __len__(self):
        return len(self.images)

@dataclass
class Prompt:
    text: str
    images: list
    frontier_keys: dict = field(default_factory=dict)
    n_history: int = 0

def build_prompt(images, states, ref, frontiers, goal: str, image_length: int=50) -> Prompt:
    if len(images) < 5:
        part1 = FIRST_OBSERVATION
        prompt_images = list(images[-image_length + 1:])
        n_history = 0
    else:
        hist = states[-image_length + 1:-4]
        coords = [relative_pose(s, ref) for s in hist]
        part1 = 'Firstly, we provide all the historical memory images: ' + ''.join((f'[{x:.2f}, {y:.2f},yaw:{yaw:.2f}]<image>' for x, y, yaw in coords))
        prompt_images = list(images[-image_length + 1:])
        n_history = len(hist)
    keys = {}
    if len(frontiers):
        parts = ['The coordinates of the explorable frontiers are: ']
    else:
        parts = [NO_FRONTIER]
    for i, f in enumerate(frontiers):
        local = to_local(f, ref)
        x, z = (local[0], local[2])
        keys[x, z] = i
        parts.append(f'[{x:.2f}, {z:.2f}]')
    text = '\n'.join([part1, PANORAMA, ''.join(parts), 'instruction: ' + goal])
    if text.count('<image>') != len(prompt_images):
        raise ValueError(f"{text.count('<image>')} <image> markers for {len(prompt_images)} images")
    return Prompt(text, prompt_images, keys, n_history)

def parse_response(text: str):
    match = _COORD.search(text)
    found = 'found' in text
    if match:
        try:
            coord = json.loads(match.group(1))
            if isinstance(coord, list):
                return (coord, found)
        except json.JSONDecodeError:
            return (None, found)
    return (None, found)

def resolve(text: str, prompt: Prompt, ref) -> dict:
    coord, found = parse_response(text)
    out = dict(found=found, coord=None, target_xy=None, frontier_index=None, frontier_dist=None)
    if coord is None or len(coord) < 2:
        return out
    try:
        xz = [float(coord[0]), float(coord[1])]
    except (TypeError, ValueError):
        return out
    out['coord'] = xz
    if found:
        world = from_local(np.array([xz[0], 0.0, xz[1]]), ref)
        out['target_xy'] = habitat_to_mujoco_xy(world)
    else:
        best = None
        for key, idx in prompt.frontier_keys.items():
            d = float(np.linalg.norm(np.array(xz) - np.array(key)))
            if best is None or d < best[0]:
                best = (d, idx)
        if best is not None:
            out['frontier_index'], out['frontier_dist'] = (best[1], best[0])
    return out
