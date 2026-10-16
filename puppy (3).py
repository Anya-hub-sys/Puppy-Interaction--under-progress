"""Webcam Puppy v0.2

A cartoon puppy (11 layered PNG parts, two-segment legs) composited onto your
webcam feed. Behaviour = mood value + weighted random choice. No ML, no cloud.
Breathing is your own After Effects animation, looped ping-pong style.

Run:   python puppy.py            (mediapipe hand + leg tracking)
       python puppy.py --mouse    (mouse acts as your hand, for testing)
Keys:  q / Esc = quit, d = debug overlay, m = toggle mirror, click = move puppy
       s = force sit/stand, [ ] = rear-leg fold amount, , . = sit height
Files: puppy_rig.json (anchors + AE parenting), puppy_breathing.json (torso
       scale keyframes, optional), assets/ folder with one PNG per part.
"""
import argparse
import json
import math
import os
import random
import re
import time

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

# ---------------- settings you can tweak ----------------
ASSET_DIR = "assets"
RIG_FILE = "puppy_rig.json"
BREATH_FILE = "puppy_breathing.json"   # optional; skip breathing if absent
BLINK_FILE = "puppy_blink.json"        # optional; skip blinking if absent
BLINK_INTERVAL = (2.5, 6.0)            # seconds between blinks, randomized
EYE_LOOK_GAIN = 0.02    # deg of pupil rotation per px of horizontal body offset
EYE_LOOK_MAX = 9.0      # deg clamp -- small, since it's a rotation standing in for a slide
PUPPY_SCALE = 0.45        # rig pixels -> screen pixels
FLOOR_Y = 0.92            # where the paws touch the floor (fraction of frame height)
AROUSAL_DECAY_S = 10.0    # mood falls to ~37% every this many seconds
PET_GAIN = 0.5            # how fast petting raises the mood
MIRROR = True             # flip the camera like a mirror (press m to toggle live)

# walk-to-you behaviour: petting slowly builds "bond"; once bonded enough,
# moving far enough away makes the puppy walk over to you.
BOND_GAIN = 0.12          # slower than arousal -- needs sustained petting, not a quick touch
BOND_DECAY_S = 45.0       # attachment fades slowly once petting stops
BOND_THRESHOLD = 0.3      # minimum bond before the puppy will follow you
WALK_TRIGGER_DIST = 260   # px: how far away you need to be to trigger a follow
WALK_STOP_DIST = 90       # px: how close the puppy gets before it stops
WALK_SPEED = 140          # px/sec the puppy moves while walking
WALK_CYCLE_HZ = 1.6       # steps per second (trot cycle speed)
WALK_LEG_AMP = 26         # degrees of upper-leg swing while walking

POSE_KEYS = ["torso_rot", "dy", "head_rot", "tail_rot",
             "fl", "fr", "rl", "rr",      # upper leg angles
             "fl2", "fr2", "rl2", "rr2",  # lower leg angles, relative to upper
             "eye_rot", "pupil_rot", "brow_l_rot", "brow_r_rot"]  # face


def zero_pose():
    return {k: 0.0 for k in POSE_KEYS}


# Base postures in degrees (clockwise positive). "2" keys bend the knee
# relative to the upper leg above it, so a folded rear leg = thigh swings
# forward (rl) while the lower leg folds back under it (rl2). Guesses --
# tune live with [ ] and , . once you see it on your real art.
BASE_POSTURES = {
    # Real authored sit-down end-pose (from the AE sit export), not a guess:
    # torso tucks back -30deg and drops ~162 rig px (scaled to screen px
    # below), head tilts up +34, both rear legs fold -34 with the lower-leg
    # segments left at rest (the sit pose doesn't bend them separately),
    # front legs ease back -5.
    "sit": dict(zero_pose(), torso_rot=-30, dy=round(161.503 * PUPPY_SCALE, 1),
                rl=-34, rr=-34, rl2=0, rr2=0,
                fl=-5, fr=-5, head_rot=34),
    "stand": dict(zero_pose()),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------------- breathing (authored in After Effects, looped) ----------------
class Breathing:
    """Loops through torso_scale_y_percent samples forever.

    loop="cycle": plays the samples forward once per period, then jumps back
    to the start -- use this when the authored clip already returns close to
    its starting scale by its last frame (a self-contained breath in+out).
    loop="pingpong": plays forward then mirrors backward through the same
    samples -- use this only if the clip captures just half a breath (e.g.
    the inhale) and the exhale should be its reverse.
    Forcing pingpong on a clip authored as a full cycle doubles the apparent
    cycle length and can make the timing/easing feel wrong.
    """

    def __init__(self, path):
        self.samples, self.step, self.loop = [1.0], 1.0, "cycle"
        if not os.path.exists(path):
            print(f"[info] no {path}; torso will not breathe")
            return
        with open(path) as f:
            data = json.load(f)
        vals = data.get("torso_scale_y_percent", [100.0])
        self.samples = [v / 100.0 for v in vals]
        n = max(len(self.samples) - 1, 1)
        self.step = data.get("duration_seconds", n / data.get("fps_samples_per_second", 10)) / n
        self.loop = data.get("loop", "cycle")

    def scale_at(self, t):
        n = len(self.samples)
        if n < 2:
            return self.samples[0]
        if self.loop == "pingpong":
            period = 2 * (n - 1) * self.step
            tt = t % period
            u = tt / self.step if tt <= (n - 1) * self.step else (period - tt) / self.step
        else:  # "cycle": forward only, then wrap to the start
            period = (n - 1) * self.step
            u = (t % period) / self.step
        i0 = int(u)
        i1 = min(i0 + 1, n - 1)
        f = u - i0
        return self.samples[i0] * (1 - f) + self.samples[i1] * f


class BlinkClip:
    """Authored one-shot blink curve: both eyebrows + the eye layer, sampled
    at a fixed rate (same shape export_animation.jsx produces). Played once
    per blink, then the brain goes idle until the next random interval."""

    def __init__(self, path):
        self.fps = 24.0
        self.eye_rot = [0.0]
        self.brow_l_rot = [0.0]
        self.brow_r_rot = [0.0]
        if not os.path.exists(path):
            print(f"[info] no {path}; eyes/eyebrows will not blink")
            return
        with open(path) as f:
            data = json.load(f)
        self.fps = data.get("fps", 24.0)
        self.eye_rot = data.get("eye_rot", [0.0])
        self.brow_l_rot = data.get("brow_l_rot", [0.0])
        self.brow_r_rot = data.get("brow_r_rot", [0.0])

    @property
    def duration(self):
        return (len(self.eye_rot) - 1) / self.fps

    def sample(self, t):
        """t: seconds since the blink started -> (eye_rot, brow_l_rot, brow_r_rot)."""
        n = len(self.eye_rot)
        u = clamp(t * self.fps, 0.0, n - 1)
        i0 = int(u)
        i1 = min(i0 + 1, n - 1)
        f = u - i0

        def lerp(arr):
            return arr[i0] * (1 - f) + arr[i1] * f

        return lerp(self.eye_rot), lerp(self.brow_l_rot), lerp(self.brow_r_rot)


# ---------------- rig / parts ----------------
def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def find_png(part_name):
    want = norm(part_name)
    for f in sorted(os.listdir(ASSET_DIR)):
        stem, ext = os.path.splitext(f)
        if ext.lower() == ".png" and norm(stem).endswith(want):
            return os.path.join(ASSET_DIR, f)
    raise FileNotFoundError(f"No PNG in '{ASSET_DIR}/' matching part '{part_name}'")


def classify(name):
    """-> (role, angle_key). role feeds z-order/breathing; angle_key feeds pose."""
    n = name.lower()
    if "torso" in n:
        return "torso", "torso_rot"
    if "head" in n:
        return "head", "head_rot"
    if "tail" in n:
        return "tail", "tail_rot"
    if "pupil" in n:
        return "pupil", "pupil_rot"
    if "eyebrow" in n or "eye brow" in n:  # covers "L eyebrow" and "R eye brow"
        side = "l" if n[0] == "l" else "r"
        return f"brow_{side}", f"brow_{side}_rot"
    if "eye" in n:  # checked after eyebrow, since "eyebrow" also contains "eye"
        return "eye", "eye_rot"
    limb = ("f" if "front" in n else "r") + ("r" if "right" in n else "l")
    key = limb + "2" if "lower" in n else limb
    return key, key


class Part:
    def __init__(self, name, info, scale):
        png_path = find_png(name)
        print(f"[match] '{name}' -> {png_path}")
        img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim < 3 or img.shape[2] != 4:
            raise ValueError(f"{name}: PNG must have a transparent (alpha) channel")
        ys, xs = np.where(img[..., 3] > 0)  # trim to visible pixels, like the AE layer bounds
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = img.shape[:2]
        rw, rh = info["size"]
        if abs(w - rw) > 3 or abs(h - rh) > 3:
            print(f"[warn] {name}: PNG is {w}x{h} but rig says {rw}x{rh}; anchors may be off")
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
        a = img[..., 3:4].astype(np.float32) / 255.0
        self.tex = np.concatenate([img[..., :3].astype(np.float32) * a, a], axis=2)  # premultiplied
        self.size = np.array([self.tex.shape[1], self.tex.shape[0]], float)
        self.anchor = np.array(info["anchor"], float) * scale
        self.parent_name = info["parent"]
        # world_anchor: this part's anchor point in COMP space (unambiguous --
        # doesn't depend on guessing how AE's position-minus-parent-anchor
        # math chains through parenting, which breaks if a layer was
        # positioned *before* being parented, a very common workflow).
        self.world_anchor = np.array(info["world_anchor"], float) * scale
        self.local_offset = None  # set in Puppy.__init__, relative to parent's world_anchor
        self.role, self.angle_key = classify(name)
        self.tex_flipped = cv2.flip(self.tex, 1)  # precomputed once, for facing left


def warp(frame, part, pivot, angle_deg, extra_scale=1.0, flip=False):
    """Rotate+scale `part` about its anchor and put the anchor at `pivot`.
    flip=True mirrors the part left-right (for facing the other direction)."""
    th = math.radians(angle_deg)
    c, s = math.cos(th) * extra_scale, math.sin(th) * extra_scale
    ax, ay = part.anchor
    tex = part.tex_flipped if flip else part.tex
    w, h = part.size
    if flip:
        ax = w - ax  # mirrored texture's anchor is reflected too
    tx, ty = pivot[0] - (c * ax - s * ay), pivot[1] - (s * ax + c * ay)
    xs = [tx, c * w + tx, c * w - s * h + tx, -s * h + tx]
    ys = [ty, s * w + ty, s * w + c * h + ty, c * h + ty]
    H, W = frame.shape[:2]
    x0, y0 = max(int(math.floor(min(xs))), 0), max(int(math.floor(min(ys))), 0)
    x1, y1 = min(int(math.ceil(max(xs))), W), min(int(math.ceil(max(ys))), H)
    if x1 <= x0 or y1 <= y0:
        return
    M = np.array([[c, -s, tx - x0], [s, c, ty - y0]], np.float32)
    warped = cv2.warpAffine(tex, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    out = warped[..., :3] + roi * (1.0 - warped[..., 3:4])
    frame[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)


# Paint order, back to front (cosmetic only; the skeleton math doesn't care).
Z_ORDER = ["rr", "rr2", "rl", "rl2", "torso", "tail", "head",
           "brow_l", "brow_r", "eye", "pupil", "fr", "fr2", "fl", "fl2"]


class Puppy:
    def __init__(self, rig, scale):
        self.by_name = {n: Part(n, i, scale) for n, i in rig.items()}
        self.torso = next(p for p in self.by_name.values() if p.role == "torso")
        self.name_of = {part: name for name, part in self.by_name.items()}
        self.kids_of = {}
        for part in self.by_name.values():
            self.kids_of.setdefault(part.parent_name, []).append(part)

        # Compute each part's local_offset (joint position relative to its
        # parent's anchor) from WORLD anchor coordinates, not AE's raw
        # Position field -- sidesteps AE's parenting math entirely.
        for part in self.by_name.values():
            if part.parent_name is None:
                part.local_offset = np.zeros(2)
            else:
                parent = self.by_name[part.parent_name]
                part.local_offset = part.world_anchor - parent.world_anchor

        # rest-pose bounding box (screen-space, torso anchor at origin) so we
        # know where the floor is and how big the puppy's silhouette is.
        mins, maxs = [np.zeros(2)], [self.torso.size]

        def walk(name, base):
            for child in self.kids_of.get(name, []):
                p0 = base + child.local_offset - child.anchor
                mins.append(p0)
                maxs.append(p0 + child.size)
                walk(self.name_of[child], base + child.local_offset)

        walk(self.name_of[self.torso], np.zeros(2))
        self.rel_min = np.min(mins, axis=0) - self.torso.anchor
        self.rel_max = np.max(maxs, axis=0) - self.torso.anchor

    def origin(self, pos):
        return np.array([pos[0], pos[1] - self.rel_max[1]], float)

    def screen_box(self, pos):
        o = self.origin(pos)
        return (*(o + self.rel_min), *(o + self.rel_max))

    def _transforms(self, o, pose):
        """pivot + cumulative angle for every part, computed parent-first."""
        root_name = self.name_of[self.torso]
        t = {root_name: (o, pose["torso_rot"])}

        def walk(name):
            ppos, pang = t[name]
            c, s = math.cos(math.radians(pang)), math.sin(math.radians(pang))
            for child in self.kids_of.get(name, []):
                cname = self.name_of[child]
                d = child.local_offset
                pivot = ppos + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
                ang = pang + pose[child.angle_key]
                t[cname] = (pivot, ang)
                walk(cname)

        walk(root_name)
        return t

    def draw(self, frame, pos, pose, breath_scale=1.0, facing=1):
        # facing=1 is the art's native orientation; facing=-1 mirrors the
        # whole puppy left-right (screen-space reflection about its own
        # position) so it can visibly turn to walk the other way.
        o = self.origin(pos) + np.array([0.0, pose["dy"]])
        tf = self._transforms(o, pose)
        flip = facing < 0
        for role in Z_ORDER:
            part = next((p for p in self.by_name.values() if p.role == role), None)
            if part is None:
                continue
            pivot, ang = tf[self.name_of[part]]
            if flip:
                pivot = np.array([2 * o[0] - pivot[0], pivot[1]])
                ang = -ang
            scale = breath_scale if part is self.torso else 1.0
            warp(frame, part, pivot, ang, scale, flip=flip)


# ---------------- brain: mood + weighted random behaviour ----------------
class Brain:
    def __init__(self):
        self.arousal = 0.1
        self.bond = 0.0
        self.walking = False
        self.walk_phase = 0.0
        self.facing = 1  # 1 = art's native orientation, -1 = mirrored
        self.base = "sit"
        self.action = None
        self.cooldown = time.time() + 1.0
        self.was_petting = False
        self.pose = zero_pose()
        self.blink_clip = BlinkClip(BLINK_FILE)
        self.blink_t0 = None  # None = not currently blinking
        self.blink_next = time.time() + random.uniform(*BLINK_INTERVAL)

    def start(self, name, now):
        if name in ("sit", "stand"):
            self.base = name
            self.cooldown = now + random.uniform(2.0, 4.0)
            return
        dur = {"wag": random.uniform(2.0, 3.5), "tilt": random.uniform(1.4, 2.4), "hop": 0.7}[name]
        self.action = dict(name=name, t0=now, dur=dur, side=random.choice((-1, 1)))

    def weights(self):
        ar, sitting = self.arousal, self.base == "sit"
        # Petting raises arousal -> dog should stand up, alert, tail wagging.
        # Being left alone lets arousal decay -> dog settles back into a sit.
        return {
            "sit": (0.1 if sitting else 1.4) * max(0.0, 0.7 - ar) + (0.05 if sitting else 0.0),
            "stand": (1.6 if sitting else 0.1) * max(0.05, ar) if sitting else 0.1,
            "wag": 0.4 + 2.5 * ar,
            "tilt": 0.5 + 0.8 * ar,
            "hop": 0.0 if sitting else 3.0 * max(0.0, ar - 0.45),
        }

    def update(self, dt, now, pet, look, body=None, pos=None):
        ar = self.arousal
        ar = clamp(ar + pet * PET_GAIN * dt, 0.0, 1.0)
        ar -= ar * dt / AROUSAL_DECAY_S
        self.arousal = ar

        # bond: a slower, stickier affection score built up by petting --
        # needs sustained contact (not one quick touch) to cross the
        # threshold that lets the puppy start following you around.
        self.bond = clamp(self.bond + pet * BOND_GAIN * dt, 0.0, 1.0)
        self.bond -= self.bond * dt / BOND_DECAY_S

        # start/stop walking toward the person based on distance + bond
        if body is not None and pos is not None:
            dist = body[0] - pos[0]
            if not self.walking and self.bond > BOND_THRESHOLD and abs(dist) > WALK_TRIGGER_DIST:
                self.walking = True
                self.action = None
            elif self.walking and abs(dist) < WALK_STOP_DIST:
                self.walking = False
                self.base = "stand"
                self.cooldown = now + random.uniform(1.5, 3.0)

        petting = pet > 0.2
        if petting and not self.was_petting and (self.action is None or self.action["name"] == "tilt"):
            self.start("wag", now)
        self.was_petting = petting

        if self.walking:
            # move toward the person at a steady walking speed; the floor
            # line (y) stays constant, only x changes, so it never looks
            # like it's lunging up toward a raised hand.
            # body can briefly become None (you step out of frame / mediapipe
            # loses you) -- keep stepping in place rather than crashing or
            # freezing, so the legs don't just lock mid-stride.
            if body is not None:
                step = WALK_SPEED * dt
                dist = body[0] - pos[0]
                pos[0] += clamp(dist, -step, step)
                if abs(dist) > 2.0:  # avoid flickering direction near the target
                    self.facing = 1 if dist >= 0 else -1
            self.walk_phase += dt * WALK_CYCLE_HZ * 2 * math.pi
            target = dict(BASE_POSTURES["stand"])
        else:
            if self.action and now - self.action["t0"] >= self.action["dur"]:
                self.action = None
                self.cooldown = now + random.uniform(0.6, 2.2) * (1.5 - ar)
            if self.action is None and now >= self.cooldown:
                w = self.weights()
                self.start(random.choices(list(w), weights=list(w.values()))[0], now)
            target = dict(BASE_POSTURES[self.base])

        if look is not None:
            target["head_rot"] += 10.0 * look
        k = 1.0 - math.exp(-dt * 5.0)
        for key in POSE_KEYS:
            self.pose[key] += (target[key] - self.pose[key]) * k
        pose = dict(self.pose)

        if self.walking:
            # true 4-beat walk (not a trot): each leg is a quarter-cycle
            # behind the last -- rear-left, front-left, rear-right, front-right
            # -- matching a real dog's walk gait rather than diagonal pairs.
            ph = self.walk_phase
            for leg, leg2, off in (("rl", "rl2", 0.0), ("fl", "fl2", 0.5 * math.pi),
                                    ("rr", "rr2", math.pi), ("fr", "fr2", 1.5 * math.pi)):
                pose[leg] += WALK_LEG_AMP * math.sin(ph + off)
                pose[leg2] += 12 + WALK_LEG_AMP * 0.7 * max(0.0, math.sin(ph + off + 0.6))
            # pelvis bobs down on each hind-leg contact -- two bounces per
            # full stride cycle, same as the reference sheet.
            pose["dy"] -= 6 * abs(math.sin(ph))
            pose["torso_rot"] += 3 * math.sin(ph)
        else:
            a = self.action
            if a:
                u = clamp((now - a["t0"]) / a["dur"], 0.0, 1.0)
                e = clamp(min(1.0, 5 * u, 5 * (1 - u)), 0.0, 1.0)
                e = e * e * (3 - 2 * e)
                if a["name"] == "wag":
                    pose["tail_rot"] += e * (18 + 14 * ar) * math.sin(2 * math.pi * (3 + 3 * ar) * (now - a["t0"]))
                elif a["name"] == "tilt":
                    pose["head_rot"] += e * 14 * a["side"]
                elif a["name"] == "hop":
                    arc = 4 * u * (1 - u)
                    pose["dy"] -= 150 * arc
                    pose["torso_rot"] -= 4 * arc
                    for leg in ("fl", "fr"):
                        pose[leg] -= 15 * arc
                    for leg in ("rl", "rr"):
                        pose[leg] += 15 * arc

        # idle tail sway only -- breathing (torso scale) is supplied by AE, not here.
        pose["tail_rot"] += (3 + 5 * ar) * math.sin(2 * math.pi * 0.5 * now)

        # blink: independent of sit/stand/walk -- just plays the authored
        # curve once every random interval, then goes idle again.
        if self.blink_t0 is None and now >= self.blink_next:
            self.blink_t0 = now
        if self.blink_t0 is not None:
            bt = now - self.blink_t0
            if bt >= self.blink_clip.duration:
                self.blink_t0 = None
                self.blink_next = now + random.uniform(*BLINK_INTERVAL)
            else:
                eye_off, brow_l_off, brow_r_off = self.blink_clip.sample(bt)
                pose["eye_rot"] += eye_off
                pose["brow_l_rot"] += brow_l_off
                pose["brow_r_rot"] += brow_r_off

        # eyes follow the tracked body left/right -- a small pupil rotation
        # standing in for a sideways slide (this rig's rotation-only, so the
        # pupil's anchor does the "sliding" by pivoting it within the eye).
        if body is not None and pos is not None:
            pose["pupil_rot"] += clamp((body[0] - pos[0]) * EYE_LOOK_GAIN,
                                        -EYE_LOOK_MAX, EYE_LOOK_MAX)

        return pose


# ---------------- tracking + interaction ----------------
class Tracker:
    def __init__(self, use_mouse):
        self.mouse = None
        self.use_mouse = use_mouse
        if not use_mouse and (mp is None or not hasattr(mp, "solutions")):
            print("[info] mediapipe 'solutions' not available -> mouse mode. "
                  "Try: pip uninstall mediapipe -y && pip install mediapipe==0.10.14")
            self.use_mouse = True
        if not self.use_mouse:
            self.hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=0,
                                                  min_detection_confidence=0.6,
                                                  min_tracking_confidence=0.5)
            self.pose = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)

    def process(self, frame):
        if self.use_mouse:
            return ([self.mouse] if self.mouse else []), [], None
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hr = self.hands.process(rgb)
        hands = [(lm.landmark[9].x * w, lm.landmark[9].y * h) for lm in (hr.multi_hand_landmarks or [])]
        legs = []
        body = None
        pr = self.pose.process(rgb)
        if pr.pose_landmarks:
            lm_list = pr.pose_landmarks.landmark
            for i in (25, 26, 27, 28, 31, 32):
                lm = lm_list[i]
                if lm.visibility > 0.5:
                    legs.append((lm.x * w, lm.y * h))
            # hip midpoint (23, 24): a stable horizontal body position, used
            # to steer the puppy's walk-to-you behaviour -- kept separate
            # from hand tracking so walking doesn't jitter toward your hand.
            hips = [lm_list[i] for i in (23, 24) if lm_list[i].visibility > 0.5]
            if hips:
                body = (sum(lm.x for lm in hips) / len(hips) * w,
                        sum(lm.y for lm in hips) / len(hips) * h)
        return hands, legs, body


class Interaction:
    def __init__(self):
        self.prev = {}
        self.speed = {}

    def update(self, hands, legs, box, dt):
        x0, y0, x1, y1 = box
        padx, pady = (x1 - x0) * 0.08, (y1 - y0) * 0.08

        def inside(p):
            return x0 - padx <= p[0] <= x1 + padx and y0 - pady <= p[1] <= y1 + pady

        pet, prev, speed = 0.0, {}, {}
        for i, p in enumerate(hands):
            v = 0.0
            if i in self.prev:
                v = math.dist(p, self.prev[i]) / max(dt, 1e-3)
                v = 0.7 * self.speed.get(i, v) + 0.3 * v
            prev[i], speed[i] = p, v
            if inside(p):
                pet = max(pet, 0.25 + 0.75 * min(v / 500.0, 1.0))
        self.prev, self.speed = prev, speed

        if any(inside(p) for p in legs):
            pet = max(pet, 0.1)

        cx, cy, bw = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0
        near = [p for p in hands + legs if math.dist(p, (cx, cy)) < 1.6 * bw]
        look = None
        if near:
            p = min(near, key=lambda q: math.dist(q, (cx, cy)))
            look = clamp((p[0] - cx) / bw, -1.0, 1.0)
        return pet, look


# ---------------- main loop ----------------
def draw_debug(frame, puppy, pos, brain, hands, legs, lines):
    x0, y0, x1, y1 = map(int, puppy.screen_box(pos))
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 220, 255), 1)
    for p in hands:
        cv2.circle(frame, (int(p[0]), int(p[1])), 8, (0, 255, 0), 2)
    for p in legs:
        cv2.circle(frame, (int(p[0]), int(p[1])), 8, (255, 120, 0), 2)
    act = brain.action["name"] if brain.action else "-"
    cv2.putText(frame, f"{brain.base} | {act}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(frame, (12, 40), (212, 54), (255, 255, 255), 1)
    cv2.rectangle(frame, (12, 40), (12 + int(200 * brain.arousal), 54), (0, 200, 255), -1)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (12, 78 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", action="store_true", help="use the mouse as your hand")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    with open(RIG_FILE) as f:
        rig = json.load(f)

    if isinstance(rig, dict) and isinstance(rig.get("layers"), list):
        # Newer export shape: {"comp": {...}, "layers": [{"name": ..., "width": ..., "height": ...,
        # "worldAnchor": [...], ...}, ...]}. world_anchor is the unambiguous,
        # already-resolved comp-space anchor position AE computed for us --
        # we use that instead of trying to re-derive it from "position" +
        # parenting math, which breaks for layers positioned before parenting.
        rig = {
            layer["name"]: {
                "size": [layer["width"], layer["height"]],
                "anchor": layer["anchor"],
                "world_anchor": layer["worldAnchor"],
                "parent": layer["parent"],
            }
            for layer in rig["layers"]
            if not layer["name"].lower().startswith("layer ")
        }
    else:
        # Older flat-dict rig shape (no worldAnchor) -- fall back to walking
        # the parent chain ourselves to resolve world-space anchors.
        rig = {k: v for k, v in rig.items() if not k.lower().startswith("layer ")}
        world = {}

        def resolve(name):
            if name in world:
                return world[name]
            info = rig[name]
            local = np.array(info["position"], float)
            if info["parent"] is None:
                world[name] = local
            else:
                world[name] = resolve(info["parent"]) + local - np.array(rig[info["parent"]]["anchor"], float)
            return world[name]

        for n in rig:
            rig[n]["world_anchor"] = resolve(n).tolist()

    puppy = Puppy(rig, PUPPY_SCALE)
    breath = Breathing(BREATH_FILE)
    brain, inter = Brain(), Interaction()
    tracker = Tracker(args.mouse)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state = {"pos": None}

    def on_mouse(event, x, y, flags, _):
        if tracker.use_mouse:
            tracker.mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            state["pos"] = [x, y]

    win = "Webcam Puppy"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    debug, last, mirror = True, time.time(), MIRROR
    t0 = last

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        if state["pos"] is None:
            state["pos"] = [W * 0.5, H * FLOOR_Y]
        pos = state["pos"]

        now = time.time()
        dt = min(now - last, 0.1)
        last = now

        hands, legs, body = tracker.process(frame)
        pet, look = inter.update(hands, legs, puppy.screen_box(pos), dt)
        pose = brain.update(dt, now, pet, look, body, pos)
        puppy.draw(frame, pos, pose, breath.scale_at(now - t0), brain.facing)
        if debug:
            sit = BASE_POSTURES["sit"]
            lines = [f"tracking: {'MOUSE' if tracker.use_mouse else 'mediapipe'}  "
                     f"hands:{len(hands)}  legs:{len(legs)}  mirror:{mirror}",
                     f"sit fold: rl/rr={sit['rl']:.0f}/{sit['rl2']:.0f}  drop={sit['dy']:.0f}  "
                     f"(m mirror, s sit/stand, [ ] fold, , . drop)",
                     f"bond:{brain.bond:.2f}  walking:{brain.walking}  "
                     f"body:{'seen' if body is not None else 'none'}"]
            draw_debug(frame, puppy, pos, brain, hands, legs, lines)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("d"):
            debug = not debug
        elif key == ord("m"):
            mirror = not mirror
        elif key == ord("s"):
            brain.base = "stand" if brain.base == "sit" else "sit"
            brain.cooldown = now + 10.0
        elif key in (ord("["), ord("]")):
            d = 5 if key == ord("]") else -5
            BASE_POSTURES["sit"]["rl"] -= d
            BASE_POSTURES["sit"]["rr"] -= d
            BASE_POSTURES["sit"]["rl2"] += d
            BASE_POSTURES["sit"]["rr2"] += d
        elif key in (ord(","), ord(".")):
            BASE_POSTURES["sit"]["dy"] += 4 if key == ord(".") else -4

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()