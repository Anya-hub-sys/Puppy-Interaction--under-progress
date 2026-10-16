# Webcam Puppy

A cartoon puppy composited live onto your webcam feed. It reacts to you via
hand/body tracking (mediapipe) — gets excited when you pet it, stands, sits,
walks over to you, blinks, breathes, and looks toward you — all rendered from
layered PNG parts exported out of After Effects. No ML training, no cloud.

## 1. Requirements

- Python 3.9+
- A webcam
- Windows/Mac/Linux all fine

## 2. Project layout

```
.
├── puppy.py              # main program — rename to this if yours is puppy (3).py etc.
├── puppy_rig.json         # part anchors + AE parenting, exported from AE
├── puppy_breathing.json   # torso breathing keyframes (optional)
├── puppy_blink.json       # eye/eyebrow blink keyframes (optional)
├── export_rig.jsx         # AE script: (re-)generate puppy_rig.json
├── export_breathing.jsx   # AE script: (re-)generate puppy_breathing.json
├── export_animation.jsx   # AE script: export any other keyframed clip (sit, etc.)
├── assets/                # one PNG per rig part (transparent background)
└── venv/                  # your local virtual environment (don't commit this)
```

> **Note:** rename whichever copy of the puppy script you're keeping (e.g.
> `puppy (3).py`) to plain `puppy.py` before running the commands below — or
> just swap `puppy.py` for your actual filename everywhere it appears.

`.idea` and `venv` are IDE/environment folders — add them to `.gitignore`
rather than committing them (see step 6).

## 3. Clone and set up a virtual environment

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install opencv-python numpy mediapipe
```

Or, if you add a `requirements.txt` to the repo (recommended):

```
opencv-python
numpy
mediapipe
```

then just:

```bash
pip install -r requirements.txt
```

mediapipe is optional — without it, the program still runs in `--mouse` mode
(see below), just without real hand/body tracking.

## 5. Run it

```bash
python puppy.py
```

Other useful flags:

```bash
python puppy.py --mouse        # use your mouse as your hand, no webcam tracking needed
python puppy.py --camera 1     # use a different webcam index (default is 0)
```

Keyboard controls while it's running:

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `d` | Toggle debug overlay |
| `m` | Toggle mirror |
| Click | Move the puppy |
| `s` | Force sit/stand |
| `[` / `]` | Adjust rear-leg fold amount |
| `,` / `.` | Adjust sit height |

## 6. Suggested `.gitignore`

```
venv/
.idea/
__pycache__/
*.pyc
```

## 7. Re-exporting rig/animation data from After Effects

If you change the rig (add/move/rename layers) or re-author an animation:

- **Rig/anchors** → open the puppy comp in AE, run `export_rig.jsx`, save as
  `puppy_rig.json` (overwrite the old one).
- **Breathing** → select the torso layer, set the work area to one breath
  cycle, run `export_breathing.jsx`, save as `puppy_breathing.json`.
- **Any other keyframed clip** (blink, sit, etc.) → set the work area to one
  play-through of the clip, run `export_animation.jsx`, save with a name
  that matches what `puppy.py` expects for that clip.

Drop the new JSON into the project root (same folder as `puppy.py`) and
re-run — no code changes needed unless you're adding brand-new rig parts.

## 8. Troubleshooting

- **Blank/black window** — check `--camera` index; try `0`, `1`, `2`.
- **"No PNG in 'assets/' matching part 'X'"** — a layer name in
  `puppy_rig.json` doesn't match any filename in `assets/`; rename the PNG
  or the AE layer so they match (case/spacing-insensitive, but the words
  must match).
- **Puppy doesn't blink/breathe but otherwise runs fine** — those two files
  are optional; the program just logs `[info] no puppy_blink.json; ...` and
  carries on without them.
