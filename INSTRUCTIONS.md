# How to run Lumi, the lamp character

Everything lives in this `lamp/` folder. There are two processes:

| Process | Command | What it does |
|---|---|---|
| **body** | `python -m lampbot.body` | Simulated lamp (servos, light, safety limits) + 3D viewer at http://127.0.0.1:8765 |
| **brain** | `python -m lampbot.brain` | Camera, microphone, speaker, the character state machine, and the Claude "mind" |

`python -m lampbot` starts both, restarts the brain if it crashes, and opens the viewer.

---

## 1. First-time setup

### Windows (development machine)

```powershell
cd lamp
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\fetch_assets.py     # three.js, face model, voice, whisper (~250 MB, once)
copy .env.example .env                            # then put your key in .env (see step 2)
```

### Ubuntu 24.04 (target laptop)

```bash
cd lamp
./scripts/install_ubuntu.sh             # apt packages, venv, models
./scripts/install_ubuntu.sh --service   # optional: also install + start systemd user services
```

Log out and back in once afterwards so the `video`/`audio` group membership applies.

## 2. API key (for conversation, scene memory and goals)

Put your Anthropic key in `lamp/.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Without a key the lamp still works in **local-reflex mode**. It notices you, tracks your face, greets you,
listens and transcribes. It also responds to simple keywords ("sing", "dance", "play music", "turn red", "sleep", "stop").
It cannot describe the scene or plan goals. The log says `running with local reflexes only` when this happens.

## 3. Run it

```powershell
.venv\Scripts\python -m lampbot          # Windows
```
```bash
.venv/bin/python -m lampbot             # Ubuntu
```

The viewer opens in your browser. Sit in front of the laptop camera.

## 4. The demo script (one continuous interaction)

1. **Engagement.** Walk into view and the lamp turns towards you (state `NOTICED`). Look at the camera for
   about half a second and its light comes on bright as it perks up and says *"Hi there!"* (`GREETING`). Look away for 2 s and it says
   *"Bye-bye!"* or *"See you later!"*, its light goes dark, and it lets go (`DISENGAGING`). Keep not looking for 5 s and it starts
   humming to itself; look back and the hum stops with *"Hi there! Welcome back."*
2. **Character response.** On greeting you get a two-note chime, a light flash, a perk-up motion and a spoken hello.
   **Wave** at it (side to side, beside or above your head) and it wags its head back at you and says
   *"Hi there!"*, even if you were not looking right at it.
3. **Spoken interaction.** In `LISTENING` (soft blue pulsing light) ask anything. The light turns amber while
   it thinks, then pulses with its voice while it talks.
4. **Scene memory.** Hold an object up to the camera, or put one on the desk, and say *"what do you think of
   this?"*. It appears under *Scene memory* in the viewer. Put it away, chat about something else, then ask
   *"what colour was the thing I showed you?"*.
5. **Goal-directed action.** Say *"shine your light on the mug"*. The lamp says what it will do, aims at the mug
   (a yellow dot in the viewer shows the 3D target), then takes a fresh camera frame to check. If you moved the
   mug, it says so and re-aims.
6. **Music and SFX.** *"Can you dance?"* plays a bouncy loop while it dances. *"Play something chill"* and
   *"stop the music"* also work. *"Sing me a song"* gets a short original song in its own voice, sung over a
   soft backing while it sways. Leave it alone for a minute and it dozes off humming.

**No microphone?** Type into the box under *Conversation* in the viewer. Typed text goes through exactly the same
pipeline as speech, minus speech-to-text.

## 5. Useful commands

| Command | Purpose |
|---|---|
| `python -m lampbot --no-browser` | Don't open the viewer automatically |
| `python -m lampbot --body-only` | Just the simulated lamp (no camera or mic) |
| `python scripts/dry_run.py` (with the body running) | Run the goal loop (aim → verify → correct) with a scripted mind, no API needed |
| `python scripts/engagement_trial.py --trials 20 --waves 10` | Guided look/look-away (and wave/no-wave) trials; measures engagement precision/recall and wave detection |
| `python scripts/report.py [runs/<dir>]` | Latency p50/p95, engagement stats, CPU and memory for a run (default: latest) |
| `python -m sounddevice` | List audio devices (for `LAMP_MIC_DEVICE` / `LAMP_SPEAKER_DEVICE`) |

Every run writes `runs/<timestamp>/events.jsonl` and `brain.log`. Scene memory persists in `data/memory.json`;
delete that file to make the lamp forget.

## 6. Settings

Any field in `lampbot/config.py` can be set in `.env` as `LAMP_<NAME>`. The common ones are listed in `.env.example`:

- `LAMP_MODEL` (default `claude-opus-5`) and `LAMP_EFFORT` (default `low`, chosen for latency).
- `LAMP_CAMERA_INDEX`, `LAMP_MIC_DEVICE`, `LAMP_SPEAKER_DEVICE`.
- `LAMP_MAX_YAW_RATIO`: how far you can turn your head and still count as "looking". Raise it if it disengages
  too eagerly; lower it if it engages when you look past it.
- `LAMP_CAMERA_POS`: where the laptop camera sits relative to the lamp base. This affects where it aims.
- `LAMP_WHISPER_MODEL`: `tiny.en` is faster, `small.en` is more accurate.
- `LAMP_HUM_AFTER_S` (default 5): how long nobody looks before it starts humming.
- `LAMP_WAVE_REVERSALS` (default 3): side-to-side swings needed to count as a wave. Raise it if it sees waves
  when you talk with your hands; lower it if it misses your waves.
- `LAMP_OFFLINE=true` never calls the cloud.

## 7. Troubleshooting

- **Camera won't open:** close other apps using it, or try `LAMP_CAMERA_INDEX=1`. On Linux check `v4l2-ctl --list-devices`.
- **It never hears you:** it only listens while engaged (looking at it) and while its own audio is silent.
  Watch the state readout: it must say `LISTENING`. Check `LAMP_MIC_DEVICE`.
- **It hears itself / music:** the mic is gated while it speaks. Lower `LAMP_MUSIC_VOLUME` if the music trips it.
- **Viewer is blank:** run `scripts/fetch_assets.py` (it vendors three.js) and reload.
- **Port 8765 busy:** set `LAMP_BODY_PORT`.
