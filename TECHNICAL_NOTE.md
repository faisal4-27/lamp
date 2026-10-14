# Lumi: technical note

Lumi is a desk-lamp character for the supplied five-joint lamp. Key decision: **one state machine decides everything the lamp does**, so light, motion, voice, sound and music act as one character.

## 1. Architecture and data flow

```
 camera -> Vision: face, attention, waves      mic -> Hearing: speech to text
              |  events                                   |  events
+-- BRAIN process ---------------------------------------------------------------------+
| Character state machine: the only place behaviour is decided                         |
| local reflexes: hi/bye, follow face, wave back, hum  |  audio mixer -> speaker       |
| decisions: Claude (fixed JSON) <- scene memory      |  skills -> aiming -> motion    |
+--------------------------------------------------------------------------------------+
        WebSocket JSON:  motion, light  (down)   |   joint state 30 Hz  (up)
+-- BODY process ----------------------------------------------------------------------+
| 200 Hz joint simulation, joint limits, safe timing  ->  3D viewer in browser         |
+--------------------------------------------------------------------------------------+
```

Senses report events, the character decides, and the body enforces safety. **Why two processes:** a real robot separates motor firmware from its main computer. If the brain crashes, the body rests safely within 3 s and the brain restarts.

## 2. Protocol

JSON messages over a local WebSocket (`lampbot/protocol.py`):

- **Brain → body:** timed poses, a face-following target, light colour/brightness, stop.
- **Body → brain:** joint positions, speeds and load 30 times a second, plus a report whenever a move was slowed for safety.

**Why:** the brain can only request movement, so no bug or AI output can push a joint past its limits.

## 3. Perception, action and memory

Face, attention, wave and speech detection run locally; Claude is used only for open-ended understanding and never outputs joint angles. It replies in a fixed JSON format: what to say, a mood, the objects it sees (with image positions), and actions from a fixed skill list (`look_at`, `shine_on`, `nod`, `dance`, `sing`, …). The code turns an object's image position into a 3D point on the desk and calculates the joint angles to point at it, preferring low-strain postures.

For goals ("shine on the mug") the lamp acts, then **looks again** with a fresh image. *The code, not the AI*, decides whether the aim is good; if the object moved, it re-aims, up to twice.

Scene memory (colour, position, "last seen") is saved to a file and sent with every question, so the lamp can answer about objects that are gone. Each cloud call sends one small image, the transcript, memory and recent dialogue, and only while someone is engaged. No audio or video stream is sent. If Claude fails, keyword reflexes answer instead.

## 4. Character integration

- **Engagement means attention, not just presence.** It engages after you face it for 0.6 s and lets go after 2 s of looking away. The delays stop blinks from causing flicker.
- **Looking at it** brings a perk-up, a chime, a **bright light** and "Hi there!". **Looking away** turns the light **dark** with a "Bye-bye!". After 5 s alone it hums, and a wave gets a head-wag back and a hi.
- **Sound.** All sound effects, music, the hum and songs are generated in code in one musical key. Music quietens under speech, and the light pulses with the voice.
- **Thinking cues** (amber light, head tilt) cover Claude's wait. The microphone listens only while the lamp is silent.

## 5. Simulation and physical reasoning

The body simulates the supplied robot model (URDF) at 200 Hz: joints stay in safe ranges, moves are slowed to at most 80% of rated speed, acceleration is limited (gentler on the heavy base and shoulder), and every move starts and stops smoothly. Gestures are sized to these limits.

## 6. Deployment (Ubuntu 24.04, 4 cores, 8 GB, no GPU)

`scripts/install_ubuntu.sh` installs the audio and camera packages, sets permissions, creates a Python environment and downloads all models once (about 250 MB). Every model runs on the CPU; only Claude needs the internet. Optional systemd services restart the lamp automatically. The body gets CPU priority, and the brain is capped at 2 GB of RAM and 3 cores.

## 7. Measurements

Four live sessions on a Windows laptop (2026-10-14); stage timings also restricted to 4 cores to mimic the target.

- **Response time** (from when you stop talking to the lamp's first sound): median **6.3 s** (5.2–8.0 s, 6 turns). Claude takes 5.7 s of that; local keyword replies take about 1 s.
- **Engagement reliability** (guided test, 33 prompts, each scored 6 s after the prompt): **precision 0.91, recall 1.00**. It noticed 10 of 10 looks and let go correctly in 12 of 13 look-away/leave trials, a median 3.0 s after the prompt (reaction time plus the designed 2 s delay). **Waves:** 4 of 5 detected, 0 false alarms in 5 hands-busy trials. In normal use: 15 engagements, one 2.2 s false start, no flicker.
- **CPU and memory:** the brain averages about 1.4 cores and 530 MB, the body under 0.1 core and 62 MB. On 4 cores, speech-to-text takes 1.2 s.
- **Lesson learned:** restricted to 4 cores, the voice engine ran 14× slower (threads sized for all 16 dev cores); a real 4-core laptop is unaffected.

## 8. Known limitations and scope

- **Claude replies take about 6 s;** the thinking cues mask but don't remove the wait.
- **Aiming is approximate.** In the one live goal attempt, the checks disagreed with the first estimate, and the lamp settled on "close enough".
- "Looking" uses head direction, not eye gaze; the wave rule could fire on fast gestures; you can't interrupt the lamp while it talks.
- Not yet run on the Ubuntu target. Left out on purpose (6–8 h timebox): hardware drivers, multi-person conversation, trained gesture models.

## 9. How this was built

Built with **Claude Code** (Anthropic's AI coding assistant) as a pair programmer: it wrote most of the code, ran the tests and benchmarks, and drafted this note from measured data. I set the goals, chose the character's behaviours, ran the live sessions and reviewed the results, and I am responsible for the design.
