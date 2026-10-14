// Lamp viewer: renders the URDF with three.js and mirrors body state + brain HUD.
// It is display-only; the body process owns the simulation.
import * as THREE from "three";
import { OrbitControls } from "/web/vendor/OrbitControls.js";
import { STLLoader } from "/web/vendor/STLLoader.js";

const JOINT_SHORT = {
  base_yaw_joint: "base", shoulder_pitch_joint: "shoulder", elbow_pitch_joint: "elbow",
  neck_yaw_joint: "neck", head_pitch_joint: "head",
};

// ---------------------------------------------------------------- scene
const host = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x15171c);
scene.fog = new THREE.Fog(0x15171c, 4, 10);

const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50);
camera.position.set(-1.45, 0.9, 1.15);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(-0.1, 0.32, 0);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x3a332c, 1.6));
const key = new THREE.DirectionalLight(0xffffff, 1.4);
key.position.set(-1.5, 2.5, 1.5);
key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
scene.add(key);

const desk = new THREE.Mesh(
  new THREE.CircleGeometry(2.2, 64),
  new THREE.MeshStandardMaterial({ color: 0x3a3f4a, roughness: 0.85 })
);
desk.rotation.x = -Math.PI / 2;
desk.receiveShadow = true;
scene.add(desk);
const grid = new THREE.GridHelper(4, 40, 0x555b66, 0x2c3038);
grid.position.y = 0.001;
scene.add(grid);

// URDF is z-up; three.js is y-up.
const robotRoot = new THREE.Group();
robotRoot.rotation.x = -Math.PI / 2;
scene.add(robotRoot);

function onResize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener("resize", onResize);
onResize();

// ---------------------------------------------------------------- URDF
const vec = (s, d = [0, 0, 0]) => (s ? s.trim().split(/\s+/).map(Number) : d);
function originOf(el) {
  const g = new THREE.Group();
  const o = el && el.querySelector(":scope > origin");
  const [x, y, z] = vec(o && o.getAttribute("xyz"));
  const [r, p, yw] = vec(o && o.getAttribute("rpy"));
  g.position.set(x, y, z);
  g.rotation.set(r, p, yw, "ZYX"); // URDF fixed-axis RPY == intrinsic ZYX
  return g;
}

const materials = {};
const joints = {}; // name -> {group, axis}
let lightDisc = null, spot = null, emitterBulb = null;

async function loadRobot() {
  const text = await (await fetch("/robot/dummy_lamp_5dof.urdf")).text();
  const xml = new DOMParser().parseFromString(text, "application/xml");
  for (const m of xml.querySelectorAll("robot > material")) {
    const c = vec(m.querySelector("color").getAttribute("rgba"));
    materials[m.getAttribute("name")] = c;
  }
  const linkGroups = {};
  const stl = new STLLoader();
  for (const link of xml.querySelectorAll("robot > link")) {
    const name = link.getAttribute("name");
    const lg = new THREE.Group();
    lg.name = name;
    linkGroups[name] = lg;
    for (const vis of link.querySelectorAll(":scope > visual")) {
      const og = originOf(vis);
      const geo = vis.querySelector("geometry > *");
      const matName = vis.querySelector("material")?.getAttribute("name");
      const rgba = materials[matName] || [0.8, 0.8, 0.8, 1];
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
        roughness: matName === "fixture_chrome" ? 0.3 : 0.5,
        metalness: matName === "fixture_chrome" ? 0.35 : 0.0,
        transparent: rgba[3] < 1, opacity: rgba[3], side: THREE.DoubleSide,
      });
      let mesh;
      if (geo.tagName === "cylinder") {
        const g = new THREE.CylinderGeometry(+geo.getAttribute("radius"), +geo.getAttribute("radius"),
          +geo.getAttribute("length"), 40);
        g.rotateX(Math.PI / 2); // three cylinders are y-aligned, URDF z-aligned
        mesh = new THREE.Mesh(g, mat);
      } else if (geo.tagName === "sphere") {
        mesh = new THREE.Mesh(new THREE.SphereGeometry(+geo.getAttribute("radius"), 32, 16), mat);
      } else if (geo.tagName === "mesh") {
        const g = await new Promise((res, rej) =>
          stl.load("/robot/" + geo.getAttribute("filename"), res, undefined, rej));
        g.computeVertexNormals();
        mesh = new THREE.Mesh(g, mat);
      } else continue;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      og.add(mesh);
      lg.add(og);
      if (matName === "fixture_light" && name === "lamp_head_link") {
        mat.emissive = new THREE.Color(1, 0.95, 0.76);
        lightDisc = mat;
      }
      if (name === "light_emitter_link") {
        mat.emissive = new THREE.Color(1, 0.95, 0.76);
        emitterBulb = mat;
      }
    }
  }
  let rootLink = null;
  const childLinks = new Set();
  for (const j of xml.querySelectorAll("robot > joint")) childLinks.add(j.querySelector("child").getAttribute("link"));
  for (const n in linkGroups) if (!childLinks.has(n)) rootLink = n;
  robotRoot.add(linkGroups[rootLink]);

  for (const j of xml.querySelectorAll("robot > joint")) {
    const parent = linkGroups[j.querySelector("parent").getAttribute("link")];
    const child = linkGroups[j.querySelector("child").getAttribute("link")];
    const og = originOf(j);
    const jg = new THREE.Group();
    og.add(jg);
    jg.add(child);
    parent.add(og);
    const axis = new THREE.Vector3(...vec(j.querySelector("axis")?.getAttribute("xyz"), [1, 0, 0])).normalize();
    joints[j.getAttribute("name")] = { group: jg, axis };
  }

  // A real spotlight from the emitter so the beam visibly lands on the desk.
  spot = new THREE.SpotLight(0xfff2c2, 0, 4, 0.5, 0.45, 1.2);
  spot.castShadow = true;
  spot.shadow.mapSize.set(1024, 1024);
  const tgt = new THREE.Object3D();
  tgt.position.set(1, 0, 0);
  linkGroups["light_emitter_link"].add(spot, tgt);
  spot.target = tgt;
}

// ---------------------------------------------------------------- markers
const camMarker = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.06, 0.02),
  new THREE.MeshStandardMaterial({ color: 0x111111 }));
const camLens = new THREE.Mesh(new THREE.SphereGeometry(0.008, 12, 8),
  new THREE.MeshStandardMaterial({ color: 0x4aa3ff, emissive: 0x2266aa }));
camLens.position.set(-0.012, 0, 0);
camMarker.add(camLens);
camMarker.visible = false;
robotRoot.add(camMarker);

const aimMarker = new THREE.Mesh(new THREE.SphereGeometry(0.025, 20, 12),
  new THREE.MeshBasicMaterial({ color: 0xffd27a, transparent: true, opacity: 0.85 }));
aimMarker.visible = false;
robotRoot.add(aimMarker);
const aimLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineDashedMaterial({ color: 0xffd27a, dashSize: 0.02, gapSize: 0.015 }));
aimLine.visible = false;
robotRoot.add(aimLine);

// ---------------------------------------------------------------- HUD
const $ = (id) => document.getElementById(id);
const jointRows = {};
function ensureJointRows(names) {
  const box = $("joints");
  for (const n of names) {
    if (jointRows[n]) continue;
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span>${JOINT_SHORT[n] || n}</span><div class="bar"><div></div></div><span class="num"></span>`;
    box.appendChild(row);
    jointRows[n] = { bar: row.querySelector(".bar > div"), num: row.querySelector(".num") };
  }
}
function addDialog(role, text) {
  const d = document.createElement("div");
  d.className = "msg " + role;
  d.textContent = text;
  $("dialog").appendChild(d);
  $("dialog").scrollTop = 1e9;
  while ($("dialog").children.length > 60) $("dialog").firstChild.remove();
}

function applyState(s) {
  for (const [name, q] of Object.entries(s.q)) {
    const j = joints[name];
    if (j) j.group.quaternion.setFromAxisAngle(j.axis, q);
  }
  ensureJointRows(Object.keys(s.q));
  for (const [name, load] of Object.entries(s.load || {})) {
    const r = jointRows[name];
    if (!r) continue;
    r.bar.style.width = Math.min(100, load * 100) + "%";
    r.bar.style.background = load > 0.8 ? "var(--bad)" : "var(--accent)";
    r.num.textContent = (s.q[name] ?? 0).toFixed(2);
  }
  const L = s.light;
  if (L && spot) {
    const c = new THREE.Color(L.rgb[0], L.rgb[1], L.rgb[2]);
    spot.color.copy(c);
    spot.intensity = 12 * L.brightness;
    for (const m of [lightDisc, emitterBulb]) {
      if (!m) continue;
      m.emissive.copy(c);
      m.emissiveIntensity = 0.15 + 2.2 * L.brightness;
    }
  }
}

function applyHud(m) {
  switch (m.topic) {
    case "state":
      $("state").textContent = m.state;
      $("state").style.color = m.color || "var(--text)";
      break;
    case "attention":
      $("attn").style.width = Math.round(100 * (m.score ?? 0)) + "%";
      $("attn").style.background = m.engaged ? "var(--good)" : "var(--accent)";
      break;
    case "camera":
      $("cam").src = "data:image/jpeg;base64," + m.jpeg;
      $("cam").style.visibility = "visible";
      break;
    case "dialog":
      addDialog(m.role, m.text);
      break;
    case "memory": {
      const box = $("memory");
      box.innerHTML = "";
      if (!m.objects.length) box.innerHTML = '<div class="mem"><span>nothing remembered yet</span></div>';
      for (const o of m.objects) {
        const d = document.createElement("div");
        d.className = "mem";
        d.innerHTML = `<b></b> <span></span>`;
        d.querySelector("b").textContent = o.name;
        d.querySelector("span").textContent = `${o.summary}`;
        box.appendChild(d);
      }
      break;
    }
    case "layout": {
      const [x, y, z] = m.camera_pos;
      camMarker.position.set(x, y, z);
      camMarker.visible = true;
      break;
    }
    case "aim":
      if (m.point) {
        aimMarker.position.set(...m.point);
        aimMarker.visible = true;
        aimLine.geometry.setFromPoints([new THREE.Vector3(...m.from), new THREE.Vector3(...m.point)]);
        aimLine.computeLineDistances();
        aimLine.visible = true;
      } else {
        aimMarker.visible = aimLine.visible = false;
      }
      break;
    case "metrics":
      $("metrics").textContent = m.text;
      break;
  }
}

// ---------------------------------------------------------------- socket
let sock = null;
document.getElementById("say").addEventListener("submit", (e) => {
  e.preventDefault();
  const box = document.getElementById("sayText");
  const text = box.value.trim();
  if (text && sock && sock.readyState === 1) sock.send(JSON.stringify({ type: "operator", text }));
  box.value = "";
});

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws?role=viewer`);
  sock = ws;
  ws.onopen = () => ($("conn").textContent = "body connected");
  ws.onclose = () => {
    $("conn").textContent = "body disconnected — retrying…";
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "state") applyState(m);
    else if (m.type === "hud") applyHud(m);
    else if (m.type === "motion_done" && m.clamped?.length)
      addDialog("sys", `motion ${m.id}: clamped ${m.clamped.map((n) => JOINT_SHORT[n]).join(", ")}`);
  };
}

loadRobot().then(connect);
renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});
