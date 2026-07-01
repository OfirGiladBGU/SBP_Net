"use strict";
/*
 * SBP-Net interactive demo -- WebGL2 frontend.
 *
 * Phase 3: point-cloud renderer + orbit camera, draws GET /volume.
 * Phase 4: GPU color-picking (IDs -> offscreen framebuffer) + the live loop:
 *          pick -> lock + loader -> POST /reconstruct -> append returned voxels
 *          in highlight color -> re-render -> unlock.
 *
 * The backend owns the volume state; this frontend is a pure mirror -- it only
 * appends what the backend returns (constraint 2). One reconstruction at a time
 * (constraint 3): the loader is the concurrency lock.
 */

// --------------------------------------------------------------------------- //
// Minimal mat4 / vec3 helpers (column-major, no external libraries).          //
// --------------------------------------------------------------------------- //
const M4 = {
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return [f / aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0];
  },
  lookAt(eye, center, up) {
    const z = norm(sub(eye, center)); const x = norm(cross(up, z)); const y = cross(z, x);
    return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
            -dot(x,eye),-dot(y,eye),-dot(z,eye),1];
  },
  multiply(a, b) {
    const o = new Array(16);
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
      o[c*4+r] = a[0*4+r]*b[c*4+0] + a[1*4+r]*b[c*4+1] + a[2*4+r]*b[c*4+2] + a[3*4+r]*b[c*4+3];
    }
    return o;
  },
};
const sub = (a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const len=(a)=>Math.hypot(a[0],a[1],a[2]);
const norm=(a)=>{const l=len(a)||1;return [a[0]/l,a[1]/l,a[2]/l];};

// --------------------------------------------------------------------------- //
// Shaders (GLSL ES 3.00).                                                     //
// --------------------------------------------------------------------------- //
const VS_RENDER = `#version 300 es
layout(location=0) in vec3 a_position;
layout(location=1) in float a_type;      // 0 input, 1 reconstructed, 2 last-added
uniform mat4 u_proj, u_view;
uniform float u_pointSize;
uniform float u_showRecon;
out vec3 v_color;
out float v_discard;
void main() {
  gl_Position = u_proj * u_view * vec4(a_position, 1.0);
  gl_PointSize = u_pointSize;
  v_discard = (a_type > 0.5 && u_showRecon < 0.5) ? 1.0 : 0.0;
  if (a_type < 0.5)       v_color = vec3(0.50, 0.55, 0.63);   // input  (grey)
  else if (a_type < 1.5)  v_color = vec3(0.26, 0.82, 0.48);   // recon  (green)
  else                    v_color = vec3(1.00, 0.81, 0.30);   // last   (amber)
}`;
const FS_RENDER = `#version 300 es
precision highp float;
in vec3 v_color; in float v_discard;
out vec4 fragColor;
void main() {
  if (v_discard > 0.5) discard;
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;          // round points
  fragColor = vec4(v_color, 1.0);
}`;

const VS_PICK = `#version 300 es
layout(location=0) in vec3 a_position;
layout(location=1) in float a_type;
uniform mat4 u_proj, u_view;
uniform float u_pointSize;
uniform float u_showRecon;
flat out vec3 v_id;
out float v_discard;
void main() {
  gl_Position = u_proj * u_view * vec4(a_position, 1.0);
  gl_PointSize = u_pointSize + 3.0;       // slightly bigger => forgiving click target
  v_discard = (a_type > 0.5 && u_showRecon < 0.5) ? 1.0 : 0.0;
  int id = gl_VertexID + 1;               // 0 reserved for background
  v_id = vec3(float(id & 0xFF), float((id >> 8) & 0xFF), float((id >> 16) & 0xFF)) / 255.0;
}`;
const FS_PICK = `#version 300 es
precision highp float;
flat in vec3 v_id; in float v_discard;
out vec4 fragColor;
void main() {
  if (v_discard > 0.5) discard;
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  fragColor = vec4(v_id, 1.0);
}`;

// Instanced lit voxel cubes (Phase 5): one unit cube stamped at every voxel via
// drawArraysInstanced. Axis-aligned faces + a directional light + hemispheric
// ambient give per-face shading -> a clear sense of depth as the object orbits.
const VS_VOXEL = `#version 300 es
layout(location=0) in vec3 a_cubePos;    // unit-cube vertex (edge 1, centered)
layout(location=1) in vec3 a_normal;     // face normal
layout(location=2) in vec3 a_offset;     // per-instance voxel center (world)
layout(location=3) in float a_type;      // per-instance: 0 input, 1 recon, 2 last
uniform mat4 u_proj, u_view;
uniform float u_voxelSize;
uniform float u_showRecon;
out vec3 v_color; out vec3 v_normal; flat out float v_cull;
void main() {
  v_cull = (a_type > 0.5 && u_showRecon < 0.5) ? 1.0 : 0.0;
  gl_Position = u_proj * u_view * vec4(a_offset + a_cubePos * u_voxelSize, 1.0);
  v_normal = a_normal;
  if (a_type < 0.5)       v_color = vec3(0.55, 0.60, 0.68);   // input  (grey)
  else if (a_type < 1.5)  v_color = vec3(0.26, 0.82, 0.48);   // recon  (green)
  else                    v_color = vec3(1.00, 0.81, 0.30);   // last   (amber)
}`;
const FS_VOXEL = `#version 300 es
precision highp float;
in vec3 v_color; in vec3 v_normal; flat in float v_cull;
out vec4 fragColor;
void main() {
  if (v_cull > 0.5) discard;
  vec3 N = normalize(v_normal);
  vec3 L = normalize(vec3(0.45, 0.85, 0.35));
  float diff = max(dot(N, L), 0.0);
  float ambient = 0.42 + 0.18 * N.y;              // hemispheric: brighter from above
  vec3 col = v_color * (ambient + 0.70 * diff);
  fragColor = vec4(col, 1.0);
}`;

// A unit cube (edge length 1, centered on origin) with one outward normal per face.
function unitCubeGeometry() {
  const faces = [
    { n: [0, 0, 1],  c: [[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]] },     // +Z
    { n: [0, 0, -1], c: [[1,-1,-1],[-1,-1,-1],[-1,1,-1],[1,1,-1]] }, // -Z
    { n: [1, 0, 0],  c: [[1,-1,1],[1,-1,-1],[1,1,-1],[1,1,1]] },     // +X
    { n: [-1, 0, 0], c: [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1]] }, // -X
    { n: [0, 1, 0],  c: [[-1,1,1],[1,1,1],[1,1,-1],[-1,1,-1]] },     // +Y
    { n: [0, -1, 0], c: [[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]] }, // -Y
  ];
  const pos = [], nor = [], tri = [0, 1, 2, 0, 2, 3];
  for (const f of faces) for (const i of tri) {
    pos.push(f.c[i][0] * 0.5, f.c[i][1] * 0.5, f.c[i][2] * 0.5);
    nor.push(f.n[0], f.n[1], f.n[2]);
  }
  return { positions: new Float32Array(pos), normals: new Float32Array(nor) };
}

// --------------------------------------------------------------------------- //
// GL setup helpers.                                                           //
// --------------------------------------------------------------------------- //
function compile(gl, type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

// --------------------------------------------------------------------------- //
// The demo app.                                                               //
// --------------------------------------------------------------------------- //
class Demo {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { antialias: true, preserveDrawingBuffer: false });
    if (!gl) throw new Error("WebGL2 not available");
    this.gl = gl;
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0.043, 0.055, 0.078, 1.0);

    this.progRender = program(gl, VS_RENDER, FS_RENDER);
    this.progPick = program(gl, VS_PICK, FS_PICK);
    this.progVoxel = program(gl, VS_VOXEL, FS_VOXEL);

    // Geometry buffers (grown as reconstructions come back).
    this.positions = [];   // flat world xyz per point
    this.types = [];        // one per point
    this.voxels = [];       // flat voxel (i,j,k) per point -> sent to backend
    this.count = 0;
    this.shape = [1, 1, 1];
    this.center = [0, 0, 0];
    this.cubeSize = 32;
    this.voxelSize = 0.02;  // world edge length of one voxel cube (set in loadVolume)

    // Per-instance data: voxel center (posBuf) + type (typeBuf). Shared by the
    // points renderer (as vertices) and the voxel renderer (as instances).
    this.posBuf = gl.createBuffer();
    this.typeBuf = gl.createBuffer();

    // --- points VAO (attribs are per-vertex) ---
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.typeBuf);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    // --- voxel VAO (unit cube per vertex + posBuf/typeBuf per instance) ---
    const cube = unitCubeGeometry();
    this.cubePosBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cubePosBuf);
    gl.bufferData(gl.ARRAY_BUFFER, cube.positions, gl.STATIC_DRAW);
    this.cubeNorBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cubeNorBuf);
    gl.bufferData(gl.ARRAY_BUFFER, cube.normals, gl.STATIC_DRAW);

    this.cubeVao = gl.createVertexArray();
    gl.bindVertexArray(this.cubeVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cubePosBuf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cubeNorBuf);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(2, 1);   // one offset per instance
    gl.bindBuffer(gl.ARRAY_BUFFER, this.typeBuf);
    gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(3, 1);   // one type per instance
    gl.bindVertexArray(null);

    // Offscreen framebuffer for GPU color-picking.
    this.pickFbo = gl.createFramebuffer();
    this.pickTex = gl.createTexture();
    this.pickDepth = gl.createRenderbuffer();
    this._pickSize = [0, 0];

    // Orbit camera state.
    this.cam = { theta: 0.9, phi: 1.1, radius: 2.4, target: [0, 0, 0] };
    this.pointSize = 4.0;
    this.showRecon = true;
    this.renderMode = "voxels";   // "voxels" (lit cubes) or "points"
    this.busy = false;

    this._bindUI();
    this._bindPointer();
    this._resize();
    window.addEventListener("resize", () => this._resize());
    requestAnimationFrame(() => this._render());
  }

  // ----- data ------------------------------------------------------------- //
  _worldOf(i, j, k) {
    // Center the cloud and scale into a comfortable unit-ish box.
    const s = 1.0 / Math.max(this.shape[0], this.shape[1], this.shape[2]);
    return [(i - this.center[0]) * s, (j - this.center[1]) * s, (k - this.center[2]) * s];
  }

  _appendVoxels(flatCoords, type) {
    for (let n = 0; n < flatCoords.length; n += 3) {
      const i = flatCoords[n], j = flatCoords[n + 1], k = flatCoords[n + 2];
      const w = this._worldOf(i, j, k);
      this.positions.push(w[0], w[1], w[2]);
      this.voxels.push(i, j, k);
      this.types.push(type);
    }
    this.count = this.types.length;
  }

  _uploadBuffers() {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(this.positions), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.typeBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(this.types), gl.STATIC_DRAW);
  }

  async loadVolume() {
    const data = await (await fetch("/volume")).json();
    this.shape = data.shape;
    this.cubeSize = data.cube_size;
    this.center = [this.shape[0] / 2, this.shape[1] / 2, this.shape[2] / 2];
    // A voxel spans one index unit; _worldOf scales by 1/max(shape). 0.9 leaves a
    // hairline gap between neighbours so cube edges stay legible.
    this.voxelSize = (1.0 / Math.max(this.shape[0], this.shape[1], this.shape[2])) * 0.9;
    this.positions = []; this.types = []; this.voxels = [];
    this._appendVoxels(data.original, 0);
    this._appendVoxels(data.reconstructed, 1);
    this._uploadBuffers();
    this.fit();
    this._setMeta(data);
    this._setStatus(`loaded ${data.name} — ${this.count.toLocaleString()} voxels`);
  }

  // ----- camera ----------------------------------------------------------- //
  _eye() {
    const { theta, phi, radius, target } = this.cam;
    const sp = Math.sin(phi);
    return [ target[0] + radius * sp * Math.cos(theta),
             target[1] + radius * Math.cos(phi),
             target[2] + radius * sp * Math.sin(theta) ];
  }
  fit() { this.cam.target = [0, 0, 0]; this.cam.radius = 2.4; }

  // ----- rendering -------------------------------------------------------- //
  _matrices() {
    const gl = this.gl;
    const aspect = gl.drawingBufferWidth / Math.max(1, gl.drawingBufferHeight);
    const proj = M4.perspective(Math.PI / 4, aspect, 0.01, 100);
    const view = M4.lookAt(this._eye(), this.cam.target, [0, 1, 0]);
    return { proj, view };
  }

  _drawPoints(prog, dpr) {
    const gl = this.gl;
    const { proj, view } = this._matrices();
    gl.useProgram(prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, "u_proj"), false, proj);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, "u_view"), false, view);
    gl.uniform1f(gl.getUniformLocation(prog, "u_pointSize"), this.pointSize * dpr);
    gl.uniform1f(gl.getUniformLocation(prog, "u_showRecon"), this.showRecon ? 1.0 : 0.0);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.count);
    gl.bindVertexArray(null);
  }

  _drawVoxels() {
    const gl = this.gl;
    const { proj, view } = this._matrices();
    const p = this.progVoxel;
    gl.useProgram(p);
    gl.uniformMatrix4fv(gl.getUniformLocation(p, "u_proj"), false, proj);
    gl.uniformMatrix4fv(gl.getUniformLocation(p, "u_view"), false, view);
    gl.uniform1f(gl.getUniformLocation(p, "u_voxelSize"), this.voxelSize);
    gl.uniform1f(gl.getUniformLocation(p, "u_showRecon"), this.showRecon ? 1.0 : 0.0);
    gl.bindVertexArray(this.cubeVao);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 36, this.count);
    gl.bindVertexArray(null);
  }

  _render() {
    const gl = this.gl;
    const dpr = window.devicePixelRatio || 1;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (this.count) {
      if (this.renderMode === "voxels") this._drawVoxels();
      else this._drawPoints(this.progRender, dpr);
    }
    requestAnimationFrame(() => this._render());
  }

  // ----- GPU color-picking ------------------------------------------------ //
  _ensurePickTarget(w, h) {
    const gl = this.gl;
    if (this._pickSize[0] === w && this._pickSize[1] === h) return;
    this._pickSize = [w, h];
    gl.bindTexture(gl.TEXTURE_2D, this.pickTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.pickDepth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.pickFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.pickTex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.pickDepth);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // Returns the point index under (px, py) in drawingBuffer pixels, or -1.
  _pickAt(px, py) {
    const gl = this.gl;
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    this._ensurePickTarget(w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.pickFbo);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    this._drawPoints(this.progPick, window.devicePixelRatio || 1);
    gl.clearColor(0.043, 0.055, 0.078, 1.0);

    // Read a small window and take the nearest hit to the cursor (tolerance).
    const R = 6, x0 = Math.max(0, px - R), y0 = Math.max(0, py - R);
    const bw = Math.min(2 * R + 1, w - x0), bh = Math.min(2 * R + 1, h - y0);
    const buf = new Uint8Array(bw * bh * 4);
    gl.readPixels(x0, y0, bw, bh, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    let best = -1, bestD = 1e9;
    for (let yy = 0; yy < bh; yy++) for (let xx = 0; xx < bw; xx++) {
      const o = (yy * bw + xx) * 4;
      const id = buf[o] | (buf[o + 1] << 8) | (buf[o + 2] << 16);
      if (id === 0) continue;
      const dx = (x0 + xx) - px, dy = (y0 + yy) - py, d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = id - 1; }
    }
    return best;
  }

  // ----- interaction / the live loop -------------------------------------- //
  async _pickAndReconstruct(clientX, clientY) {
    if (this.busy) return;                       // constraint 3: single-flight
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const px = Math.round((clientX - rect.left) * dpr);
    const py = Math.round((rect.height - (clientY - rect.top)) * dpr); // gl y is bottom-up
    const idx = this._pickAt(px, py);
    if (idx < 0 || idx >= this.count) { this._setStatus("no point under cursor"); return; }

    const i = this.voxels[idx * 3], j = this.voxels[idx * 3 + 1], k = this.voxels[idx * 3 + 2];
    this._setBusy(true, `reconstructing around (${i}, ${j}, ${k})…`);
    try {
      const resp = await fetch("/reconstruct", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x: i, y: j, z: k }),
      });
      if (resp.status === 409) { this._setStatus("busy — a reconstruction is already running"); return; }
      const data = await resp.json();
      if (!resp.ok) { this._setStatus(`error: ${data.error || resp.status}`); return; }

      // Demote any previous "last-added" (2) to plain reconstructed (1).
      for (let t = 0; t < this.types.length; t++) if (this.types[t] === 2) this.types[t] = 1;
      this._appendVoxels(data.new, 2);           // pure mirror: only append backend output
      this._uploadBuffers();
      this._setStatus(`+${data.added} voxels — ${this.count.toLocaleString()} total`);
    } catch (e) {
      this._setStatus(`request failed: ${e}`);
    } finally {
      this._setBusy(false);
    }
  }

  async reset() {
    if (this.busy) return;
    this._setBusy(true, "resetting…");
    try {
      const data = await (await fetch("/reset", { method: "POST" })).json();
      this.positions = []; this.types = []; this.voxels = [];
      this._appendVoxels(data.original, 0);
      this._appendVoxels(data.reconstructed, 1);
      this._uploadBuffers();
      this._setStatus(`reset — ${this.count.toLocaleString()} voxels`);
    } finally { this._setBusy(false); }
  }

  // ----- UI wiring -------------------------------------------------------- //
  _bindUI() {
    document.getElementById("toggleVoxel").addEventListener("change", (e) => {
      this.renderMode = e.target.checked ? "voxels" : "points";
    });
    document.getElementById("pointSize").addEventListener("input", (e) => {
      this.pointSize = parseFloat(e.target.value);
    });
    document.getElementById("toggleRecon").addEventListener("change", (e) => {
      this.showRecon = e.target.checked;
    });
    document.getElementById("resetBtn").addEventListener("click", () => this.reset());
    document.getElementById("fitBtn").addEventListener("click", () => this.fit());
  }

  _bindPointer() {
    const c = this.canvas;
    let dragging = false, panning = false, lx = 0, ly = 0, moved = 0;
    c.addEventListener("contextmenu", (e) => e.preventDefault());
    c.addEventListener("pointerdown", (e) => {
      dragging = true; panning = e.button === 2 || e.shiftKey; lx = e.clientX; ly = e.clientY;
      moved = 0; c.setPointerCapture(e.pointerId);
    });
    c.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const dx = e.clientX - lx, dy = e.clientY - ly; lx = e.clientX; ly = e.clientY;
      moved += Math.abs(dx) + Math.abs(dy);
      if (panning) {
        const s = this.cam.radius * 0.0015;
        const eye = this._eye(); const fwd = norm(sub(this.cam.target, eye));
        const right = norm(cross(fwd, [0, 1, 0])); const up = cross(right, fwd);
        this.cam.target = [ this.cam.target[0] - (right[0]*dx - up[0]*dy) * s,
                            this.cam.target[1] - (right[1]*dx - up[1]*dy) * s,
                            this.cam.target[2] - (right[2]*dx - up[2]*dy) * s ];
      } else {
        this.cam.theta += dx * 0.008;
        this.cam.phi = Math.min(Math.PI - 0.05, Math.max(0.05, this.cam.phi - dy * 0.008));
      }
    });
    c.addEventListener("pointerup", (e) => {
      dragging = false; c.releasePointerCapture(e.pointerId);
      if (moved < 5 && e.button === 0 && !e.shiftKey) this._pickAndReconstruct(e.clientX, e.clientY);
    });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.cam.radius = Math.min(20, Math.max(0.2, this.cam.radius * (1 + Math.sign(e.deltaY) * 0.1)));
    }, { passive: false });
  }

  _setBusy(on, msg) {
    this.busy = on;
    document.getElementById("loader").classList.toggle("on", on);
    if (msg) this._setStatus(msg);
  }
  _setStatus(s) { document.getElementById("status").textContent = s; }
  _setMeta(d) {
    document.getElementById("meta").innerHTML =
      `volume <b>${d.name}</b> &middot; ${d.shape.join("×")} &middot; cube ${d.cube_size}³`;
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.floor(this.canvas.clientWidth * dpr);
    this.canvas.height = Math.floor(this.canvas.clientHeight * dpr);
  }
}

// --------------------------------------------------------------------------- //
window.addEventListener("DOMContentLoaded", async () => {
  try {
    const demo = new Demo(document.getElementById("gl"));
    window._demo = demo;
    await demo.loadVolume();
  } catch (e) {
    document.getElementById("status").textContent = "init error: " + e.message;
    console.error(e);
  }
});
