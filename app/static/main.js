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
  // View matrix straight from a camera basis (right/up/back) + eye. Replaces a
  // lookAt(eye, target, WORLD_UP): the basis comes from the trackball quaternion,
  // so the camera keeps whatever orientation the user rotated into instead of
  // being re-levelled to a fixed world up vector every frame.
  viewFromBasis(right, up, back, eye) {
    return [right[0],up[0],back[0],0, right[1],up[1],back[1],0, right[2],up[2],back[2],0,
            -dot(right,eye),-dot(up,eye),-dot(back,eye),1];
  },
  identity() {
    return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
  },
  /**
   * Model rotation from INITIAL_VOLUMES_ROTATION: [x, y, z] DEGREES, applied as
   * extrinsic X then Y then Z (R = Rz*Ry*Rx) about the origin -- which is the
   * volume's centre, since _worldOf() already centres the cloud there.
   *
   * This orients the drawn model only. Voxel indices are untouched, so clicks
   * still send the backend unrotated coordinates (picking resolves a vertex ID
   * to `this.voxels`, never to a world position).
   */
  rotationXYZ(dx, dy, dz) {
    const r = Math.PI / 180;
    const cx = Math.cos(dx*r), sx = Math.sin(dx*r);
    const cy = Math.cos(dy*r), sy = Math.sin(dy*r);
    const cz = Math.cos(dz*r), sz = Math.sin(dz*r);
    // Rows of Rz*Ry*Rx, written out column-major for GL.
    const m = [
      [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
      [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
      [-sy,   cy*sx,            cy*cx           ],
    ];
    return [m[0][0],m[1][0],m[2][0],0,
            m[0][1],m[1][1],m[2][1],0,
            m[0][2],m[1][2],m[2][2],0,
            0,0,0,1];
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
// Quaternion helpers -- the camera ORIENTATION (trackball).                    //
//                                                                             //
// The camera used to be spherical angles (theta, phi) turned into a view with
// lookAt(eye, target, WORLD_UP). That makes a drag rotate about the *global* Y
// axis, so the motion stops matching what you see as soon as you orbit away
// from the equator (and it gimbal-locks at the poles). Instead we keep a
// quaternion for the camera frame and apply every drag delta about the camera's
// OWN axes -- i.e. in view space, not model space -- so dragging always moves
// the object in the direction of the cursor, from any viewpoint.
// --------------------------------------------------------------------------- //
const Q = {
  identity: () => [0, 0, 0, 1],
  // Rotation of `angle` radians about `axis` (axis must be unit length).
  axisAngle(axis, angle) {
    const s = Math.sin(angle / 2);
    return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
  },
  // Hamilton product. `multiply(q, d)` applies d in q's LOCAL frame (post-
  // multiply) -- that is the whole trick behind view-relative rotation.
  multiply(a, b) {
    const [ax,ay,az,aw] = a, [bx,by,bz,bw] = b;
    return [ aw*bx + ax*bw + ay*bz - az*by,
             aw*by - ax*bz + ay*bw + az*bx,
             aw*bz + ax*by - ay*bx + az*bw,
             aw*bw - ax*bx - ay*by - az*bz ];
  },
  normalize(q) {
    const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    return [q[0]/l, q[1]/l, q[2]/l, q[3]/l];
  },
  // Rotate a vector by the quaternion: v' = v + w*t + u x t, with t = 2 * (u x v).
  rotate(q, v) {
    const u = [q[0], q[1], q[2]], w = q[3];
    const t = cross(u, v).map((c) => 2 * c);
    const c = cross(u, t);
    return [v[0] + w*t[0] + c[0], v[1] + w*t[1] + c[1], v[2] + w*t[2] + c[2]];
  },
  // Camera frame from yaw (about world Y) then pitch (about the camera's own X).
  yawPitch(yaw, pitch) {
    return Q.multiply(Q.axisAngle([0, 1, 0], yaw), Q.axisAngle([1, 0, 0], pitch));
  },
};

// The opening three-quarter view: identical framing to the old spherical camera's
// (azimuth 0.9, polar 1.1) starting pose, re-expressed as a camera orientation.
const DEFAULT_ROT = () => Q.yawPitch(Math.PI / 2 - 0.9, 1.1 - Math.PI / 2);

// Turntable: degrees per second, and how long the demo sits untouched before it
// starts spinning by itself (attract mode).
const SPIN_DEG_PER_SEC = 18;
const IDLE_SPIN_AFTER_MS = 30000;
// Longest frame the turntable will honour. A backgrounded tab delivers one huge
// delta on return; without this the model would lurch by half a turn.
const MAX_FRAME_SECONDS = 0.1;

// --------------------------------------------------------------------------- //
// Shaders (GLSL ES 3.00).                                                     //
// --------------------------------------------------------------------------- //
const VS_RENDER = `#version 300 es
layout(location=0) in vec3 a_position;
layout(location=1) in float a_type;      // 0 input, 1 reconstructed, 2 last-added
uniform mat4 u_proj, u_view, u_model;
uniform float u_pointSize;
uniform float u_showRecon;
out vec3 v_color;
out float v_discard;
void main() {
  gl_Position = u_proj * u_view * u_model * vec4(a_position, 1.0);
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
uniform mat4 u_proj, u_view, u_model;
uniform float u_pointSize;
uniform float u_showRecon;
flat out vec3 v_id;
out float v_discard;
void main() {
  gl_Position = u_proj * u_view * u_model * vec4(a_position, 1.0);
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
uniform mat4 u_proj, u_view, u_model;
uniform float u_voxelSize;
uniform float u_showRecon;
out vec3 v_color; out vec3 v_normal; flat out float v_cull;
void main() {
  v_cull = (a_type > 0.5 && u_showRecon < 0.5) ? 1.0 : 0.0;
  gl_Position = u_proj * u_view * u_model * vec4(a_offset + a_cubePos * u_voxelSize, 1.0);
  v_normal = mat3(u_model) * a_normal;
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

// The crop box: the DATA_2D_SIZE^3 region the clicked cube was taken from, i.e.
// exactly what the 6 projections in the bottom panel were computed from. Drawn
// as a unit cube stretched by uniforms, so nothing is re-uploaded when it moves.
const VS_BOX = `#version 300 es
layout(location=0) in vec3 a_position;   // unit cube (edge 1, centered)
uniform mat4 u_proj, u_view, u_model;
uniform vec3 u_boxCenter, u_boxSize;
void main() {
  gl_Position = u_proj * u_view * u_model * vec4(u_boxCenter + a_position * u_boxSize, 1.0);
}`;
const FS_BOX = `#version 300 es
precision highp float;
uniform vec4 u_color;
out vec4 fragColor;
void main() { fragColor = u_color; }`;

// The 12 edges of a unit cube (centered, edge 1) as a GL_LINES vertex list.
function unitCubeEdges() {
  const c = [
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5],  // 0..3 bottom
    [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5,  0.5, 0.5], [-0.5,  0.5, 0.5],  // 4..7 top
  ];
  const pairs = [[0,1],[1,2],[2,3],[3,0], [4,5],[5,6],[6,7],[7,4], [0,4],[1,5],[2,6],[3,7]];
  const out = [];
  for (const [a, b] of pairs) out.push(...c[a], ...c[b]);
  return new Float32Array(out);            // 24 vertices
}

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
    this.progBox = program(gl, VS_BOX, FS_BOX);

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

    // --- crop-box VAOs (both just position; the cube is stretched by uniforms) ---
    // Faces reuse the voxel renderer's unit cube; edges get their own line list.
    this.boxFaceVao = gl.createVertexArray();
    gl.bindVertexArray(this.boxFaceVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cubePosBuf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    this.boxEdgeBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.boxEdgeBuf);
    gl.bufferData(gl.ARRAY_BUFFER, unitCubeEdges(), gl.STATIC_DRAW);
    this.boxEdgeVao = gl.createVertexArray();
    gl.bindVertexArray(this.boxEdgeVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.boxEdgeBuf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    // Offscreen framebuffer for GPU color-picking.
    this.pickFbo = gl.createFramebuffer();
    this.pickTex = gl.createTexture();
    this.pickDepth = gl.createRenderbuffer();
    this._pickSize = [0, 0];

    // Trackball camera state: an orientation quaternion (not Euler angles) +
    // distance to the target. Drags rotate `rot` about the camera's own axes.
    this.cam = { rot: DEFAULT_ROT(), radius: 2.4, target: [0, 0, 0] };
    this.pointSize = 4.0;
    this.showRecon = true;
    this.renderMode = "voxels";   // "voxels" (lit cubes) or "points"
    this.busy = false;

    // 2D projections panel state (last picked cube's before/after views).
    this.views = { before: {}, after: {} };
    this.flipSide = "after";

    // The last picked cube's crop region: {start:[x,y,z], size} in voxel coords.
    // Same lifetime as the projections panel -- they describe the same cube.
    this.lastCube = null;
    this.showCropBox = false;

    // Last GET /cache response; gates the "Load Cached Result" button.
    this.cache = null;

    // Display orientation of the volume, from the dataset's
    // INITIAL_VOLUMES_ROTATION. Render-only -- see M4.rotationXYZ.
    this.baseRot = M4.identity();
    this.modelRot = M4.identity();

    // Turntable state. `on` is whether it is spinning right now; `manual` is
    // what the user asked for. Attract mode can spin without touching `manual`,
    // so the user's own choice is never quietly overwritten.
    this.spin = { on: false, manual: false, idle: false, angle: 0,
                  idleEnabled: true, lastInput: performance.now() };

    // Coalesce many live voxel appends into one GPU upload per frame.
    this.dirty = false;

    this._bindUI();
    this._bindPointer();
    this._resize();
    window.addEventListener("resize", () => this._resize());
    requestAnimationFrame((t) => this._render(t));
  }

  // ----- data ------------------------------------------------------------- //
  _worldOf(i, j, k) {
    // Center the cloud and scale into a comfortable unit-ish box.
    const s = 1.0 / Math.max(this.shape[0], this.shape[1], this.shape[2]);
    return [(i - this.center[0]) * s, (j - this.center[1]) * s, (k - this.center[2]) * s];
  }

  /**
   * Voxel coordinates arrive base64-packed (see DemoState.occupied_coords) --
   * JSON-encoding ~921k integers cost the server seconds. Decodes to a typed
   * array, which _appendVoxels indexes exactly like a plain array. Plain arrays
   * (the small per-cube SSE progress events) are passed straight through.
   */
  _decodeCoords(field) {
    if (!field) return [];
    if (Array.isArray(field)) return field;
    const binary = atob(field.data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return field.enc === "int32" ? new Int32Array(bytes.buffer) : new Int16Array(bytes.buffer);
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

  /** Rebuild the whole cloud from a backend state snapshot. */
  _applySnapshot(data, refit = false) {
    const rot = data.rotation || [0, 0, 0];
    this.baseRot = M4.rotationXYZ(rot[0], rot[1], rot[2]);
    this.spin.angle = 0;                    // a different object starts unspun
    this._updateModelRot();
    this.shape = data.shape;
    this.cubeSize = data.cube_size;
    this.center = [this.shape[0] / 2, this.shape[1] / 2, this.shape[2] / 2];
    // A voxel spans one index unit; _worldOf scales by 1/max(shape). 0.9 leaves a
    // hairline gap between neighbours so cube edges stay legible.
    this.voxelSize = (1.0 / Math.max(this.shape[0], this.shape[1], this.shape[2])) * 0.9;
    this.positions = []; this.types = []; this.voxels = [];
    this._appendVoxels(this._decodeCoords(data.original), 0);
    this._appendVoxels(this._decodeCoords(data.reconstructed), 1);
    this._uploadBuffers();
    if (refit) this.fit();          // a different object: reframe the camera
    this._setMeta(data);
    // Every path that reseats the volume comes through here (load / select /
    // upload / reset), so this is the one place the cache button must resync.
    this._refreshCacheState();
  }

  async loadVolume() {
    const data = await (await fetch("/volume")).json();
    this._applySnapshot(data, true);
    this._setStatus(`loaded ${data.name} — ${this.count.toLocaleString()} voxels`);
  }

  // ----- dataset / volume pickers ----------------------------------------- //
  /** Populate both pickers from GET /configs (the app/configs/ descriptors). */
  async loadConfigs() {
    const data = await (await fetch("/configs")).json();
    const cfgSel = document.getElementById("configSel");
    const volSel = document.getElementById("volumeSel");

    cfgSel.innerHTML = "";
    for (const c of data.configs) {
      const o = document.createElement("option");
      o.value = c.name;
      o.textContent = c.available ? c.label : `${c.label} (unavailable)`;
      o.disabled = !c.available;
      o.selected = c.name === data.active.config;
      o.title = c.available ? c.config_filename : c.problems.join("; ");
      cfgSel.appendChild(o);
    }
    // Without the supervisor there is no process to restart us (see server.py).
    cfgSel.disabled = !data.supervised;
    cfgSel.title = data.supervised
      ? "Switching dataset restarts the backend on that config"
      : "Started with --serve — dataset switching needs the supervisor";

    const active = data.configs.find((c) => c.name === data.active.config);
    volSel.innerHTML = "";
    // A "Load file…" volume isn't in VOLUMES_PATH, so give it its own entry to
    // sit on -- otherwise the picker would misleadingly show some other volume.
    if (data.active.custom) {
      const o = document.createElement("option");
      o.value = "";                                     // no path: not re-selectable
      o.textContent = `${data.active.volume_name} (loaded file)`;
      o.selected = true;
      volSel.appendChild(o);
    }
    for (const v of (active ? active.volumes : [])) {
      const o = document.createElement("option");
      o.value = v.path; o.textContent = v.name;
      o.selected = v.path === data.active.volume;
      volSel.appendChild(o);
    }
    volSel.disabled = volSel.options.length < 2;
  }

  /** Same config, different volume -> no restart, the models stay loaded. */
  async selectVolume(path) {
    if (this.busy) return;
    this._setBusy(true, "Loading volume…");
    try {
      const resp = await fetch("/volume/select", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      const data = await resp.json();
      if (!resp.ok) { this._setStatus(`error: ${data.detail || data.error || resp.status}`); return; }
      this._applySnapshot(data, true);
      this._clearPanel();                       // the projections belong to the old volume
      this._setStatus(`loaded ${data.name} — ${this.count.toLocaleString()} voxels`);
    } catch (e) {
      this._setStatus(`request failed: ${e}`);
    } finally {
      this._setBusy(false);
    }
  }

  /** Load a volume the user picked from anywhere on their machine (no restart). */
  async uploadVolume(file) {
    if (!file || this.busy) return;
    this._setBusy(true, `Loading ${file.name}…`);
    try {
      const form = new FormData();
      form.append("file", file);
      const resp = await fetch("/volume/upload", { method: "POST", body: form });
      const data = await resp.json();
      if (!resp.ok) {
        this._setStatus(`error: ${data.error || resp.status}${data.detail ? " — " + data.detail : ""}`);
        return;
      }
      this._applySnapshot(data, true);
      this._clearPanel();                       // the projections belong to the old volume
      this._setStatus(`loaded ${data.name} — ${this.count.toLocaleString()} voxels`);
    } catch (e) {
      this._setStatus(`upload failed: ${e}`);
    } finally {
      this._setBusy(false);
      try { await this.loadConfigs(); } catch (_) {}   // show it in the volume picker
    }
  }

  /**
   * Relaunch the backend on the CURRENT dataset, then reload the page.
   *
   * A from-scratch restart without having to switch away and back: the worker
   * re-reads the descriptor and the configs/ yaml, reloads the volume and the
   * model weights, and the page reload picks up edited JS/CSS too. Use it after
   * editing anything the running process already read.
   */
  async reloadApp() {
    if (this.busy) return;
    const name = this.cache ? this.cache.config : null;
    const active = name || (await (await fetch("/configs")).json()).active.config;
    this._setBusy(true, "Reloading app…");
    try {
      // restart:true on purpose -- a dataset switch rebinds in place (fast), but
      // Reload App is what you press after editing code, so it must relaunch.
      const resp = await fetch("/config", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: active, restart: true }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        const detail = Array.isArray(data.detail) ? data.detail.join("; ") : (data.detail || "");
        this._setStatus(`reload failed: ${data.error || resp.status}${detail ? " — " + detail : ""}`);
        this._setBusy(false);
        return;
      }
      await this._waitForBackend();
      window.location.reload();          // leaves the loader up until the page swaps
    } catch (e) {
      this._setStatus(`reload failed: ${e.message || e}`);
      this._setBusy(false);
    }
  }

  /**
   * Different config -> the backend restarts on it (the active config is fixed
   * at import time; see the process-model note in server.py). Wait for the new
   * worker to answer, then redraw from it.
   */
  async switchConfig(name) {
    if (this.busy) return;
    this._setBusy(true, "Switching dataset…");
    try {
      const resp = await fetch("/config", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        const detail = Array.isArray(data.detail) ? data.detail.join("; ") : (data.detail || "");
        this._setStatus(`error: ${data.error || resp.status}${detail ? " — " + detail : ""}`);
        await this.loadConfigs();               // re-sync the picker with reality
        return;
      }
      if (data.restarting === false) {
        // Fast path: the backend rebound in place and handed us the new state,
        // so there is nothing to wait for.
        this._applySnapshot(data, true);
        await this.loadConfigs();
        this._clearPanel();
        this._setStatus(`dataset: ${data.label} — ${this.count.toLocaleString()} voxels`);
        return;
      }
      this._setBusy(true, `Loading ${data.label} — restarting backend…`);
      await this._waitForBackend();
      await this.loadVolume();
      await this.loadConfigs();
      this._clearPanel();
      this._setStatus(`dataset: ${data.label}`);
    } catch (e) {
      this._setStatus(`switch failed: ${e.message || e}`);
      try { await this.loadConfigs(); } catch (_) {}
    } finally {
      this._setBusy(false);
    }
  }

  /** Poll until the relaunched worker serves again (model load takes a while). */
  async _waitForBackend(timeoutMs = 180000) {
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    await sleep(700);                           // don't mistake the dying worker for the new one
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      try {
        const r = await fetch("/volume", { cache: "no-store" });
        if (r.ok) return true;
      } catch (_) { /* still down */ }
      await sleep(200);                         // tight poll: the wait IS the delay
    }
    throw new Error("backend did not come back in time");
  }

  // ----- camera ----------------------------------------------------------- //
  // The camera's own axes in world space. `back` points from target to eye
  // (the +Z of view space), so eye = target + back * radius.
  _axes() {
    const q = this.cam.rot;
    return { right: Q.rotate(q, [1, 0, 0]), up: Q.rotate(q, [0, 1, 0]), back: Q.rotate(q, [0, 0, 1]) };
  }
  _eye(axes) {
    const { back } = axes || this._axes();
    const { radius, target } = this.cam;
    return [target[0] + back[0]*radius, target[1] + back[1]*radius, target[2] + back[2]*radius];
  }

  /**
   * Rotate the camera by a drag delta expressed in SCREEN pixels.
   *
   * Both deltas are applied about the camera's LOCAL axes (yaw about its own up,
   * pitch about its own right) by post-multiplying the orientation. That is what
   * makes the rotation view-relative: dragging right always sweeps the object
   * right across the screen, whatever the current viewpoint, with no pole
   * singularity and no clamping.
   */
  rotateBy(dx, dy, speed = 0.008) {
    const yaw = Q.axisAngle([0, 1, 0], -dx * speed);
    const pitch = Q.axisAngle([1, 0, 0], -dy * speed);
    this.cam.rot = Q.normalize(Q.multiply(this.cam.rot, Q.multiply(yaw, pitch)));
  }

  /** Roll about the view axis (the axis pointing at the viewer). */
  rollBy(dx, speed = 0.008) {
    this.cam.rot = Q.normalize(Q.multiply(this.cam.rot, Q.axisAngle([0, 0, 1], dx * speed)));
  }

  fit() {
    this.cam.target = [0, 0, 0];
    this.cam.radius = 2.4;
    this.cam.rot = DEFAULT_ROT();
    this.spin.angle = 0;              // put the object back where the config aimed it
    this._updateModelRot();
  }

  // ----- rendering -------------------------------------------------------- //
  _matrices() {
    const gl = this.gl;
    const aspect = gl.drawingBufferWidth / Math.max(1, gl.drawingBufferHeight);
    const proj = M4.perspective(Math.PI / 4, aspect, 0.01, 100);
    const axes = this._axes();
    const view = M4.viewFromBasis(axes.right, axes.up, axes.back, this._eye(axes));
    return { proj, view };
  }

  _drawPoints(prog, dpr) {
    const gl = this.gl;
    const { proj, view } = this._matrices();
    gl.useProgram(prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, "u_proj"), false, proj);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, "u_view"), false, view);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, "u_model"), false, this.modelRot);
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
    gl.uniformMatrix4fv(gl.getUniformLocation(p, "u_model"), false, this.modelRot);
    gl.uniform1f(gl.getUniformLocation(p, "u_voxelSize"), this.voxelSize);
    gl.uniform1f(gl.getUniformLocation(p, "u_showRecon"), this.showRecon ? 1.0 : 0.0);
    gl.bindVertexArray(this.cubeVao);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 36, this.count);
    gl.bindVertexArray(null);
  }

  /**
   * World-space center + extent of the last picked cube's crop region.
   *
   * Voxel i is drawn as a cell centered on _worldOf(i) with edge `s`, so the
   * crop spans from the near face of cell `start` to the far face of cell
   * `start + size - 1` -- i.e. exactly `size` cells wide.
   */
  _cropBoxWorld() {
    const { start, size } = this.lastCube;
    const s = 1.0 / Math.max(this.shape[0], this.shape[1], this.shape[2]);
    const half = s / 2;
    const center = [0, 1, 2].map((a) => (start[a] + size / 2 - this.center[a]) * s - half);
    return { center, size: [size * s, size * s, size * s] };
  }

  /**
   * Model matrix = turntable spin ON TOP OF the dataset's base orientation.
   *
   * Order matters: the spin is applied after the base rotation, about world Y,
   * so it is a horizontal turntable about the screen vertical whatever the
   * dataset's INITIAL_VOLUMES_ROTATION happens to be. (Base-then-spin would
   * turn the object about whichever of its own axes happened to land there.)
   */
  _updateModelRot() {
    this.modelRot = M4.multiply(M4.rotationXYZ(0, this.spin.angle, 0), this.baseRot);
  }

  /** Start/stop the turntable. `manual` marks it as the user's own choice. */
  setAnimate(on, manual = true) {
    this.spin.on = !!on;
    if (manual) { this.spin.manual = !!on; this.spin.idle = false; }
    const box = document.getElementById("toggleAnimate");
    if (box) box.checked = this.spin.on;
    // Lighting is fixed in world space, so a spinning model self-shades; the
    // canvas keeps its crosshair because clicking still works while it turns.
  }

  /** Any real interaction: reset the idle timer and drop out of attract mode. */
  _noteInput() {
    this.spin.lastInput = performance.now();
    if (this.spin.idle) {
      this.spin.idle = false;
      this.setAnimate(this.spin.manual, false);   // back to what the user chose
    }
  }

  /**
   * Advance the turntable by `dt` seconds, and start it by itself after a quiet
   * spell. The step is clamped here rather than at the call site, so the "never
   * lurch" guarantee holds however this gets driven.
   */
  _tickSpin(dt) {
    if (this.spin.idleEnabled && !this.spin.on && !this.busy &&
        performance.now() - this.spin.lastInput > IDLE_SPIN_AFTER_MS) {
      this.spin.idle = true;
      this.setAnimate(true, false);               // attract mode, not a user choice
    }
    if (!this.spin.on) return;
    const step = Math.min(MAX_FRAME_SECONDS, Math.max(0, dt));
    this.spin.angle = (this.spin.angle + step * SPIN_DEG_PER_SEC) % 360;
    this._updateModelRot();
  }

  /** The red box around the region the bottom panel's projections came from. */
  _drawCropBox() {
    const gl = this.gl;
    const { proj, view } = this._matrices();
    const { center, size } = this._cropBoxWorld();
    const p = this.progBox;
    gl.useProgram(p);
    gl.uniformMatrix4fv(gl.getUniformLocation(p, "u_proj"), false, proj);
    gl.uniformMatrix4fv(gl.getUniformLocation(p, "u_view"), false, view);
    gl.uniformMatrix4fv(gl.getUniformLocation(p, "u_model"), false, this.modelRot);
    gl.uniform3fv(gl.getUniformLocation(p, "u_boxCenter"), center);
    gl.uniform3fv(gl.getUniformLocation(p, "u_boxSize"), size);
    const color = gl.getUniformLocation(p, "u_color");

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // Faint tinted faces make the region obvious at a glance. No depth write and
    // front-face culling only, so the structure inside stays fully visible.
    gl.depthMask(false);
    gl.enable(gl.CULL_FACE);
    gl.uniform4f(color, 1.0, 0.25, 0.28, 0.10);
    gl.bindVertexArray(this.boxFaceVao);
    gl.drawArrays(gl.TRIANGLES, 0, 36);
    gl.disable(gl.CULL_FACE);

    gl.bindVertexArray(this.boxEdgeVao);
    // Edges twice: solid where they are genuinely in front, faint where the
    // volume occludes them -- so the box reads as 3D but never disappears.
    gl.uniform4f(color, 1.0, 0.28, 0.30, 1.0);
    gl.drawArrays(gl.LINES, 0, 24);
    gl.disable(gl.DEPTH_TEST);
    gl.uniform4f(color, 1.0, 0.45, 0.48, 0.30);
    gl.drawArrays(gl.LINES, 0, 24);
    gl.enable(gl.DEPTH_TEST);

    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(null);
  }

  _render(timestamp) {
    const gl = this.gl;
    const dpr = window.devicePixelRatio || 1;
    // Spin in degrees per SECOND, not per frame, so it looks the same whatever
    // the refresh rate. _tickSpin clamps the step itself.
    const now = timestamp || performance.now();
    const dt = (now - (this._lastFrame || now)) / 1000;
    this._lastFrame = now;
    this._tickSpin(dt);
    if (this.dirty) { this._uploadBuffers(); this.dirty = false; }  // coalesced live uploads
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (this.count) {
      if (this.renderMode === "voxels") this._drawVoxels();
      else this._drawPoints(this.progRender, dpr);
    }
    // After the volume: the box is an overlay on top of the structure.
    if (this.showCropBox && this.lastCube) this._drawCropBox();
    requestAnimationFrame((t) => this._render(t));
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

      // Remember the region those projections came from, for the crop box.
      if (data.start) this.lastCube = { start: data.start, size: this.cubeSize };

      // Update the 2D projections panel (what the network saw vs. produced).
      if (data.views) { this.views = data.views; this._setFlip("after"); }
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
      this._applySnapshot(data);            // same volume: keep the camera where it is
      this._setStatus(`reset — ${this.count.toLocaleString()} voxels`);
    } finally { this._setBusy(false); }
  }

  // ----- UI wiring -------------------------------------------------------- //
  _bindUI() {
    document.getElementById("configSel").addEventListener("change", (e) => {
      this.switchConfig(e.target.value);
    });
    document.getElementById("volumeSel").addEventListener("change", (e) => {
      if (!e.target.value) return;              // the "(loaded file)" entry has no path
      this.selectVolume(e.target.value);
    });
    document.getElementById("loadFileBtn").addEventListener("click", () => {
      document.getElementById("fileInput").click();
    });
    document.getElementById("fileInput").addEventListener("change", (e) => {
      const file = e.target.files[0];
      e.target.value = "";                      // so re-picking the same file fires again
      this.uploadVolume(file);
    });
    document.getElementById("toggleVoxel").addEventListener("change", (e) => {
      this.renderMode = e.target.checked ? "voxels" : "points";
      this._syncPointSize();
    });
    document.getElementById("pointSize").addEventListener("input", (e) => {
      this.pointSize = parseFloat(e.target.value);
    });
    document.getElementById("toggleRecon").addEventListener("change", (e) => {
      this.showRecon = e.target.checked;
    });
    document.getElementById("toggleCropBox").addEventListener("change", (e) => {
      this.showCropBox = e.target.checked;
    });
    document.getElementById("toggleAnimate").addEventListener("change", (e) => {
      this.setAnimate(e.target.checked);
    });
    document.getElementById("toggleIdleSpin").addEventListener("change", (e) => {
      this.spin.idleEnabled = e.target.checked;
      this.spin.lastInput = performance.now();
      if (!e.target.checked && this.spin.idle) this.setAnimate(this.spin.manual, false);
    });
    document.getElementById("reloadBtn").addEventListener("click", () => this.reloadApp());
    document.getElementById("resetBtn").addEventListener("click", () => this.reset());
    document.getElementById("fitBtn").addEventListener("click", () => this.fit());
    document.getElementById("fullBtn").addEventListener("click", () => this.runFullInference());
    document.getElementById("cacheBtn").addEventListener("click", () => {
      this.runFullInference({ cacheOnly: true });
    });
    document.getElementById("cancelBtn").addEventListener("click", () => this.cancelFullInference());
    document.getElementById("flipBtn").addEventListener("click", () => {
      this._setFlip(this.flipSide === "after" ? "before" : "after");
    });
    this._syncPointSize();
  }

  /** Point size drives gl_PointSize, which only the points renderer uses --
   *  so the slider is live in points (PCD) mode and greyed out in voxel view. */
  _syncPointSize() {
    const on = this.renderMode === "points";
    document.getElementById("pointSize").disabled = !on;
    document.getElementById("pointSizeRow").classList.toggle("off", !on);
  }

  // ----- 2D projections panel --------------------------------------------- //
  _setFlip(side) {
    this.flipSide = side;
    const b = document.getElementById("flipBtn");
    b.className = side;
    b.innerHTML = `Showing: ${side === "after" ? "After" : "Before"} &nbsp;&#8635;`;
    this._renderPanel();
  }

  /** Hide the projections panel — its images belong to a volume we just left.
   *  The crop box describes the same cube, so it goes with them. The toggle
   *  itself lives in that panel, so it disappears with it; `showCropBox` is
   *  left alone so the preference survives to the next click. */
  _clearPanel() {
    this.views = { before: {}, after: {} };
    document.getElementById("views").innerHTML = "";
    document.getElementById("panel").classList.remove("on");
    this.lastCube = null;
  }

  _renderPanel() {
    const views = this.views[this.flipSide] || {};
    const order = ["top", "bottom", "front", "back", "left", "right"];
    const el = document.getElementById("views");
    el.innerHTML = "";
    for (const v of order) {
      if (!views[v]) continue;
      const fig = document.createElement("figure");
      const img = document.createElement("img"); img.src = views[v]; img.alt = v;
      const cap = document.createElement("figcaption"); cap.textContent = v;
      fig.appendChild(img); fig.appendChild(cap); el.appendChild(fig);
    }
    if (el.children.length) document.getElementById("panel").classList.add("on");
  }

  // ----- full inference (streamed progress) ------------------------------- //
  /**
   * `Run Full Inference` computes for real and stores the result;
   * `Load Cached Result` ({cacheOnly:true}) replays app/cache and never
   * computes. Kept explicit so what happens on stage is never a surprise —
   * one button shows the pipeline working, the other is the instant path.
   */
  runFullInference(opts = {}) {
    if (this.busy) return;
    const cacheOnly = !!opts.cacheOnly;
    const query = cacheOnly ? "?cache_only=1" : "?refresh=1";
    this._setBusy(true, cacheOnly ? "Loading cached result…" : "Running full inference…");
    this._progressOn(true);
    this._showCancel(!cacheOnly);              // a cached replay finishes in ~2s
    this._setProgress(0, 0, 0);
    let done = false;
    const es = new EventSource("/full_inference" + query);
    es.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.type === "progress") {
        // Live repair: append this cube's new voxels; the render loop uploads once/frame.
        if (m.new && m.new.length) { this._appendVoxels(m.new, 1); this.dirty = true; }
        this._setProgress(m.done, m.total, m.added);
      } else if (m.type === "done") {
        done = true; es.close();
        this._setProgress(m.done, m.total, m.added);
        this.dirty = true;
        this._showCancel(false); this._progressOn(false); this._setBusy(false);
        const tag = m.cancelled ? "cancelled" : "complete";
        // `cached` = replayed from app/cache; `saved` = this run filled the cache.
        const src = m.cached ? " — from cache" : (m.saved ? " — cached for next time" : "");
        this._setStatus(`full inference ${tag} — +${(m.added || 0).toLocaleString()} voxels `
                        + `(${m.done}/${m.total} cubes)${src}`);
        this._refreshCacheState();             // a fresh run may have just filled it
      } else if (m.type === "error") {
        done = true; es.close();
        this._showCancel(false); this._progressOn(false); this._setBusy(false);
        this._setStatus(`full inference error: ${m.error}`);
        this._refreshCacheState();
      }
    };
    es.onerror = () => {
      if (done) return;
      es.close(); this._showCancel(false); this._progressOn(false); this._setBusy(false);
      this._setStatus("full inference: connection error");
    };
  }

  /** Sync the "Load Cached Result" button with GET /cache. */
  async _refreshCacheState() {
    try {
      const info = await (await fetch("/cache", { cache: "no-store" })).json();
      this.cache = info;
      const b = document.getElementById("cacheBtn");
      const entry = info.entry || {};
      b.disabled = this.busy || !info.loadable;
      if (info.loadable) {
        // Explicit line break (the button is `white-space: pre-line`) so the count
        // reads as a deliberate second line instead of an overflowing wrap.
        b.textContent = `Load Cached Result
${(entry.added || 0).toLocaleString()} voxels`;
        b.title = `Cached ${entry.created} — ${entry.total_cubes} cubes, `
                + `${entry.seconds}s to compute. Loads instantly.`;
      } else {
        b.textContent = "Load Cached Result";
        b.title = info.reason ? `Unavailable: ${info.reason}`
                              : "No cached result for this volume yet";
      }
    } catch (_) { /* leave the button as it was */ }
  }

  cancelFullInference() {
    // Keep the EventSource open: the server finishes its in-flight cube(s),
    // persists the partial result, and sends the final "done" we handle above.
    this._setStatus("cancelling…");
    document.getElementById("cancelBtn").disabled = true;
    fetch("/full_inference/cancel", { method: "POST" }).catch(() => {});
  }

  _showCancel(on) {
    const b = document.getElementById("cancelBtn");
    b.style.display = on ? "" : "none";
    b.disabled = false;
  }

  _bindPointer() {
    const c = this.canvas;
    // Attract mode watches for ANY interaction, including the control panel --
    // otherwise it would start spinning while you are still using the UI.
    for (const ev of ["pointerdown", "pointermove", "wheel", "keydown"]) {
      window.addEventListener(ev, () => this._noteInput(), { passive: true, capture: true });
    }
    let dragging = false, panning = false, rolling = false, lx = 0, ly = 0, moved = 0;
    c.addEventListener("contextmenu", (e) => e.preventDefault());
    c.addEventListener("pointerdown", (e) => {
      dragging = true;
      panning = e.button === 2 || e.shiftKey;
      rolling = !panning && (e.ctrlKey || e.altKey);
      lx = e.clientX; ly = e.clientY;
      moved = 0; c.setPointerCapture(e.pointerId);
    });
    c.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const dx = e.clientX - lx, dy = e.clientY - ly; lx = e.clientX; ly = e.clientY;
      moved += Math.abs(dx) + Math.abs(dy);
      if (panning) {
        // Pan along the camera's own screen axes, so the content tracks the cursor.
        const s = this.cam.radius * 0.0015;
        const { right, up } = this._axes();
        this.cam.target = [ this.cam.target[0] - (right[0]*dx - up[0]*dy) * s,
                            this.cam.target[1] - (right[1]*dx - up[1]*dy) * s,
                            this.cam.target[2] - (right[2]*dx - up[2]*dy) * s ];
      } else if (this.spin.on) {
        // Turntable owns the object's orientation while it runs -- letting the
        // trackball fight it would make the spin stutter and drift. Panning and
        // zooming stay live, so you can still frame what you are watching.
      } else if (rolling) {
        this.rollBy(dx);                      // Ctrl/Alt-drag: spin about the view axis
      } else {
        this.rotateBy(dx, dy);                // view-relative trackball
      }
    });
    c.addEventListener("pointerup", (e) => {
      dragging = false; c.releasePointerCapture(e.pointerId);
      const plainClick = e.button === 0 && !e.shiftKey && !e.ctrlKey && !e.altKey;
      if (moved < 5 && plainClick) this._pickAndReconstruct(e.clientX, e.clientY);
    });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.cam.radius = Math.min(20, Math.max(0.2, this.cam.radius * (1 + Math.sign(e.deltaY) * 0.1)));
    }, { passive: false });
  }

  _setBusy(on, msg) {
    this.busy = on;
    document.getElementById("loader").classList.toggle("on", on);
    document.getElementById("fullBtn").disabled = on;
    document.getElementById("reloadBtn").disabled = on;
    // Stays disabled while busy, and whenever there is nothing cached to load.
    document.getElementById("cacheBtn").disabled = on || !(this.cache && this.cache.loadable);
    if (msg) { document.getElementById("loaderText").textContent = msg; this._setStatus(msg); }
    if (!on) this._progressOn(false);
  }
  _progressOn(on) { document.getElementById("loaderBar").classList.toggle("on", on); }
  _setProgress(done, total, added) {
    const pct = total ? Math.round((100 * done) / total) : 0;
    document.getElementById("loaderBar").firstElementChild.style.width = pct + "%";
    const extra = added != null ? ` · +${added.toLocaleString()} voxels` : "";
    document.getElementById("loaderText").textContent =
      total ? `Full inference — ${done}/${total} cubes (${pct}%)${extra}` : "Preparing full inference…";
  }
  _setStatus(s) { document.getElementById("status").textContent = s; }
  _setMeta(d) {
    const cfg = d.config_label ? `${d.config_label} &middot; ` : "";
    document.getElementById("meta").innerHTML =
      `${cfg}volume <b>${d.name}</b> &middot; ${d.shape.join("×")} &middot; cube ${d.cube_size}³`;
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
    await demo.loadConfigs();
    await demo._refreshCacheState();
  } catch (e) {
    document.getElementById("status").textContent = "init error: " + e.message;
    console.error(e);
  }
});
