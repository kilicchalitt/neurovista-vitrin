/* NeuroVista tanıtım sahnesi: kaydırmaya bağlı WebGL2 sahnesi (bağımlılıksız).
 *
 * Aşamalar (s = sahne bölümünde kaydırılan yol / sahne yüksekliği):
 *   0    → 0.6   sıvı cam beyin: kırılan ışık, içte dönen renk girdapları, parlak yansımalar
 *   0.45 → 1.0   cam beyin büyür ve ışık küresine dönüşür
 *   0.9  → 1.9   nöronlardan oluşan ışık küresi; model sonuç kartları kürenin noktalarına iğnelenir
 *   1.9  → 2.6   kamera kürenin içine girer, nöron ağı ekranı kaplar; iş akışı kartları
 *   2.6  → 3.15  ışık aşağı çekilir, yatay ışık bantları
 *   3.15 → son   yatay sinyal çizgileri ve ortada tek parlak nöron
 * Sahne içindeki [data-show="a,b"] katmanlarının görünürlüğünü ve [data-pin] kartlarının konumunu da yönetir.
 */
(() => {
  'use strict';
  const canvas = document.getElementById('neuroScene');
  const section = document.getElementById('scene');
  if (!canvas || !section) return;
  const root = document.documentElement;
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false, depth: false });
  if (!gl) { root.classList.add('no-webgl'); return; }

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const isLight = () => document.body.classList.contains('light-mode');
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
  const mix = (a, b, t) => a + (b - a) * t;
  const FOCAL = 1.9;
  const easeOut = (t) => 1 - Math.pow(1 - t, 3);

  // ------------------------------------------------------------------ Gölgelendiriciler
  const FULLSCREEN_VS = `#version 300 es
  void main(){ vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;

  const COMMON = `
  mat2 rot(float a){ float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
  vec3 coolPalette(float k){
    vec3 a = vec3(0.30, 0.95, 1.00), b = vec3(0.20, 0.42, 1.00), c = vec3(0.58, 0.34, 1.00), d = vec3(0.92, 0.58, 1.00);
    float x = fract(k);
    return x < 0.33 ? mix(a, b, x / 0.33) : x < 0.66 ? mix(b, c, (x - 0.33) / 0.33) : mix(c, mix(d, a, (x - 0.66) / 0.34), (x - 0.66) / 0.34);
  }`;

  // Sıvı cam beyin: iki loblu, ortasında yarık olan, sürekli dalgalanan saydam gövde.
  const BLOB_FS = `#version 300 es
  precision highp float;
  uniform vec2 uRes, uCenter; uniform float uTime, uAlpha, uLight, uScale, uWhite;
  out vec4 outColor;
  ${COMMON}

  float sdf(vec3 p){
    float t = uTime * 0.35;
    vec3 q = p;
    q.xz *= rot(0.55 + sin(t * 0.6) * 0.35);
    q.yz *= rot(-0.22 + sin(t * 0.45) * 0.12);
    vec3 a = vec3(abs(q.x), q.yz);
    vec3 r = vec3(0.6, 0.7, 0.86);
    float d = (length((a - vec3(0.26, 0.0, 0.0)) / r) - 1.0) * 0.6;
    d += 0.08 * exp(-q.x * q.x * 55.0) * smoothstep(-0.35, 0.45, q.y);            // boylamsal yarık
    d += 0.07 * sin(3.1 * q.x + t * 2.1) * sin(2.7 * q.y + t * 1.7) * sin(2.9 * q.z + t * 1.3);   // sıvı dalga
    d += 0.03 * sin(7.0 * q.x + t * 1.5) * sin(6.5 * q.y - t * 1.2) * sin(7.3 * q.z + t);
    d += 0.012 * sin(15.0 * q.y + 5.0 * sin(3.0 * q.z)) * sin(13.0 * q.z + 4.0 * sin(3.0 * q.x));   // kıvrım izi
    d += 0.016 * sin(11.0 * q.x + t * 2.4) * sin(12.0 * q.y - t * 1.9) * sin(10.5 * q.z + t * 1.6);  // küçük sıvı kabarcıkları
    return d;
  }
  vec3 normalAt(vec3 p){
    const vec2 k = vec2(1.0, -1.0); const float e = 0.002;
    return normalize(k.xyy * sdf(p + k.xyy * e) + k.yyx * sdf(p + k.yyx * e) + k.yxy * sdf(p + k.yxy * e) + k.xxx * sdf(p + k.xxx * e));
  }
  vec3 env(vec3 d){
    vec3 c = mix(vec3(0.01, 0.02, 0.05), vec3(0.10, 0.18, 0.40), smoothstep(-0.3, 0.9, d.y));
    c += vec3(1.0, 1.0, 1.05) * smoothstep(0.90, 0.985, dot(d, normalize(vec3(0.5, 0.75, 0.55)))) * 3.2;
    c += vec3(0.55, 0.85, 1.0) * smoothstep(0.82, 0.97, dot(d, normalize(vec3(-0.85, 0.15, 0.45)))) * 1.6;
    c += vec3(0.7, 0.5, 1.0) * smoothstep(0.85, 0.98, dot(d, normalize(vec3(0.2, -0.8, 0.5)))) * 0.9;
    return c;
  }
  // İçeriden geçen ışın boyunca dönen renk girdaplarını biriktirir (kırılan ışığın içi).
  vec3 interior(vec3 ro, vec3 rd){
    vec3 acc = vec3(0.0); float trans = 1.0, t = 0.02;
    for (int i = 0; i < 16; i++) {
      vec3 p = ro + rd * t;
      if (sdf(p) > 0.005) break;
      float sw = sin(p.x * 7.0 + uTime * 0.7 + sin(p.y * 5.5 + uTime * 0.45) * 1.8) * sin(p.z * 6.5 - uTime * 0.55 + sin(p.x * 4.5) * 1.5);
      float dens = pow(smoothstep(0.3, 1.0, sw * 0.5 + 0.5), 1.6);
      float sw2 = sin(p.y * 6.0 - uTime * 0.6 + sin(p.z * 4.0) * 1.4) * sin(p.x * 5.5 + uTime * 0.4);
      acc += coolPalette(p.y * 0.55 + sw * 0.3 + uTime * 0.04) * dens * trans * 0.3;
      acc += coolPalette(0.55 + p.x * 0.4 + uTime * 0.03) * pow(max(sw2, 0.0), 3.0) * trans * 0.18;
      trans *= 0.87;
      t += 0.095;
    }
    return acc;
  }
  void main(){
    if (uAlpha < 0.004) { outColor = vec4(0.0); return; }
    vec2 uv = ((gl_FragCoord.xy - 0.5 * uRes) / uRes.y - uCenter) / uScale;
    vec3 ro = vec3(0.0, 0.0, 3.4), rd = normalize(vec3(uv, -${FOCAL.toFixed(2)}));
    float b = dot(ro, rd), c = dot(ro, ro) - 1.35 * 1.35, h = b * b - c;
    vec3 col = vec3(0.0); float alpha = 0.0, halo = 0.0;
    if (h > 0.0) {
      float t = max(0.0, -b - sqrt(h)), tMax = -b + sqrt(h);
      bool hit = false;
      for (int i = 0; i < 64; i++) {
        float d = sdf(ro + rd * t);
        halo += exp(-max(d, 0.0) * 14.0) * 0.02;
        if (d < 0.001) { hit = true; break; }
        t += d * 0.8;
        if (t > tMax) break;
      }
      if (hit) {
        vec3 p = ro + rd * t, n = normalAt(p);
        float ndv = max(dot(n, -rd), 0.0);
        float fres = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
        vec3 refl = env(reflect(rd, n));
        vec3 inner = interior(p - n * 0.02, refract(rd, n, 0.75));
        vec3 film = coolPalette(ndv * 1.8 + uTime * 0.06 + p.y * 0.4 + 0.3) * pow(1.0 - ndv, 2.0) * 0.95;   // ince film yanardönerliği
        vec3 body = inner * 1.55 + vec3(0.02, 0.03, 0.10) + vec3(0.08, 0.05, 0.24) * (1.0 - ndv);
        col = mix(body, refl, fres) + film;
        col += pow(max(dot(reflect(rd, n), normalize(vec3(0.5, 0.75, 0.55))), 0.0), 90.0) * 1.8;
        col = mix(col, vec3(0.85, 0.97, 1.0) * 1.6, uWhite);                          // ışık küresine dönüşürken beyazlaşır
        col = mix(col, col * 0.85 + vec3(0.02, 0.05, 0.12), uLight * 0.35);
        alpha = 1.0;
      }
    }
    vec3 haloCol = mix(vec3(0.35, 0.45, 1.0), vec3(0.25, 0.45, 0.95), uLight);
    float haloA = clamp(halo * 0.5, 0.0, 0.55) * (1.0 - alpha);
    outColor = vec4(col * alpha + haloCol * haloA, alpha + haloA) * uAlpha;
  }`;

  // Ara görüntüyü (yarım çözünürlüklü cam beyin) ekrana önceden çarpılmış alfa ile kopyalar.
  const COPY_FS = `#version 300 es
  precision highp float;
  uniform sampler2D uTex; uniform vec2 uRes;
  out vec4 outColor;
  void main(){ outColor = texture(uTex, gl_FragCoord.xy / uRes); }`;

  // Nöron küresi: aynı perspektif; kamera +z'de uCam uzaklıkta.
  const FIELD_COMMON = `
  uniform vec2 uRes, uCenter, uRot; uniform float uCam, uTime, uPx, uScale, uCollapse;
  ${COMMON}
  vec4 project(vec3 p, out float depth){
    p *= 1.0 - 0.85 * uCollapse;
    p.xz *= rot(-uRot.y); p.yz *= rot(-uRot.x);
    depth = uCam - p.z;
    float w = max(depth, 0.0001);
    vec2 uv = ${FOCAL.toFixed(2)} * p.xy * uScale + uCenter * w;
    return vec4(uv.x * 2.0 * uRes.y / uRes.x, uv.y * 2.0, 0.0, w) * (depth > 0.03 ? 1.0 : 0.0);
  }`;

  const POINT_VS = `#version 300 es
  precision highp float;
  in vec3 aPos; in float aSeed;
  ${FIELD_COMMON}
  out float vSeed, vFade;
  void main(){
    vec3 p = aPos + 0.01 * sin(uTime * 0.7 + aSeed * 40.0 + aPos.yzx * 8.0);
    float depth; gl_Position = project(p, depth);
    gl_PointSize = clamp(uPx * (0.45 + aSeed * 1.25) / max(depth, 0.05), 0.0, 60.0);
    vFade = smoothstep(0.06, 0.3, depth);
    vSeed = aSeed;
  }`;

  const POINT_FS = `#version 300 es
  precision highp float;
  uniform float uTime, uIntensity, uLight;
  in float vSeed, vFade;
  out vec4 outColor;
  void main(){
    vec2 c = gl_PointCoord - 0.5; float r2 = dot(c, c);
    if (r2 > 0.25) discard;
    float core = exp(-r2 * 70.0), glow = exp(-r2 * 14.0) * 0.4;
    float fire = 0.5 + 2.4 * pow(0.5 + 0.5 * sin(uTime * (0.9 + vSeed * 2.4) + vSeed * 61.0), 14.0);
    float a = (core + glow) * fire * vFade * uIntensity;
    if (uLight > 0.5) {
      vec3 col = mix(vec3(0.05, 0.35, 0.75), vec3(0.2, 0.25, 0.8), fract(vSeed * 3.7));
      outColor = vec4(col * a * 0.8, a * 0.8);
    } else {
      vec3 col = mix(vec3(0.7, 0.95, 1.0), vec3(0.45, 0.65, 1.0), fract(vSeed * 3.7)) * (1.0 + core);
      outColor = vec4(col * a, a * 0.6);
    }
  }`;

  const LINE_VS = `#version 300 es
  precision highp float;
  in vec3 aPos; in float aSeed;
  ${FIELD_COMMON}
  out float vFade;
  void main(){
    float depth; gl_Position = project(aPos + 0.01 * sin(uTime * 0.7 + aSeed * 40.0 + aPos.yzx * 8.0), depth);
    vFade = smoothstep(0.15, 0.5, depth) * (1.0 - uCollapse);
  }`;

  const LINE_FS = `#version 300 es
  precision highp float;
  uniform float uIntensity, uLight;
  in float vFade;
  out vec4 outColor;
  void main(){
    float a = vFade * uIntensity * mix(0.13, 0.2, uLight);
    outColor = vec4(mix(vec3(0.55, 0.85, 1.0), vec3(0.1, 0.3, 0.7), uLight) * a, a);
  }`;

  const PULSE_VS = `#version 300 es
  precision highp float;
  in vec3 aFrom, aTo; in vec2 aMotion;
  ${FIELD_COMMON}
  out float vFade;
  void main(){
    float k = fract(uTime * aMotion.x + aMotion.y);
    float depth; gl_Position = project(mix(aFrom, aTo, k), depth);
    gl_PointSize = clamp(uPx * 1.1 / max(depth, 0.05), 0.0, 36.0);
    vFade = smoothstep(0.12, 0.4, depth) * sin(3.14159 * k) * (1.0 - uCollapse);
  }`;

  const PULSE_FS = `#version 300 es
  precision highp float;
  uniform float uIntensity, uLight;
  in float vFade;
  out vec4 outColor;
  void main(){
    vec2 c = gl_PointCoord - 0.5; float r2 = dot(c, c);
    if (r2 > 0.25) discard;
    float a = (exp(-r2 * 60.0) + 0.35 * exp(-r2 * 12.0)) * vFade * uIntensity;
    outColor = vec4(mix(vec3(0.9, 1.0, 1.0), vec3(0.0, 0.4, 0.85), uLight) * a, a * (uLight > 0.5 ? 0.9 : 0.5));
  }`;

  // Işık dolgusu, aşağı çekilen bant, yatay sinyal çizgileri ve tek nokta (tam ekran).
  const POST_FS = `#version 300 es
  precision highp float;
  uniform vec2 uRes, uCenter; uniform float uTime, uLight, uScale, uCam, uSphere, uBand, uLines, uDot, uAura;
  out vec4 outColor;
  float hash(float n){ return fract(sin(n * 127.1) * 43758.5453); }
  float hash2(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  void main(){
    vec2 px = gl_FragCoord.xy;
    vec2 uv = (px - 0.5 * uRes) / uRes.y;
    vec2 q = uv - uCenter;
    float r = length(q);
    vec3 col = vec3(0.0); float a = 0.0;
    bool light = uLight > 0.5;

    // Işık küresi dolgusu (kamera dışarıdayken küre, içerideyken tüm ekran)
    if (uSphere > 0.001) {
      float R = uCam > 1.02 ? ${FOCAL.toFixed(2)} * uScale / sqrt(uCam * uCam - 1.0) : 10.0;
      float inside = smoothstep(R * 1.04, R * 0.82, r);
      float limb = smoothstep(R * 0.35, R * 0.98, r) * inside;
      float fill = inside * (0.82 + 0.18 * limb);
      float yb = mix(0.65, -0.75, uBand);
      float band = uBand > 0.0 ? smoothstep(yb + 0.04, yb - 0.18, uv.y) : 1.0;     // ışık aşağı çekilir
      float grain = hash2(floor(px * 0.5) + floor(uTime * 10.0)) * 0.08;
      float f = fill * band * uSphere;
      if (light) { col += mix(vec3(0.86, 0.96, 1.0), vec3(0.42, 0.78, 1.0), limb) * f; a += f * 0.95; }
      else { col += mix(vec3(0.5, 0.88, 1.0), vec3(0.18, 0.5, 1.0), limb * limb) * (1.0 + grain) * f; a += f; }
      if (uBand > 0.0) {                                                              // bant çizgisinden yana uzanan ışık çizgileri
        for (int i = 0; i < 5; i++) {
          float y = yb - 0.03 - float(i) * 0.045;
          float line = exp(-pow((uv.y - y) * uRes.y / 1.6, 2.0)) * smoothstep(0.0, 0.6, abs(uv.x)) * uSphere * (1.0 - uBand * 0.6);
          col += (light ? vec3(0.1, 0.45, 0.9) : vec3(0.75, 0.92, 1.0)) * line; a += line;
        }
      }
    }

    // Yatay sinyal çizgileri: ortada aksiyon potansiyeli biçiminde küçük bir tepe
    if (uLines > 0.001) {
      float step = 0.042;
      float idx = floor((uv.y + 2.0) / step);
      for (int k = -1; k <= 1; k++) {
        float id = idx + float(k);
        float y0 = id * step - 2.0 + (hash(id) - 0.5) * step * 0.6;
        float spike = exp(-pow(uv.x / 0.05, 2.0)) * (0.022 - 0.035 * exp(-pow((uv.x - 0.05) / 0.04, 2.0))) * exp(-abs(y0) * 2.2);
        float y = y0 + spike * (hash(id + 7.0) > 0.5 ? 1.0 : -1.0);
        float d = abs(uv.y - y) * uRes.y;
        float lineA = smoothstep(1.4, 0.2, d) * (0.2 + 0.45 * hash(id + 3.0)) * mix(0.3, 1.0, smoothstep(0.04, 0.3, abs(uv.y - 0.12)));
        float fadeX = smoothstep(1.15, 0.2, abs(uv.x) / (uRes.x / uRes.y) * 1.6);
        lineA *= mix(1.0, fadeX, 0.35) * uLines;
        col += (light ? vec3(0.12, 0.16, 0.24) : vec3(0.92, 0.95, 1.0)) * lineA; a += lineA;
      }
    }
    // Merkezdeki aura ve tek nöron
    if (uAura > 0.001) {
      float aura = exp(-r * r * 3.0) * uAura;
      col += (light ? vec3(0.35, 0.4, 0.95) * 0.35 : vec3(0.3, 0.22, 0.75) * 0.6) * aura; a += aura * (light ? 0.25 : 0.5);
    }
    if (uDot > 0.001) {
      vec2 dq = uv - vec2(0.0, -0.17);
      float dr = dot(dq, dq);
      float dotA = (exp(-dr * 2600.0) + 0.35 * exp(-dr * 260.0)) * uDot;
      col += (light ? vec3(0.05, 0.45, 0.9) : vec3(0.75, 0.95, 1.0)) * dotA; a += dotA;
    }
    outColor = vec4(col, clamp(a, 0.0, 1.0));
  }`;

  // ------------------------------------------------------------------ GL yardımcıları
  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || 'shader');
    return s;
  }
  function program(vs, fs) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs)); gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) || 'link');
    const u = {};
    for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i < n; i++) {
      const name = gl.getActiveUniform(p, i).name;
      u[name] = gl.getUniformLocation(p, name);
    }
    return { p, u, a: (name) => gl.getAttribLocation(p, name) };
  }
  function buffer(data) {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return b;
  }
  function attrib(prog, name, buf, size, stride, offset) {
    const loc = prog.a(name);
    if (loc < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride * 4, offset * 4);
  }
  const set = (prog, name, ...v) => {
    const loc = prog.u[name];
    if (loc == null) return;
    if (v.length === 1) gl.uniform1f(loc, v[0]); else gl.uniform2f(loc, v[0], v[1]);
  };

  let blob, copy, post, points, lines, pulses;
  try {
    blob = program(FULLSCREEN_VS, BLOB_FS);
    copy = program(FULLSCREEN_VS, COPY_FS);
    post = program(FULLSCREEN_VS, POST_FS);
    points = program(POINT_VS, POINT_FS);
    lines = program(LINE_VS, LINE_FS);
    pulses = program(PULSE_VS, PULSE_FS);
  } catch (error) {
    console.warn('NeuroVista sahnesi başlatılamadı:', error);
    root.classList.add('no-webgl');
    return;
  }
  root.classList.add('has-webgl');

  // ------------------------------------------------------------------ Nöron küresi
  const compact = Math.min(window.innerWidth, window.innerHeight) < 700;
  const NEURONS = compact ? 1400 : 2600;
  let seed = 11;
  const rand = () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
  const pos = [];
  for (let i = 0; i < NEURONS; i++) {
    // çoğu kabuğa yakın (parlak kenar), bir kısmı içte
    const u = rand() * 2 - 1, phi = rand() * Math.PI * 2;
    const r = rand() < 0.75 ? 0.62 + 0.38 * Math.sqrt(rand()) : Math.cbrt(rand()) * 0.62;
    const s = Math.sqrt(1 - u * u);
    pos.push(r * s * Math.cos(phi), r * u, r * s * Math.sin(phi));
  }
  const pointData = new Float32Array(NEURONS * 4);
  for (let i = 0; i < NEURONS; i++) pointData.set([pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2], rand()], i * 4);
  const CELL = 0.12, grid = new Map();
  const cellKey = (x, y, z) => `${Math.floor(x / CELL)},${Math.floor(y / CELL)},${Math.floor(z / CELL)}`;
  for (let i = 0; i < NEURONS; i++) {
    const k = cellKey(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(i);
  }
  const edges = [];
  for (let i = 0; i < NEURONS; i++) {
    const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2], near = [];
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const cell = grid.get(`${Math.floor(x / CELL) + dx},${Math.floor(y / CELL) + dy},${Math.floor(z / CELL) + dz}`);
      if (!cell) continue;
      for (const j of cell) {
        if (j <= i) continue;
        const d = Math.hypot(pos[j * 3] - x, pos[j * 3 + 1] - y, pos[j * 3 + 2] - z);
        if (d < CELL) near.push([d, j]);
      }
    }
    near.sort((a, b) => a[0] - b[0]);
    for (const [, j] of near.slice(0, 2)) edges.push(i, j);
  }
  const lineData = new Float32Array(edges.length * 4);
  edges.forEach((n, k) => lineData.set([pos[n * 3], pos[n * 3 + 1], pos[n * 3 + 2], pointData[n * 4 + 3]], k * 4));
  const PULSES = compact ? 260 : 480;
  const pulseData = new Float32Array(PULSES * 8);
  for (let k = 0; k < PULSES; k++) {
    const e = Math.floor(rand() * (edges.length / 2)) * 2;
    const [from, to] = rand() > 0.5 ? [edges[e], edges[e + 1]] : [edges[e + 1], edges[e]];
    pulseData.set([pos[from * 3], pos[from * 3 + 1], pos[from * 3 + 2], pos[to * 3], pos[to * 3 + 1], pos[to * 3 + 2], 0.35 + rand() * 0.9, rand()], k * 8);
  }

  const emptyVao = gl.createVertexArray();
  const blobTarget = { tex: gl.createTexture(), fbo: gl.createFramebuffer(), w: 0, h: 0 };
  function ensureBlobTarget(w, h) {
    if (blobTarget.w === w && blobTarget.h === h) return;
    blobTarget.w = w; blobTarget.h = h;
    gl.bindTexture(gl.TEXTURE_2D, blobTarget.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, blobTarget.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, blobTarget.tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  const pointVao = gl.createVertexArray(); gl.bindVertexArray(pointVao);
  const pointBuf = buffer(pointData);
  attrib(points, 'aPos', pointBuf, 3, 4, 0); attrib(points, 'aSeed', pointBuf, 1, 4, 3);
  const lineVao = gl.createVertexArray(); gl.bindVertexArray(lineVao);
  const lineBuf = buffer(lineData);
  attrib(lines, 'aPos', lineBuf, 3, 4, 0); attrib(lines, 'aSeed', lineBuf, 1, 4, 3);
  const pulseVao = gl.createVertexArray(); gl.bindVertexArray(pulseVao);
  const pulseBuf = buffer(pulseData);
  attrib(pulses, 'aFrom', pulseBuf, 3, 8, 0); attrib(pulses, 'aTo', pulseBuf, 3, 8, 3); attrib(pulses, 'aMotion', pulseBuf, 2, 8, 6);
  gl.bindVertexArray(null);

  // ------------------------------------------------------------------ Katmanlar, iğneler, kaydırma
  const layers = [...section.querySelectorAll('[data-show]')].map((el) => {
    const [a, b] = el.dataset.show.split(',').map(Number);
    return { el, a, b };
  });
  // Kartların iğnelendiği küre noktaları (dünya koordinatı)
  const pins = [...section.querySelectorAll('[data-pin]')].map((el) => {
    const v = el.dataset.pin.split(',').map(Number), n = Math.hypot(...v);
    return { el, p: v.map((x) => (x / n) * 0.96) };
  });

  const maxQuality = compact ? 0.85 : 1;
  let quality = maxQuality, fastFrames = 0, wokeAt = performance.now();
  const BLOB_RES = compact ? 0.7 : 0.62;           // cam beyin bu oranda çizilir (en pahalı geçiş; daha düşüğü kenarları pikselleştirir)
  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5) * quality;
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  };
  window.addEventListener('resize', () => { resize(); wake(); });
  resize();

  let scrollS = 0;
  const readScroll = () => {
    const top = section.getBoundingClientRect().top;
    scrollS = Math.max(0, -top) / Math.max(1, canvas.clientHeight);
  };
  window.addEventListener('scroll', () => { readScroll(); wake(); }, { passive: true });
  readScroll();

  function target(s, aspect) {
    const narrow = aspect < 0.95;
    const heroCenter = narrow ? [0, 0.29] : [Math.min(0.44, aspect * 0.25), -0.01];
    const heroScale = narrow ? Math.min(0.42, aspect * 0.72) : 0.7;
    const toCenter = smooth(0.2, 0.75, s);
    // Cam beyin ekranda ~0.475·ölçek yarıçapında; küre R = odak/√(cam²−1). İkisi 0.6–0.9'da eşit boyda buluşur.
    const blobScale = mix(heroScale, narrow ? 0.9 : 1.15, smooth(0.3, 0.88, s));
    const meetR = 0.475 * blobScale / (narrow ? 0.75 : 1);
    let cam = Math.sqrt(1 + (FOCAL / Math.max(meetR, 0.2)) ** 2);   // ekrana sığan küre: kartlar bunun üstünde
    cam = mix(cam, 1.3, smooth(1.2, 1.95, s));                        // sonra kamera yaklaşır
    cam = mix(cam, 0.45, smooth(1.95, 2.6, s));                       // ve içeri girer
    return {
      centerX: mix(heroCenter[0], 0, toCenter), centerY: mix(heroCenter[1], 0, toCenter),
      blobScale,
      blobAlpha: 1 - smooth(0.66, 0.9, s),
      white: smooth(0.5, 0.84, s),
      cam,
      scale: narrow ? 0.75 : 1,
      field: smooth(0.6, 0.9, s) * (1 - smooth(2.7, 3.05, s)),
      sphere: smooth(0.58, 0.88, s) * (1 - smooth(3.05, 3.2, s)),
      band: smooth(2.6, 3.15, s),
      collapse: smooth(2.6, 3.1, s),
      lines: smooth(3.0, 3.45, s),
      aura: Math.max(0.55 * (1 - smooth(0.3, 0.7, s)), smooth(3.05, 3.5, s)),   // kahramanda arka hale, sonda merkez aurası
      dot: smooth(3.15, 3.45, s),
    };
  }

  const state = {};
  let first = true, running = false, last = performance.now(), clock = 0, slowFrames = 0, sView = 0;
  const startTime = performance.now();
  let intro = reduceMotion ? 1 : 0;                 // açılışta cam beyin büyüyerek belirir

  function updateLayers(s) {
    for (const layer of layers) {
      const { el, a, b } = layer;
      const o = smooth(a, a + 0.16, s) * (1 - smooth(b - 0.16, b, s));
      if (layer.o !== undefined && Math.abs(layer.o - o) < 0.002) continue;
      layer.o = o;
      el.style.opacity = o.toFixed(3);
      el.style.transform = `translate3d(0, ${((1 - o) * 26).toFixed(1)}px, 0)`;
      const hidden = o < 0.01;
      if (layer.hidden !== hidden) { el.style.visibility = hidden ? 'hidden' : 'visible'; layer.hidden = hidden; }
      el.classList.toggle('is-live', o > 0.5);
    }
  }

  function updatePins(rotX, rotY) {
    const W = canvas.clientWidth, H = canvas.clientHeight;
    const cy = Math.cos(-rotY), sy = Math.sin(-rotY), cx = Math.cos(-rotX), sx = Math.sin(-rotX);
    for (const pin of pins) {
      let [x, y, z] = pin.p.map((v) => v * (1 - 0.85 * state.collapse));
      // GLSL'deki p.xz *= rot(a) karşılığı: (x, z) -> (x c - z s, x s + z c)
      [x, z] = [x * cy - z * sy, x * sy + z * cy];
      [y, z] = [y * cx - z * sx, y * sx + z * cx];
      const depth = state.cam - z;
      const visible = depth > 0.3 && z > -0.1;
      const ux = FOCAL * x * state.scale / depth + state.centerX, uy = FOCAL * y * state.scale / depth + state.centerY;
      pin.el.style.transform = `translate3d(${(W / 2 + ux * H).toFixed(1)}px, ${(H / 2 - uy * H).toFixed(1)}px, 0)`;
      if (pin.visible !== visible) { pin.el.classList.toggle('is-behind', !visible); pin.visible = visible; }
    }
  }

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    if (!reduceMotion) clock += dt;
    const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
    sView = first ? scrollS : mix(sView, scrollS, 1 - Math.pow(0.0004, dt));
    if (Math.abs(sView - scrollS) < 0.0005) sView = scrollS;
    intro = reduceMotion ? 1 : Math.min(1, (now - startTime) / 1600);
    const t = target(sView, aspect);
    for (const name in t) state[name] = t[name];
    first = false;
    updateLayers(sView);

    // Uyarlanabilir kalite: uyanıştan sonraki ilk 1 sn sayılmaz; tabanı 0.75, kareler hızlanınca geri yükselir
    if (now - wokeAt > 1000) {
      slowFrames = dt > 0.026 ? slowFrames + 1 : Math.max(0, slowFrames - 1);
      fastFrames = dt < 0.019 ? fastFrames + 1 : 0;
      if (slowFrames > 45 && quality > 0.75) { quality = Math.max(0.75, quality - 0.1); slowFrames = 0; resize(); }
      if (fastFrames > 240 && quality < maxQuality) { quality = Math.min(maxQuality, quality + 0.1); fastFrames = 0; resize(); }
    }

    const light = isLight() ? 1 : 0;
    const rotX = 0.25 + Math.sin(clock * 0.13) * 0.06, rotY = clock * 0.06 + sView * 0.35;
    updatePins(0.18, 0);                                 // kartlar kürenin kenarlarında sabit (başlığın üstüne binmez)

    const W = canvas.width, H = canvas.height;
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    const additive = () => gl.blendFunc(gl.ONE, gl.ONE);
    const over = () => gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    // 1) ışık dolgusu, bant, çizgiler, nokta
    if (state.sphere > 0.002 || state.lines > 0.002 || state.dot > 0.002 || state.aura > 0.002) {
      over();
      gl.useProgram(post.p); gl.bindVertexArray(emptyVao);
      set(post, 'uRes', W, H); set(post, 'uCenter', state.centerX, state.centerY); set(post, 'uTime', clock);
      set(post, 'uLight', light); set(post, 'uScale', state.scale); set(post, 'uCam', state.cam);
      set(post, 'uSphere', state.sphere * 0.95); set(post, 'uBand', state.band);
      set(post, 'uLines', state.lines); set(post, 'uDot', state.dot); set(post, 'uAura', state.aura);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    // 2) nöronlar
    if (state.field > 0.002) {
      const shared = (prog) => {
        set(prog, 'uRes', W, H); set(prog, 'uCenter', state.centerX, state.centerY); set(prog, 'uRot', rotX, rotY);
        set(prog, 'uCam', state.cam); set(prog, 'uTime', clock); set(prog, 'uPx', H * 0.0042);
        set(prog, 'uScale', state.scale); set(prog, 'uCollapse', state.collapse); set(prog, 'uLight', light);
      };
      const band = 1 - state.band;
      light ? over() : additive();
      gl.useProgram(lines.p); shared(lines); set(lines, 'uIntensity', state.field * band);
      gl.bindVertexArray(lineVao); gl.drawArrays(gl.LINES, 0, edges.length);
      gl.useProgram(points.p); shared(points); set(points, 'uIntensity', state.field * mix(1.5, 0.5, state.band));
      gl.bindVertexArray(pointVao); gl.drawArrays(gl.POINTS, 0, NEURONS);
      gl.useProgram(pulses.p); shared(pulses); set(pulses, 'uIntensity', state.field * band);
      gl.bindVertexArray(pulseVao); gl.drawArrays(gl.POINTS, 0, PULSES);
    }
    // 3) cam beyin: yarım çözünürlükte ara görüntüye, sonra ekrana
    const blobAlpha = state.blobAlpha * intro;
    if (blobAlpha > 0.004) {
      const bw = Math.max(1, Math.round(W * BLOB_RES)), bh = Math.max(1, Math.round(H * BLOB_RES));
      ensureBlobTarget(bw, bh);
      gl.bindFramebuffer(gl.FRAMEBUFFER, blobTarget.fbo);
      gl.viewport(0, 0, bw, bh);
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.disable(gl.BLEND);
      gl.useProgram(blob.p); gl.bindVertexArray(emptyVao);
      set(blob, 'uRes', bw, bh); set(blob, 'uCenter', state.centerX, state.centerY); set(blob, 'uTime', clock);
      set(blob, 'uAlpha', blobAlpha); set(blob, 'uLight', light);
      set(blob, 'uScale', state.blobScale * (0.82 + 0.18 * easeOut(intro))); set(blob, 'uWhite', state.white);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, W, H);
      gl.enable(gl.BLEND); over();
      gl.useProgram(copy.p);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, blobTarget.tex);
      gl.uniform1i(copy.u.uTex, 0); set(copy, 'uRes', W, H);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.bindVertexArray(null);

    // Bölüm ekrandan çıktıysa ya da sekme gizliyse döngüyü uyut (kaydırma geri uyandırır)
    const rect = section.getBoundingClientRect();
    const offscreen = rect.bottom < 0 || rect.top > window.innerHeight;
    const settled = sView === scrollS && intro >= 1;
    if (offscreen || document.hidden || (reduceMotion && settled)) { running = false; return; }
    requestAnimationFrame(frame);
  }

  function wake() {
    if (running || document.hidden) return;
    running = true; last = wokeAt = performance.now();
    first = true;                                    // uykudan dönüşte eski konumdan kaydırarak geçme; doğrudan yeni konum
    requestAnimationFrame(frame);
  }
  document.addEventListener('visibilitychange', wake);
  new MutationObserver(wake).observe(document.body, { attributes: true, attributeFilter: ['class'] });
  canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); running = false; root.classList.add('no-webgl'); });
  wake();
})();
