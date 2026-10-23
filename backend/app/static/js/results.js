/*
 * Kilitli sonuçları tek kaynaktan sayfaya yazar.
 *
 * Kaynak: /api/v1/results → results/v2_results.json (tools/build_results.py arşivlenmiş
 * kayıtlardan üretir). Sayfalarda sayı elle yazılmaz; öğe yalnız yolu ve biçimi taşır:
 *
 *   <span data-result="internal_test.configs.ensemble|PRIMARY.case_mean" data-format="dice">—</span>
 *
 * Yol, JSON anahtarlarının noktayla birleşimidir. "derived." ile başlayan yollar aşağıdaki
 * DERIVED tablosunda aynı JSON'dan hesaplanır. Kaynak okunamazsa öğeler "—" olarak kalır;
 * sayfada asla eski bir sayı görünmez. tests/test_web_results_binding.py her yolun JSON'da
 * çözüldüğünü ve her biçimin burada tanımlı olduğunu denetler.
 */
(function () {
  'use strict';

  const minus = (v) => (v < 0 ? '−' : '+');

  const FORMATS = {
    dice: (v) => v.toFixed(4),
    dice3: (v) => v.toFixed(3),
    auroc: (v) => v.toFixed(2),
    rho: (v) => v.toFixed(2),
    ci: (v) => `${v[0].toFixed(3)}–${v[1].toFixed(3)}`,
    range2: (v) => `${v[0].toFixed(2)}–${v[1].toFixed(2)}`,
    signed: (v) => minus(v) + Math.abs(v).toFixed(4),
    points: (v) => minus(v) + Math.abs(v * 100).toFixed(1),
    pct5: (v) => '%' + (v * 100).toFixed(5),
    x: (v) => v.toFixed(1) + '×',
    xrange: (v) => `${v[0].toFixed(1)}×–${v[1].toFixed(1)}×`,
    sec: (v) => v.toFixed(1),
    mm: (v) => v.toFixed(2),
    int: (v) => String(Math.round(v)),
    text: (v) => String(v),
  };

  const values = (obj) => Object.keys(obj).map((k) => obj[k]);

  // Türetilmiş değerler: yalnız JSON'daki sayılardan hesaplanır.
  const DERIVED = {
    'derived.speedup_range': (d) => {
      const s = values(d.openvino).map((m) => m.speedup);
      return [Math.min(...s), Math.max(...s)];
    },
    'derived.min_argmax_agreement': (d) => Math.min(...values(d.openvino).map((m) => m.argmax_agreement)),
    'derived.external_primary_cases': (d) =>
      values(d.external).filter((c) => c.analysis === 'birincil').reduce((n, c) => n + c.cases, 0),
    'derived.external_auroc_range': (d) => {
      const a = ['africa_glioma', 'utsw', 'africa_other'].map(
        (k) => d.external[k].agreement.auroc_low_agreement_detects_failure);
      return [Math.min(...a), Math.max(...a)];
    },
  };

  function resolve(data, path) {
    if (Object.prototype.hasOwnProperty.call(DERIVED, path)) return DERIVED[path](data);
    return path.split('.').reduce((o, key) => (o == null ? undefined : o[key]), data);
  }

  let data = null;

  function bind(root) {
    if (!data) return;
    (root || document).querySelectorAll('[data-result]').forEach((el) => {
      const format = FORMATS[el.dataset.format || 'text'];
      const value = resolve(data, el.dataset.result);
      if (format && value !== undefined && value !== null) {
        el.textContent = format(value);
        el.classList.add('result-bound');
      }
    });
  }

  const ready = fetch('/api/v1/results', { cache: 'no-store' })
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
    .then((json) => { data = json; bind(document); return json; })
    .catch(() => null);

  window.NeuroVistaResults = { ready, bind, resolve: (p) => (data ? resolve(data, p) : undefined) };
})();
