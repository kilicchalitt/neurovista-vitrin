/* Yumuşak tekerlek kaydırması (tanıtım ve analiz sayfaları; bağımlılıksız).
 *
 * Fare tekerleğinin 100 px'lik basamaklarını hedefe doğru akan tek bir harekete çevirir; kaydırmaya bağlı
 * sahne ve beliren kartlar bu sayede sarsılmadan ilerler. Yalnız ince işaretçili (fare) cihazlarda ve
 * hareket azaltma tercihi yokken çalışır. Klavye, kaydırma çubuğu ve dokunmatik kaydırma tarayıcıda kalır;
 * iç içe kaydırılabilir alanlarda (ör. sol panel, açık pencereler) tekerlek o alana bırakılır.
 */
(() => {
  'use strict';
  const fine = window.matchMedia('(pointer: fine)').matches;
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!fine || reduce) return;

  const root = document.documentElement;
  root.style.scrollBehavior = 'auto';               // CSS'teki smooth kaydırma bu modülle çakışmasın
  let target = window.scrollY, current = window.scrollY, animating = false, last = 0;
  const maxScroll = () => root.scrollHeight - window.innerHeight;

  // Tekerlek olayının yolunda kendi kaydırması olan bir öğe varsa ona bırak
  function scrollableParent(node) {
    for (let el = node; el && el !== document.body && el !== root; el = el.parentElement) {
      if (!(el instanceof Element)) continue;
      const style = getComputedStyle(el);
      if (/(auto|scroll)/.test(style.overflowY) && el.scrollHeight > el.clientHeight + 1) return el;
    }
    return null;
  }

  function step(now) {
    const dt = Math.min(0.05, (now - last) / 1000 || 0.016); last = now;
    current += (target - current) * (1 - Math.pow(0.00002, dt));   // çerçeve hızından bağımsız yumuşatma
    if (Math.abs(target - current) < 0.4) current = target;
    window.scrollTo(0, current);
    if (current !== target) requestAnimationFrame(step); else animating = false;
  }
  function go(y) {
    target = Math.max(0, Math.min(maxScroll(), y));
    if (!animating) { animating = true; current = window.scrollY; last = performance.now(); requestAnimationFrame(step); }
  }

  window.addEventListener('wheel', (e) => {
    if (e.ctrlKey || e.defaultPrevented || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    if (document.querySelector('.modal-overlay.active')) return;
    if (scrollableParent(e.target)) return;
    e.preventDefault();
    const unit = e.deltaMode === 1 ? 40 : e.deltaMode === 2 ? window.innerHeight : 1;
    if (!animating) target = window.scrollY;
    go(target + e.deltaY * unit);
  }, { passive: false });

  // Klavye, çubuk ya da bağlantıyla yapılan kaydırmada hedefi gerçek konuma eşitle
  window.addEventListener('scroll', () => { if (!animating) { target = current = window.scrollY; } }, { passive: true });
  window.addEventListener('resize', () => { target = Math.min(target, maxScroll()); });

  // Sayfa içi bağlantılar da aynı yumuşak hareketle gider
  document.addEventListener('click', (e) => {
    const link = e.target.closest && e.target.closest('a[href^="#"]');
    if (!link || link.dataset.sceneS || link.getAttribute('href').length < 2) return;
    const dest = document.querySelector(link.getAttribute('href'));
    if (!dest) return;
    e.preventDefault();
    const header = parseFloat(getComputedStyle(root).getPropertyValue('--header-h')) || 80;
    go(dest.getBoundingClientRect().top + window.scrollY - header);
    history.replaceState(null, '', link.getAttribute('href'));
  });
  window.NeuroVistaScroll = { go };
})();
