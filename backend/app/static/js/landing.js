/* landing.html betiği (C4: sayfadan ayrıldı) */
    // Tema Yönetimi
    const themeBtn = document.getElementById('themeBtn');
    const savedTheme = localStorage.getItem('neurovista-theme');
    if (savedTheme === 'light') document.body.classList.add('light-mode');
    
    const updateThemeIcon = () => {
      const name = document.body.classList.contains('light-mode') ? 'moon' : 'sun';
      themeBtn.innerHTML = `<svg class="ico" aria-hidden="true"><use href="#i-${name}"/></svg>`;
    };
    updateThemeIcon();

    themeBtn.onclick = () => {
      document.body.classList.toggle('light-mode');
      localStorage.setItem('neurovista-theme', document.body.classList.contains('light-mode') ? 'light' : 'dark');
      updateThemeIcon();
    };

    // Dinamik Sekme Verileri (4 Tab). Sayılar data-result öğeleriyle /api/v1/results'tan gelir.
    const R = (path, format) => `<span data-result="${path}" data-format="${format}">—</span>`;
    const tabData = [
      {
        title: "Üç Model ve Kilitli Ensemble",
        desc: "3D U-Net, SwinUNETR ve MedNeXt aynı hasta hacmini bağımsız işler. Ana maske, validation kümesinde önceden kilitlenen kuralla (etiket çoğunluğu) üç modelden üretilir; her modelin maskesi ayrıca gösterilir.",
        color: "emerald",
        solid: "#2dd4bf",
        btnGradient: "linear-gradient(115deg, #0d9488, #0891b2)",
        btnGlow: "rgba(45, 212, 191, 0.3)",
        features: [
          "Üç farklı mimari: klasik CNN, Transformer ve modern CNN",
          `Ana maske kilitli ensemble: iç test ${R("internal_test.configs.ensemble|PRIMARY.case_mean", "dice")} hasta-bazlı Dice`,
          "NCR, ED ve ET alt bölgeleri; WT, TC ve ET metrikleri"
        ],
        mockup: [
          ["Ana Maske", "Kilitli ensemble (etiket çoğunluğu)"],
          ["Modeller", "3D U-Net · SwinUNETR · MedNeXt-S"],
          ["Girdi Modaliteleri", "T1 · T1ce · T2 · FLAIR"],
          ["Hedef Bölgeler", "NCR (Nekrotik) · ED (Ödem) · ET (Aktif)"],
          ["Voksel Çözünürlüğü", "1.0 × 1.0 × 1.0 mm (İzotropik)"]
        ]
      },
      {
        title: "Intel OpenVINO FP32 CPU Hızlandırma",
        desc: "CPU modunda üç model de Intel OpenVINO ile çalışır. Hızlanma ve etiket uyumu her model için PyTorch'a karşı ayrı ölçüldü; PyTorch bilimsel referanstır ve hata durumunda devreye girer.",
        color: "cyan",
        solid: "#38bdf8",
        btnGradient: "linear-gradient(115deg, #0891b2, #2563eb)",
        btnGlow: "rgba(56, 189, 248, 0.3)",
        features: [
          `CPU hızlanması: U-Net ${R("openvino.unet.speedup", "x")}, Swin ${R("openvino.swin.speedup", "x")}, MedNeXt ${R("openvino.mednext.speedup", "x")}`,
          `Etiket (argmax) uyumu en az ${R("derived.min_argmax_agreement", "pct5")}`,
          "Olası hatalarda PyTorch'a otomatik dönüş"
        ],
        mockup: [
          ["Hızlanma Aralığı", R("derived.speedup_range", "xrange")],
          ["En Düşük Etiket Uyumu", R("derived.min_argmax_agreement", "pct5")],
          ["Çıkarım Motoru", "OpenVINO FP32 Runtime"],
          ["Ölçüm Vakası", R("openvino.unet.case", "text")]
        ]
      },
      {
        title: "Uyum Skoru, Konsensüs ve Olasılık Haritası",
        desc: "Uyum skoru, üç modelin maskeleri arasındaki ortalama ikili Dice'tır ve Ground Truth gerektirmez; düşük uyum hatalı segmentasyon riskine işaret eder. Konsensüs paneli her vokselde kaç modelin tümör bulduğunu, olasılık haritası ana maskenin softmax tümör olasılığını gösterir (Grad-CAM değildir).",
        color: "indigo",
        solid: "#8b7cff",
        btnGradient: "linear-gradient(115deg, #4f46e5, #7c3aed)",
        btnGlow: "rgba(139, 124, 255, 0.3)",
        features: [
          `Uyum ile gerçek Dice ilişkisi: iç test Spearman ρ = ${R("internal_test.agreement.spearman_rho", "rho")}`,
          `Başarısız vakayı ayırt etme: iç test AUROC ${R("internal_test.agreement.auroc_low_agreement_detects_failure", "auroc")}, harici ${R("derived.external_auroc_range", "range2")}`,
          "Konsensüs: tam uyum (N/N), çoğunluk ve azınlık vokselleri"
        ],
        mockup: [
          ["Uyum Skoru", "Modeller arası ortalama ikili Dice"],
          ["Konsensüs", "N/N · çoğunluk · azınlık"],
          ["Olasılık Haritası", "Softmax tümör olasılığı"],
          ["Not", "Açıklanabilirlik (Grad-CAM) yöntemi değildir"]
        ]
      },
      {
        title: "Etkileşimli 3B WebGL Anatomik Rekonstrüksiyon",
        desc: "Segmentasyon çıktılarından Marching Cubes algoritmasıyla gerçek poligon mesh yüzeyleri örülür. Tarayıcıda eklenti gerekmeden beyin dokusu ve tümör katmanları 3 boyutlu olarak incelenir.",
        color: "blue",
        solid: "#5b8cff",
        btnGradient: "linear-gradient(115deg, #2563eb, #4f46e5)",
        btnGlow: "rgba(91, 140, 255, 0.3)",
        features: [
          "Gerçek MR ve tahmin hacminden anlık Marching Cubes mesh üretimi",
          "Beyin, nekroz, ödem ve aktif tümör katmanlarını açıp kapama",
          "Fareyle 360 derece döndürme, yakınlaştırma ve 2B kesitle hizalı aksiyel görünüm"
        ],
        mockup: [
          ["3B Motoru", "Saf WebGL / WebGL2 (Sıfır Eklenti)"],
          ["Mesh Algoritması", "Marching Cubes (Voxel-to-Surface)"],
          ["Katmanlar", "Beyin Kabuğu · NCR · ED · ET"],
          ["Görünüm Modları", "Aksiyel · İzometrik · Sagittal"]
        ]
      }
    ];

    const tabButtons = document.querySelectorAll('.tab-pill');
    const tabTitle = document.getElementById('tabTitle');
    const tabDesc = document.getElementById('tabDesc');
    const tabList = document.getElementById('tabList');
    const tabBtn = document.getElementById('tabBtn');
    const tabMockup = document.getElementById('tabMockup');

    const showTab = (index) => {
      const data = tabData[index];
      tabButtons.forEach(b => {
        b.className = 'tab-pill';
      });
      tabButtons[index].classList.add('active', data.color);

      tabTitle.textContent = data.title;
      tabDesc.textContent = data.desc;
      tabBtn.style.background = data.btnGradient;
      tabBtn.style.boxShadow = `0 14px 34px -12px ${data.btnGlow}`;

      tabList.innerHTML = data.features.map(f => `
        <li>
          <div class="showcase-check" style="background:${data.btnGlow};color:${data.solid}"><svg class="ico" aria-hidden="true"><use href="#i-check"/></svg></div>
          <span>${f}</span>
        </li>
      `).join('');

      tabMockup.innerHTML = data.mockup.map(([k, v]) => `
        <div class="mockup-row"><span>${k}</span><strong>${v}</strong></div>
      `).join('');

      if (window.NeuroVistaResults) window.NeuroVistaResults.bind(document.getElementById('showcaseCard'));
    };

    tabButtons.forEach(btn => {
      btn.onclick = () => showTab(parseInt(btn.dataset.tab));
    });
    showTab(0);
    if (window.NeuroVistaResults) {
      window.NeuroVistaResults.ready.then(() => window.NeuroVistaResults.bind(document.getElementById('showcaseCard')));
    }

    // Sahnenin belirli bir aşamasına giden bağlantılar (data-scene-s = sahne yüksekliği cinsinden konum)
    document.querySelectorAll('[data-scene-s]').forEach((link) => {
      link.addEventListener('click', (e) => {
        const scene = document.getElementById('scene');
        if (!scene) return;
        e.preventDefault();
        const header = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--header-h')) || 80;
        const y = scene.getBoundingClientRect().top + window.scrollY + parseFloat(link.dataset.sceneS) * (window.innerHeight - header);
        if (window.NeuroVistaScroll) window.NeuroVistaScroll.go(y); else window.scrollTo({ top: y, behavior: 'smooth' });
      });
    });

    // Köşedeki araştırma bildirimi: kapatılınca bu oturumda yeniden gösterilmez
    const notice = document.getElementById('noticeCard');
    const noticeClose = document.getElementById('noticeClose');
    try { if (sessionStorage.getItem('neurovista-notice') === 'closed') notice.hidden = true; } catch (e) { /* depolama kapalı */ }
    noticeClose.onclick = () => {
      notice.hidden = true;
      try { sessionStorage.setItem('neurovista-notice', 'closed'); } catch (e) { /* depolama kapalı */ }
    };

    // Kaydırınca beliren kartlar (IntersectionObserver yoksa her şey baştan görünür)
    if ('IntersectionObserver' in window) {
      document.documentElement.classList.add('js-reveal');
      const revealer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add('is-visible');
          revealer.unobserve(entry.target);
        });
      }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });
      document.querySelectorAll('.reveal').forEach((el) => revealer.observe(el));
    }

    /* Sayfa Yenileme: Her zaman en üste kaydır */
    if (history.scrollRestoration) history.scrollRestoration = 'manual';
    window.scrollTo(0, 0);
