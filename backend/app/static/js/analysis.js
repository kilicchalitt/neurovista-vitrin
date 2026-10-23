/* index.html betiği (C4: sayfadan ayrıldı) */
    const $ = id => document.getElementById(id);
    /* Sunucu iletileri ve dosya adından gelen hasta adı HTML olarak yorumlanmasın (XSS). */
    const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);
    const panels = $("panels");
    /* Model kartları ve panelleri sunucunun etkin model setinden (/health, analiz sonucu) üretilir. */
    const MODEL_META = {
      unet: { name: '3D U-Net', tag: 'CNN' },
      swin: { name: 'SwinUNETR', tag: 'Transformer' },
      mednext: { name: 'MedNeXt-S', tag: 'ConvNeXt' },
      ensemble: { name: 'Ensemble', tag: 'Ana maske' }
    };
    /* Tekil modeller iç test hasta-bazlı Dice sırasıyla (results/v2_results.json: Swin 0.9140 > MedNeXt 0.9125 > U-Net 0.9105). */
    const MODEL_RANK = ['swin', 'mednext', 'unet'];
    const byRank = keys => [...keys].sort((a, b) => (MODEL_RANK.indexOf(a) + 1 || 99) - (MODEL_RANK.indexOf(b) + 1 || 99));
    /* Ortak SVG simgeleri (index.html içindeki sprite); emoji kullanılmaz. */
    const icon = name => `<svg class="ico" aria-hidden="true"><use href="#i-${name}"/></svg>`;
    let modelKeys = ['unet', 'swin'], hasEnsemble = false;
    let sourceInfo = [];
    /* Panel sırası en önemliden önemsize: ana maske, kaynak MR, yorum panelleri, referans, tekil modeller.
       Her giriş [kaynak anahtarı, başlık, alt etiket] birlikte taşınır; görüntü adresi, başlık ve etiket hiçbir zaman ayrışmaz. */
    function buildSourceInfo(keys, ensemble, hasGt = true) {
      const list = [];
      if (ensemble) list.push(['ensemble', 'Ensemble', 'Ana maske']);
      list.push(['original', 'Orijinal MR', 'Kaynak']);
      list.push(['consensus', 'Model Konsensüsü', 'Uyum']);
      list.push(['activation', 'Tümör Olasılığı', 'Olasılık']);
      if (hasGt) list.push(['gt', 'Ground Truth', 'Referans']);
      byRank(keys).forEach(key => list.push([key, `${(MODEL_META[key] || { name: key }).name} Tahmini`, (MODEL_META[key] || { tag: '' }).tag]));
      return list;
    }
    function sourceLabel(source) {
      if (source === 'gt') return 'Ground Truth';
      if (source === 'ensemble') return 'Ensemble (ana maske)';
      return `${(MODEL_META[source] || { name: source }).name} tahmini`;
    }
    function setMeshSources(sources, primary) {
      const select = $('meshSource');
      select.innerHTML = sources.map(source => `<option value="${source}">${source === 'gt' ? 'Ground Truth Referansı' : sourceLabel(source).replace(' tahmini', ' Tahmini')}</option>`).join('');
      if (primary && sources.includes(primary)) select.value = primary;
    }
    function renderMetrics(data) {
      const box = $('metrics');
      const keys = data ? data.model_order : modelKeys;
      const cards = [];
      const ens = data ? data.ensemble : null;
      if (data ? ens : hasEnsemble) cards.push(['ensemble', ens]);
      byRank(keys).forEach(key => cards.push([key, data ? data.models[key] : null]));
      const value = (info, region) => info ? (info.dice ? info.dice[region].toFixed(4) : 'GT yok') : '—';
      box.innerHTML = cards.map(([key, info]) => {
        const meta = MODEL_META[key] || { name: key };
        const sub = key === 'ensemble' && info
          ? `<div class="metric-sub">${info.method_name}${info.et_postprocessing ? ` · küçük ET eşiği ${info.et_postprocessing}` : ''} · validation'da kilitli</div>` : '';
        const time = info && info.inference_time_sec != null ? info.inference_time_sec + ' s' : '—';
        return `<article class="metric-card ${key}">
            <div class="metric-title"><i class="dot" aria-hidden="true"></i>${meta.name}${key === 'ensemble' ? '' : ` (${meta.tag})`}${key === 'ensemble' ? '<span class="primary-tag">Ana maske</span>' : ''}</div>
            ${sub}
            <div class="metric-grid">
              <span>WT <b>${value(info, 'WT')}</b></span>
              <span>TC <b>${value(info, 'TC')}</b></span>
              <span>ET <b>${value(info, 'ET')}</b></span>
              <span>Süre <b>${time}</b></span>
              <span class="volume">WT Hacmi <b>${info ? info.volumes_cm3.WT + ' cc' : '—'}</b></span>
            </div>
          </article>`;
      }).join('');
    }
    let patientId = null, totalSlices = 0, currentSlice = 0, currentModality = 3, selected = [];
    let isCurrentSelectionAnalyzed = false;
    let lastAnalyzedSignature = null;

    function getFileSignature(files) {
      if (!files || !files.length) return '';
      return [...files].map(f => `${f.name}:${f.size}:${f.lastModified}`).sort().join('|');
    }

    function makePanels() {
      panels.innerHTML = '';
      panels.style.setProperty('--panel-cols', sourceInfo.length <= 6 ? sourceInfo.length : 4);
      sourceInfo.forEach(([source, title, tag], index) => {
        const box = document.createElement('article');
        box.className = 'panel' + (source === 'ensemble' ? ' primary-panel' : '');
        box.dataset.source = source;
        box.innerHTML = `
          <div class="panel-head">
            <span class="panel-number">${String(index + 1).padStart(2, '0')}</span>
            <i class="dot" aria-hidden="true"></i>
            <h3>${title}</h3>
          </div>
          <div class="image-frame">
            <div id="${source}Placeholder" class="placeholder">
              <span>Çıkarım<br>Bekleniyor</span>
            </div>
            <img id="${source}" hidden alt="${source === 'ensemble' ? 'Ensemble (ana maske)' : title}">
          </div>
          <div class="panel-foot">
            <span>${tag}</span>
            <i>Aksiyel</i>
          </div>
        `;
        panels.append(box);
      });
    }
    sourceInfo = buildSourceInfo(modelKeys, hasEnsemble);
    makePanels();
    renderMetrics(null);

    function status(text, bad = false) {
      $('status').textContent = text;
      $('status').className = 'progress' + (bad ? ' error' : '');
    }

    /* Yalnız seçilen dosyalar kullanılır; tek bir NIfTI (ZIP değilse) dört modaliteyi karşılayamaz. */
    function canStart() {
      return selected.length > 1 || (selected.length === 1 && selected[0].name.toLowerCase().endsWith('.zip'));
    }

    function classify() {
      const names = selected.map(file => file.name.toLowerCase());
      if (selected.length === 1 && names[0].endsWith('.zip')) return 'ZIP arşivi seçildi. T1, T1ce, T2 ve FLAIR otomatik eşleştirilecek.';
      if (selected.length === 1) return 'Tek MR dosyası seçildi. Analiz yalnız seçilen dosyalarla yapılır: aynı hastanın T1, T1ce, T2 ve FLAIR dosyalarını birlikte seçin (ya da tek ZIP).';
      const matched = ['t1', 't1ce', 't2', 'flair'].filter(key => names.some(name => name.includes('_' + key + '.') || name.includes('_' + key + '_') || name.endsWith('_' + key + '.nii.gz')));
      return matched.length === 4 ? `Dört modalite seçildi: ${matched.join(', ').toUpperCase()}` : `${selected.length} dosya seçildi. Eksik modalite varsa analiz başlamaz.`;
    }

    function resetAnalysisState(processing = false) {
      patientId = null;
      totalSlices = 0;
      currentSlice = 0;
      if (!processing) {
        isCurrentSelectionAnalyzed = false;
        lastAnalyzedSignature = null;
        if ($('start')) $('start').textContent = 'Analizi Başlat →';
      }
      
      // Teşhis Panellerini Sıfırla
      for (const [source] of sourceInfo) {
        const img = $(source);
        const ph = $(source + 'Placeholder');
        if (img) {
          img.hidden = true;
          img.src = '';
        }
        if (ph) {
          ph.hidden = false;
          /* İşleme sırasında düz metin yerine sakin iskelet ışıltısı (hareket azaltma tercihinde durağan, CSS) */
          ph.classList.toggle('loading', processing);
          ph.innerHTML = processing ? '<span class="sr-only">Yeni analiz işleniyor</span>' : '<span>Çıkarım<br>Bekleniyor</span>';
        }
      }

      // Metrikleri Sıfırla
      renderMetrics(null);

      // Z-Kesit Kaydırıcıyı Sıfırla
      if ($('slider')) {
        $('slider').value = 0;
        $('slider').max = 0;
        $('slider').disabled = true;
      }
      if ($('sliceValue')) $('sliceValue').textContent = 'Kesit: —';
      if ($('sliceCaption')) $('sliceCaption').textContent = 'İnceleme';

      // Rapor ve 3B Butonlarını Sıfırla
      if ($('report')) $('report').style.display = 'none';
      if ($('maskLinks')) $('maskLinks').innerHTML = '';
      if ($('consensusSummary')) { $('consensusSummary').hidden = true; $('consensusSummary').textContent = ''; }
      if ($('reportButton')) $('reportButton').disabled = true;
      if ($('showMesh')) $('showMesh').disabled = true;
      if ($('meshWrap')) $('meshWrap').classList.remove('show');
      
      // 3B Sahne & Klinik Kesiti Sıfırla
      sceneObjects = [];
      if ($('meshToggles')) $('meshToggles').innerHTML = '';
      const clinImg = $('clinicalSlice');
      if (clinImg) clinImg.src = '';
      const clinIdx = $('clinicalSliceIndex');
      if (clinIdx) clinIdx.textContent = 'Kesit —';
      const clinSlider = $('clinicalSliceSlider');
      if (clinSlider) { clinSlider.disabled = true; clinSlider.value = 0; clinSlider.max = 0; }
      const clinVal = $('clinicalSliceValue');
      if (clinVal) clinVal.textContent = 'Kesit —';
      
      if (gl && program) {
        gl.clearColor(0.012, 0.027, 0.07, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      }

      localStorage.removeItem('neurovista-last-result');
    }

    $('pickFiles').onclick = () => $('files').click();
    $('files').onchange = event => {
      const newFiles = [...event.target.files];
      if (!newFiles.length) return;

      const newSig = getFileSignature(newFiles);
      if (newSig && newSig === lastAnalyzedSignature && patientId) {
        selected = newFiles;
        isCurrentSelectionAnalyzed = true;
        $('start').disabled = true;
        $('start').textContent = 'Analiz Zaten Yapıldı';
        if (window.Toast) {
          window.Toast.info('Aynı Dosyalar Seçildi', 'Seçtiğiniz dosyalar mevcut analizle aynıdır. Yeniden analiz edilmesine gerek yoktur.');
        }
        status('Seçilen dosyalar zaten analiz edildi. Sonuçlar ekranda incelenebilir.');
        return;
      }

      selected = newFiles;
      isCurrentSelectionAnalyzed = false;
      resetAnalysisState(false);
      $('patient').textContent = classify();
      $('filesInfo').textContent = selected.map(file => file.name).join(' · ');
      $('filesInfo').style.display = selected.length ? 'block' : 'none';
      $('start').disabled = !canStart();
      $('start').textContent = 'Analizi Başlat →';
      $('reviewState').textContent = selected.length ? 'Dosyalar analize hazır' : 'Hasta seçimi bekleniyor';
    };

    $('modality').onclick = event => {
      if (event.target.dataset.mod === undefined) return;
      currentModality = +event.target.dataset.mod;
      [...$('modality').children].forEach(item => item.classList.toggle('active', +item.dataset.mod === currentModality));
      refreshPanels();
    };

    let sliceRefreshTimer;
    function updateSlice(value, immediate = false) {
      currentSlice = +value;
      $('sliceValue').textContent = `Kesit: ${currentSlice + 1} / ${totalSlices}`;
      clearTimeout(sliceRefreshTimer);
      if (immediate) refreshPanels();
      else sliceRefreshTimer = setTimeout(refreshPanels, 70);
    }
    $('slider').oninput = event => updateSlice(event.target.value);
    $('slider').onchange = event => updateSlice(event.target.value, true);

    function refreshPanels() {
      if (!patientId) return;
      for (const [source] of sourceInfo) {
        const image = $(source);
        image.src = `/api/v1/slice?patient_id=${encodeURIComponent(patientId)}&source=${source}&slice_idx=${currentSlice}&modality=${currentModality}`;
        image.hidden = false;
        $(source + 'Placeholder').hidden = true;
      }
    }

    function setLoading(loading) {
      $('files').disabled = loading;
      $('pickFiles').disabled = loading;
      $('start').disabled = loading || !canStart() || isCurrentSelectionAnalyzed;
      $('reportButton').disabled = loading || !patientId;
      $('showMesh').disabled = loading || !patientId;
    }

    function useResult(data) {
      patientId = data.patient_id;
      totalSlices = data.slices;
      currentSlice = Math.floor(totalSlices / 2);
      isCurrentSelectionAnalyzed = true;
      lastAnalyzedSignature = getFileSignature(selected);
      $('start').disabled = true;
      $('start').textContent = 'Analiz Tamamlandı';
      $('slider').max = totalSlices - 1;
      $('slider').value = currentSlice;
      $('slider').disabled = false;
      $('sliceValue').textContent = `Kesit: ${currentSlice + 1} / ${totalSlices}`;
      $('sliceCaption').textContent = `${totalSlices} kesit`;
      modelKeys = data.model_order;
      hasEnsemble = !!data.ensemble;
      sourceInfo = buildSourceInfo(modelKeys, hasEnsemble, Object.values(data.models).some(m => m.dice));
      makePanels();
      renderMetrics(data);
      const agr = data.agreement;
      if (agr && $('consensusSummary')) {
        const v = agr.consensus_voxels, n = agr.n_models, fmt = x => x.toLocaleString('tr-TR');
        $('consensusSummary').hidden = false;
        $('consensusSummary').innerHTML = `<span class="agreement-badge" title="${escapeHtml(agr.note)}">Uyum ${agr.score.toFixed(3)}</span>` +
          `<b>Model konsensüsü (${n} model, araştırma görünümü):</b> ` +
          `<span class="all-key">camgöbeği</span> ${n}/${n} model tümör diyor <b>${fmt(v.all_models)}</b> voksel · ` +
          (n > 2 ? `<span class="majority-key">mor</span> çoğunluk <b>${fmt(v.majority_only)}</b> voksel · ` : '') +
          `<span class="minority-key">fuşya</span> yalnız bir model <b>${fmt(v.minority)}</b> voksel. ` +
          `Uyum skoru, modeller arası ortalama Dice'tır; hata göstergesi olarak geçerliliği henüz doğrulanmadı. Bu panel maskeleri değiştirmez.`;
      }
      $('patient').innerHTML = `<strong>Analiz Tamamlandı</strong><br>${escapeHtml(data.patient_name || data.modalities.join(', ').toUpperCase())}`;
      $('reviewState').textContent = 'Çıktılar incelemeye hazır';
      $('report').href = `/api/v1/export-report?patient_id=${patientId}&slice_idx=${currentSlice}&modality=${currentModality}&format=png`;
      $('report').style.display = 'inline-block';
      const maskSources = [...(hasEnsemble ? ['ensemble'] : []), ...byRank(modelKeys)];
      $('maskLinks').innerHTML = maskSources.map(source =>
        `<a class="mask-export" download style="display:inline-block" href="/api/v1/export-mask?patient_id=${patientId}&source=${source}" ` +
        `title="${sourceLabel(source)} maskesi, orijinal MR geometrisinde (ET = 4). 3D Slicer / ITK-SNAP ile açılabilir.">${icon('download')}${(MODEL_META[source] || { name: source }).name} maskesi (.nii.gz)</a>`).join('');
      setMeshSources([...maskSources, ...(sourceInfo.some(([s]) => s === 'gt') ? ['gt'] : [])], data.primary_source);
      $('reportButton').disabled = false;
      $('showMesh').disabled = false;
      refreshPanels();
      status(`Çıkarım tamamlandı (${data.total_time_sec} s) · ${runtimeLabel(data)}${hasEnsemble ? ' · ana maske: kilitli ensemble' : ''}.`);
    }

    /* Gerçekte kullanılan çalışma yolunu API sonucundan okur (sabit metin yerine). */
    function runtimeLabel(data) {
      const names = { cuda: 'GPU FP16', cpu_fallback: 'CPU yedek yolu', openvino: 'OpenVINO CPU', pytorch: 'PyTorch CPU' };
      if (!data || !data.models) return 'yerel çalışma zamanı';
      return Object.entries(data.models).map(([key, m]) => `${m.name}: ${names[m.runtime] || m.runtime}`).join(' · ');
    }

    async function waitForJob(jobId) {
      let networkFailures = 0;   // CPU tam yükteyken tek tük kopan sorgu analizi düşürmesin
      while (true) {
        await new Promise(resolve => setTimeout(resolve, 800));
        let response;
        try {
          response = await fetch(`/api/v1/jobs/${jobId}`);
          networkFailures = 0;
        } catch (netErr) {
          if (++networkFailures < 5) continue;
          throw { title: 'Bağlantı Hatası', detail: 'Sunucuya ulaşılamadı. Yerel sunucunun açık olduğundan emin olunuz.', guidance: 'Terminalde run.py çalışıyor mu kontrol ediniz.' };
        }
        if (response.status === 404) {
          throw { title: 'İşlem Bulunamadı (404)', detail: 'Bu analize ait oturum zaman aşımına uğradı veya bulunamadı.', guidance: 'Lütfen dosyaları yeniden seçip analizi tekrar başlatınız.' };
        }
        const job = await response.json();
        if (!response.ok) {
          throw { title: job.title || 'Analiz Durum Hatası', detail: job.detail || 'İşlem sorgulanamadı.', guidance: job.guidance || 'Sunucu günlüğünü kontrol ediniz.' };
        }
        const msg = job.message || 'Sinir ağı çıkarımı çalışıyor...';
        status(msg);
        if (window.HUD) window.HUD.update(msg);
        $('reviewState').textContent = msg;

        if (job.status === 'completed') return job.result;
        if (job.status === 'failed') {
          throw { title: 'Model Çıkarım Hatası', detail: job.message || 'Derin öğrenme modelleri segmentasyon üretirken bir hata verdi.', guidance: 'MR görüntülerinin boyutlarının birbiriyle uyumlu ve bozulmamış olduğunu teyit ediniz.' };
        }
      }
    }

    async function requestAnalysis(url, body) {
      setLoading(true);
      resetAnalysisState(true);
      if (window.HUD) window.HUD.start('Hasta MR dosyaları yükleniyor ve doğrulanıyor...');
      if (window.Toast) window.Toast.info('Analiz Başlatıldı', 'MR dosyaları işleme alındı, sinir ağları hazırlanıyor.');
      try {
        const response = await fetch(url, { method: 'POST', body });
        if (!response.ok) {
          let errData = {};
          try { errData = await response.json(); } catch {}
          const title = errData.title || `Hata (${response.status})`;
          const detail = errData.detail || 'Sunucu ile iletişim kurulamadı.';
          const guidance = errData.guidance || 'Lütfen dosyalarınızı ve bağlantınızı kontrol edip tekrar deneyiniz.';
          throw { status: response.status, title, detail, guidance };
        }
        const data = await response.json();
        useResult(await waitForJob(data.job_id));
        if (window.HUD) window.HUD.stop(true);
        if (window.Toast) window.Toast.success('Analiz Başarıyla Tamamlandı', 'Modellerin ve ensemble\'ın segmentasyon sonuçları incelemeye hazır.');
      } catch (err) {
        if (window.HUD) window.HUD.stop(false);
        isCurrentSelectionAnalyzed = false;
        $('start').disabled = false;
        $('start').textContent = 'Analizi Tekrar Dene →';
        const title = err.title || 'Analiz Hatası';
        const detail = err.detail || err.message || 'Beklenmeyen bir hata oluştu.';
        const guidance = err.guidance || 'Lütfen MR dosyalarının T1, T1ce, T2 ve FLAIR formatında olduğunu teyit ediniz.';
        if (window.Toast) window.Toast.error(title, detail);
        if (window.showErrorDialog) window.showErrorDialog(title, detail, guidance);
        status(detail, true);
        $('reviewState').textContent = 'Analiz tamamlanamadı';
      } finally {
        setLoading(false);
      }
    }

    $('start').onclick = () => {
      if (!selected.length) {
        status('Önce MR dosyalarını seçin.', true);
        return;
      }
      const currentSig = getFileSignature(selected);
      if (isCurrentSelectionAnalyzed || (currentSig && currentSig === lastAnalyzedSignature)) {
        $('start').disabled = true;
        $('start').textContent = 'Analiz Tamamlandı';
        if (window.Toast) {
          window.Toast.info('Analiz Zaten Tamamlandı', 'Seçilen MR dosyaları zaten analiz edildi. Yeni analiz için farklı hasta dosyaları seçiniz.');
        }
        status('Bu hasta dosyaları zaten analiz edildi. Yeni analiz için farklı dosya seçiniz.');
        return;
      }
      const form = new FormData();
      selected.forEach(file => form.append('files', file));
      requestAnalysis('/api/v1/predict', form);
    };

    $('reportButton').onclick = () => {
      if (patientId) window.open($('report').href, '_blank');
    };

    if (location.protocol === 'file:') {
      status('Bu sayfa dosya olarak açılmış. Çalışması için http://127.0.0.1:8000/ adresini kullanın.', true);
      $('files').disabled = true;
      $('pickFiles').disabled = true;
      $('start').disabled = true;
    }

    /* 3B WebGL Render Engine */
    const canvas = $('mesh');
    let sceneObjects = [], rotX = 0.55, rotY = 0.65, zoom = 2.05, drag = null, gl = null, program = null;

    function shader(type, source) {
      const value = gl.createShader(type);
      gl.shaderSource(value, source);
      gl.compileShader(value);
      if (!gl.getShaderParameter(value, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(value));
      return value;
    }

    function initWebGL() {
      if (gl) return;
      gl = canvas.getContext('webgl2', { antialias: true, alpha: true }) || canvas.getContext('webgl', { antialias: true, alpha: true });
      if (!gl) throw Error('Tarayıcı WebGL desteği vermiyor');
      const vertex = `
        attribute vec3 aPosition;
        attribute vec3 aNormal;
        uniform mat3 uRotation;
        uniform float uZoom;
        uniform float uAspect;
        varying vec3 vNormal;
        varying vec3 vView;
        void main(){
          vec3 p = uRotation * aPosition;
          vNormal = normalize(uRotation * aNormal);
          vView = normalize(vec3(-p.x, -p.y, 2.9 - p.z * 0.55));
          float depth = 2.9 - p.z * 0.55;
          gl_Position = vec4(p.x * uZoom / (depth * uAspect), p.y * uZoom / depth, 0.0, 1.0);
        }
      `;
      const fragment = `
        precision mediump float;
        uniform vec4 uColor;
        uniform float uBrain;
        varying vec3 vNormal;
        varying vec3 vView;
        void main(){
          vec3 n = normalize(vNormal), v = normalize(vView), light = normalize(vec3(0.34, 0.58, 1.0));
          float diffuse = max(dot(n, light), 0.0);
          float specular = pow(max(dot(reflect(-light, n), v), 0.0), 28.0);
          float rim = pow(1.0 - max(dot(n, v), 0.0), 2.4);
          vec3 shaded = uColor.rgb * (0.32 + 0.70 * diffuse);
          shaded += vec3(1.0) * (specular * (uBrain > 0.5 ? 0.10 : 0.16));
          shaded += uColor.rgb * rim * (uBrain > 0.5 ? 0.20 : 0.045);
          gl_FragColor = vec4(min(shaded, vec3(1.0)), uColor.a);
        }
      `;
      program = gl.createProgram();
      gl.attachShader(program, shader(gl.VERTEX_SHADER, vertex));
      gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fragment));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(program));
      gl.useProgram(program);
      gl.enable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    }

    function rgb(hex) {
      const value = parseInt(hex.slice(1), 16);
      return [((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255];
    }

    function prepareBuffersCentered() {
      const all = sceneObjects.flatMap(item => item.vertices), low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
      all.forEach(v => v.forEach((x, i) => { low[i] = Math.min(low[i], x); high[i] = Math.max(high[i], x); }));
      const center = low.map((x, i) => (x + high[i]) / 2), extent = Math.max(...high.map((x, i) => x - low[i])) / 2 || 1;
      sceneObjects.forEach(item => {
        const positions = new Float32Array(item.vertices.flatMap(v => [(v[1] - center[1]) / extent, (v[0] - center[0]) / extent, (v[2] - center[2]) / extent])), normals = new Float32Array(positions.length);
        item.faces.forEach(([a, b, c]) => {
          const ax = positions[a * 3], ay = positions[a * 3 + 1], az = positions[a * 3 + 2];
          const bx = positions[b * 3], by = positions[b * 3 + 1], bz = positions[b * 3 + 2];
          const cx = positions[c * 3], cy = positions[c * 3 + 1], cz = positions[c * 3 + 2];
          const ux = bx - ax, uy = by - ay, uz = bz - az, vx = cx - ax, vy = cy - ay, vz = cz - az;
          const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
          [a, b, c].forEach(i => { normals[i * 3] += nx; normals[i * 3 + 1] += ny; normals[i * 3 + 2] += nz; });
        });
        for (let i = 0; i < normals.length; i += 3) {
          const d = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
          normals[i] /= d; normals[i + 1] /= d; normals[i + 2] /= d;
        }
        item.positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, item.positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
        item.normalBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, item.normalBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
        item.indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, item.indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(item.faces.flat()), gl.STATIC_DRAW);
        item.indexCount = item.faces.length * 3;
      });
    }

    /* Arcball Quaternion Math */
    let orbit = [0, 0, 0, 1], orbitAnchor = orbit.slice(), orbitPoint = null, renderQueued = false;
    const qNormalize = q => { const d = Math.hypot(q[0], q[1], q[2], q[3]) || 1; return q.map(v => v / d); };
    const qMultiply = (a, b) => [a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1], a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0], a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3], a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]];
    const qAxis = (axis, angle) => { const s = Math.sin(angle / 2); return [axis[0]*s, axis[1]*s, axis[2]*s, Math.cos(angle / 2)]; };
    const qMatrix = q => { const [x,y,z,w] = qNormalize(q), xx=x*x, yy=y*y, zz=z*z, xy=x*y, xz=x*z, yz=y*z, wx=w*x, wy=w*y, wz=w*z; return new Float32Array([1-2*(yy+zz),2*(xy+wz),2*(xz-wy),2*(xy-wz),1-2*(xx+zz),2*(yz+wx),2*(xz+wy),2*(yz-wx),1-2*(xx+yy)]); };
    const resetOrbit = () => { orbit = [0, 0, 0, 1]; };
    const spherePoint = event => { const rect = canvas.getBoundingClientRect(), aspect = rect.width / Math.max(rect.height, 1); let x = ((event.clientX - rect.left) / rect.width * 2 - 1) * aspect, y = 1 - (event.clientY - rect.top) / rect.height * 2, z = 0; const r2 = x*x+y*y; if (r2 <= 1) z = Math.sqrt(1-r2); else { const d = Math.sqrt(r2); x /= d; y /= d; } return [x,y,z]; };
    const qBetween = (a,b) => { const cross=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]], dot=a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; if(dot < -.9999){const axis=Math.abs(a[0])<.8?[1,0,0]:[0,1,0]; return qAxis([a[1]*axis[2]-a[2]*axis[1],a[2]*axis[0]-a[0]*axis[2],a[0]*axis[1]-a[1]*axis[0]],Math.PI);} return qNormalize([cross[0],cross[1],cross[2],1+dot]); };
    const scheduleRender = () => { if (!renderQueued) { renderQueued = true; requestAnimationFrame(() => { renderQueued = false; drawMesh(); }); } };

    function drawMesh() {
      if (!gl || !program) return;
      const width = canvas.width = Math.max(1, canvas.clientWidth * devicePixelRatio);
      const height = canvas.height = Math.max(1, canvas.clientHeight * devicePixelRatio);
      gl.viewport(0, 0, width, height);

      // 3B görüntüleme alanı iki temada da koyu: yarı saydam beyin yüzeyi açık zeminde kayboluyordu (2B MR panelleri gibi).
      gl.clearColor(0.012, 0.027, 0.07, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(program);

      const position = gl.getAttribLocation(program, 'aPosition');
      const normal = gl.getAttribLocation(program, 'aNormal');
      gl.uniformMatrix3fv(gl.getUniformLocation(program, 'uRotation'), false, qMatrix(orbit));
      gl.uniform1f(gl.getUniformLocation(program, 'uZoom'), zoom);
      gl.uniform1f(gl.getUniformLocation(program, 'uAspect'), width / height);

      sceneObjects.filter(item => item.visible && item.indexCount).sort((a, b) => a.opacity - b.opacity).forEach(item => {
        gl.bindBuffer(gl.ARRAY_BUFFER, item.positionBuffer);
        gl.enableVertexAttribArray(position);
        gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, item.normalBuffer);
        gl.enableVertexAttribArray(normal);
        gl.vertexAttribPointer(normal, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, item.indexBuffer);
        const color = rgb(item.color);
        gl.uniform4f(gl.getUniformLocation(program, 'uColor'), color[0], color[1], color[2], item.opacity);
        gl.uniform1f(gl.getUniformLocation(program, 'uBrain'), item.key === 'brain' ? 1 : 0);
        gl.depthMask(item.opacity >= 0.95);
        gl.drawElements(gl.TRIANGLES, item.indexCount, gl.UNSIGNED_SHORT, 0);
      });
      gl.depthMask(true);
    }

    /* Görünüm ön ayarları. Sahnede yatay eksen ön-arka, dikey eksen sağ-sol, derinlik üst-alt (BraTS 3. eksen) olduğundan
       döndürülmemiş sahne 2B aksiyel kesitle hizalıdır; yatay eksen etrafında 90° döndürme yandan (sagittal) bakış verir. */
    const VIEWS = {
      axial: { button: 'viewAxial', orbit: [0, 0, 0, 1], title: 'AKSİYEL GÖRÜNÜM', note: '2B kesitle hizalı · <em>tümör konumu eşleşir</em>' },
      iso: { button: 'viewIso', orbit: [0.35, 0.35, -0.15, 0.85], title: 'İZOMETRİK GÖRÜNÜM', note: 'Eğik bakış · <em>derinlik ilişkisi</em>' },
      sagittal: { button: 'viewSagittal', orbit: [0.707, 0, 0, 0.707], title: 'SAGİTTAL GÖRÜNÜM', note: 'Yandan bakış · <em>üst-alt yayılım</em>' },
    };
    function setViewLabel(name) {
      const view = VIEWS[name];
      Object.values(VIEWS).forEach(item => $(item.button).classList.toggle('active', item === view));
      $('viewTitle').textContent = view ? view.title : 'SERBEST GÖRÜNÜM';
      $('viewNote').innerHTML = view ? view.note : 'Fareyle döndürüldü · <em>çift tıkla aksiyele dön</em>';
    }
    function applyView(name) {
      orbit = VIEWS[name].orbit.slice();
      zoom = 2.08;
      setViewLabel(name);
      scheduleRender();
    }
    Object.keys(VIEWS).forEach(name => { $(VIEWS[name].button).onclick = () => applyView(name); });

    const sceneClientCache = new Map();
    async function loadMesh() {
      if (!patientId || $('loadMesh').dataset.loading === 'true') return;
      const source = $('meshSource').value, sceneKey = `${patientId}:${source}`, loadButton = $('loadMesh'), sourceSelect = $('meshSource');
      loadButton.dataset.loading = 'true'; loadButton.disabled = true; sourceSelect.disabled = true;
      status('Gerçek MR yüzeyleri hazırlanıyor…');
      try {
        initWebGL();
        const cached = sceneClientCache.get(sceneKey);
        if (cached) {
          sceneObjects = cached;
          sceneObjects.forEach(item => { item.visible = true; });
        } else {
          const response = await fetch(`/api/v1/scene?patient_id=${encodeURIComponent(patientId)}&source=${source}`);
          const data = await response.json();
          if (!response.ok) throw Error(data.detail || '3B sahne oluşturulamadı');
          sceneObjects = Object.entries(data.layers).filter(([, layer]) => layer.mesh).map(([key, layer]) => ({ key, ...layer, vertices: layer.mesh.vertices, faces: layer.mesh.faces, visible: true }));
          prepareBuffersCentered();
          sceneClientCache.set(sceneKey, sceneObjects);
        }
        resetOrbit(); zoom = 2.08; setViewLabel('axial');
        $('meshToggles').innerHTML = sceneObjects.map(item => `<label><input type="checkbox" data-layer="${item.key}" checked> <span class="swatch" style="background:${item.color}"></span> ${item.label}</label>`).join('');
        $('meshToggles').onchange = event => {
          const item = sceneObjects.find(value => value.key === event.target.dataset.layer);
          if (item) { item.visible = event.target.checked; scheduleRender(); }
        };
        if (window.clinicalPreview) window.clinicalPreview();
        drawMesh();
        status(`3B sahne hazır: ${sceneObjects.length} gerçek veri katmanı. Sürükleyerek döndürebilirsiniz.`);
      } catch (error) {
        status(error.message, true);
      } finally {
        loadButton.dataset.loading = 'false'; loadButton.disabled = false; sourceSelect.disabled = false;
      }
    }
    $('loadMesh').onclick = loadMesh;
    $('showMesh').onclick = () => { $('meshWrap').classList.add('show'); loadMesh(); };
    $('meshSource').onchange = () => { if (patientId) { $('meshWrap').classList.add('show'); loadMesh(); } };

    canvas.onpointerdown = event => { orbitPoint = spherePoint(event); orbitAnchor = orbit.slice(); canvas.setPointerCapture(event.pointerId); };
    canvas.onpointermove = event => {
      if (!orbitPoint) return;
      orbit = qNormalize(qMultiply(qBetween(orbitPoint, spherePoint(event)), orbitAnchor));
      if (!$('viewTitle').textContent.startsWith('SERBEST')) setViewLabel(null);
      scheduleRender();
    };
    canvas.onpointerup = event => { orbitPoint = null; if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId); };
    canvas.onpointercancel = () => { orbitPoint = null; };
    canvas.onpointerleave = event => { if (!event.buttons) orbitPoint = null; };
    canvas.ondblclick = () => applyView('axial');
    canvas.onwheel = event => { event.preventDefault(); zoom = Math.max(0.38, Math.min(5.8, zoom * (event.deltaY > 0 ? 0.90 : 1.11))); scheduleRender(); };

    /* 2D Clinical Slice Integration */
    {
      const frame = canvas.parentNode;
      const layout = document.createElement('div'); layout.className = 'mesh-layout';
      const clinical = document.createElement('aside'); clinical.className = 'clinical-preview';
      clinical.innerHTML = `
        <header>
          <div>
            <div class="eyebrow">Klinik MR kesiti</div>
            <h3>MR + Segmentasyon Bindirmesi</h3>
          </div>
          <p>Gerçek FLAIR<br>ve seçili model</p>
        </header>
        <img id="clinicalSlice" alt="Gerçek MR kesiti ve tümör bindirmesi">
        <div class="clinical-slice-control">
          <div><span>Z-Ekseni Kesiti</span><b id="clinicalSliceValue">Kesit —</b></div>
          <input id="clinicalSliceSlider" type="range" min="0" max="0" value="0" disabled aria-label="Klinik MR kesiti için Z ekseni">
        </div>
        <footer><b id="clinicalSliceIndex">Kesit —</b> · Renkler: yeşil NCR, sarı ödem, kırmızı aktif tümör</footer>
      `;
      frame.before(layout);
      layout.append(clinical, frame);

      window.clinicalPreview = () => {
        const image = $('clinicalSlice');
        if (!image || !patientId) return;
        image.src = `/api/v1/slice?patient_id=${encodeURIComponent(patientId)}&source=${$('meshSource').value}&slice_idx=${currentSlice}&modality=${currentModality}`;
        const label = $('clinicalSliceIndex');
        if (label) label.textContent = `Kesit ${currentSlice + 1} / ${totalSlices}`;
        const modelLabel = document.querySelector('.clinical-preview header p');
        if (modelLabel) modelLabel.innerHTML = `Gerçek FLAIR<br><b>${sourceLabel($('meshSource').value)}</b>`;
        const localSlider = $('clinicalSliceSlider'), localValue = $('clinicalSliceValue');
        if (localSlider) { localSlider.max = Math.max(0, totalSlices - 1); localSlider.value = currentSlice; localSlider.disabled = false; }
        if (localValue) localValue.textContent = `Kesit ${currentSlice + 1} / ${totalSlices}`;
      };

      const localSlider = $('clinicalSliceSlider');
      if (localSlider) {
        localSlider.oninput = event => updateSlice(event.target.value);
        localSlider.onchange = event => updateSlice(event.target.value, true);
      }

      const priorRefreshPanels = refreshPanels;
      refreshPanels = () => { priorRefreshPanels(); window.clinicalPreview(); };
      const priorUpdateSlice = updateSlice;
      updateSlice = (value, immediate = false) => {
        priorUpdateSlice(value, immediate);
        const ls = $('clinicalSliceSlider'), lv = $('clinicalSliceValue');
        if (ls) ls.value = currentSlice;
        if (lv) lv.textContent = `Kesit ${currentSlice + 1} / ${totalSlices}`;
      };
      $('meshSource').addEventListener('change', window.clinicalPreview);
    }

    /* Toast Notification Helper */
    window.Toast = {
      show(type, title, message, timeout = 5000) {
        const icons = { success: icon('check'), error: icon('alert-circle'), info: icon('info'), warning: icon('alert') };
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
          <div class="toast-icon">${icons[type] || icons.info}</div>
          <div class="toast-body">
            <strong>${escapeHtml(title)}</strong>
            <p>${escapeHtml(message)}</p>
          </div>
          <button class="toast-close" type="button" aria-label="Kapat">×</button>
        `;
        toast.querySelector('.toast-close').onclick = () => toast.remove();
        $('toastHost').append(toast);
        if (timeout > 0) {
          setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(-10px)';
            setTimeout(() => toast.remove(), 250);
          }, timeout);
        }
      },
      success(title, message) { this.show('success', title, message, 5000); },
      error(title, message) { this.show('error', title, message, 7000); },
      info(title, message) { this.show('info', title, message, 4000); },
      warning(title, message) { this.show('warning', title, message, 6000); }
    };

    /* Activity HUD Process Manager */
    window.HUD = {
      start(initialMessage = 'Analiz başlatılıyor...') {
        $('activityHUD').classList.add('active');
        $('hudMessage').textContent = initialMessage;
      },
      update(message) {
        if (!$('activityHUD').classList.contains('active')) this.start(message);
        $('hudMessage').textContent = message;
      },
      stop(success = true) {
        if (success) {
          $('hudMessage').textContent = 'Çıkarım başarıyla tamamlandı';
          setTimeout(() => $('activityHUD').classList.remove('active'), 2200);
        } else {
          $('activityHUD').classList.remove('active');
        }
      }
    };

    window.showErrorDialog = (title, detail, guidance) => {
      $('modalTitle').textContent = title || 'Analiz Hatası';
      $('modalDesc').textContent = detail || 'İşlem sırasında bir hata meydana geldi.';
      $('modalGuide').textContent = guidance || 'T1, T1ce, T2 ve FLAIR dosyalarının eksiksiz ve geçerli NIfTI (.nii.gz) formatında olduğunu kontrol ediniz.';
      $('errorModal').classList.add('active');
    };
    $('modalDismiss').onclick = () => $('errorModal').classList.remove('active');
    $('modalRetry').onclick = () => {
      $('errorModal').classList.remove('active');
      $('files').click();
    };

    /* Benchmark Modal Controls */
    const openBenchmark = () => { if ($('benchmarkModal')) $('benchmarkModal').classList.add('active'); };
    const closeBenchmark = () => { if ($('benchmarkModal')) $('benchmarkModal').classList.remove('active'); };
    if ($('btnShowBenchmark')) $('btnShowBenchmark').onclick = openBenchmark;
    if ($('topbarBenchmarkBtn')) $('topbarBenchmarkBtn').onclick = openBenchmark;
    if ($('benchmarkDismiss')) $('benchmarkDismiss').onclick = closeBenchmark;
    if ($('benchmarkModal')) {
      $('benchmarkModal').onclick = (e) => { if (e.target === $('benchmarkModal')) closeBenchmark(); };
    }
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if ($('benchmarkModal')) $('benchmarkModal').classList.remove('active');
        if ($('errorModal')) $('errorModal').classList.remove('active');
      }
    });

    /* Theme Management */
    const themeBtn = $('themeToggleBtn');
    function applyTheme(theme) {
      if (theme === 'light') {
        document.body.classList.add('light-theme', 'light-mode');
        themeBtn.innerHTML = icon('moon');
        localStorage.setItem('neurovista-theme', 'light');
      } else {
        document.body.classList.remove('light-theme', 'light-mode');
        themeBtn.innerHTML = icon('sun');
        localStorage.setItem('neurovista-theme', 'dark');
      }
      scheduleRender();
    }
    const savedTheme = localStorage.getItem('neurovista-theme') || 'dark';
    applyTheme(savedTheme);
    themeBtn.onclick = () => {
      const isCurrentlyLight = document.body.classList.contains('light-theme');
      applyTheme(isCurrentlyLight ? 'dark' : 'light');
    };

    /* Sayfa Yenileme Yönetimi: Her zaman en üste ve sıfır duruma odaklan */
    if (history.scrollRestoration) history.scrollRestoration = 'manual';
    window.scrollTo(0, 0);
    localStorage.removeItem('neurovista-last-result');
    /* Donanım Rozeti: Açılışta /api/v1/health'den GPU/CPU durumunu çeker */
    (async () => {
      try {
        const res = await fetch('/api/v1/health');
        if (!res.ok) return;
        const h = await res.json();
        const badge = $('hwBadge');
        if (!badge) return;
        if (!patientId && Array.isArray(h.models)) {
          modelKeys = h.models.map(m => m.key);
          hasEnsemble = !!h.ensemble;
          sourceInfo = buildSourceInfo(modelKeys, hasEnsemble);
          makePanels();
          renderMetrics(null);
          setMeshSources([...(hasEnsemble ? ['ensemble'] : []), ...byRank(modelKeys), 'gt'], hasEnsemble ? 'ensemble' : 'swin');
        }
        const setLabel = `${(h.models || []).length} model${h.ensemble ? ' + ensemble' : ''}`;
        if ($('engineLabel')) $('engineLabel').textContent = `${setLabel} · çevrimiçi`;
        if (h.device === 'cuda' && h.device_info && h.device_info.gpu_name) {
          badge.innerHTML = '<i style="background:#10b981"></i>GPU Aktif (' + h.device_info.gpu_name + ' · FP16 · ' + setLabel + ')';
          badge.title = 'CUDA ' + (h.device_info.cuda_version || '') + ' · VRAM: ' + h.device_info.gpu_vram_mb + ' MB · Overlap: ' + h.overlap;
        } else {
          const ov = (h.models || []).filter(m => m.runtime === 'openvino').length;
          badge.innerHTML = '<i style="background:#06b6d4"></i>CPU Modu (' + (ov ? `OpenVINO ${ov}/${(h.models || []).length}` : 'PyTorch CPU') + ' · ' + setLabel + ')';
          badge.title = 'GPU kullanılmıyor · Overlap: ' + h.overlap;
        }
      } catch { /* sunucu henüz hazır değilse sessiz geç */ }
    })();
