"""Web sayfalarının yapısal tutarlılığı (tarayıcısız, statik denetim).

- Betiklerin ``$('...')`` / ``getElementById('...')`` ile başvurduğu her öğe sayfada (ya da betiğin ürettiği
  işaretlemede) vardır; yoksa düğme sessizce çalışmaz ya da betik ``null`` hatasıyla durur.
- 2B paneller BraTS hacminin 3. ekseninden (Z) alınır: kesit aksiyeldir, "koronal" diye etiketlenmez.
"""
import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "backend" / "app" / "static"
PAGES = {"index.html": "js/analysis.js", "landing.html": "js/landing.js"}


def element_ids(*texts: str) -> set[str]:
    return {i for text in texts for i in re.findall(r'id="([A-Za-z0-9_-]+)"', text)}


def referenced_ids(script: str) -> set[str]:
    return set(re.findall(r"""(?:\$|getElementById)\(\s*['"]([A-Za-z0-9_-]+)['"]\s*\)""", script))


@pytest.mark.parametrize("page,script", PAGES.items())
def test_script_element_references_exist(page, script):
    html = (STATIC / page).read_text(encoding="utf-8")
    js = (STATIC / script).read_text(encoding="utf-8")
    assert f'src="/{script}"' in html
    missing = referenced_ids(js) - element_ids(html, js)
    assert not missing, f"{script} sayfada olmayan öğelere başvuruyor: {sorted(missing)}"


def test_slice_panels_are_labelled_axial():
    text = (STATIC / "index.html").read_text(encoding="utf-8") + (STATIC / "js" / "analysis.js").read_text(encoding="utf-8")
    assert "Aksiyel" in text
    assert not re.search(r"[Kk]oronal|KORONAL", text)


@pytest.mark.parametrize("page", ["landing.html", "index.html", "404.html"])
def test_pages_declare_existing_favicon(page):
    # Simge yokken tarayıcı her açılışta /favicon.ico isteyip 404 alıyordu (konsol hatası).
    html = (STATIC / page).read_text(encoding="utf-8")
    match = re.search(r'<link rel="icon" href="/([^"]+)"', html)
    assert match and (STATIC / match.group(1)).is_file()


@pytest.mark.parametrize("page", ["landing.html", "index.html", "404.html"])
def test_local_style_assets_exist(page):
    # Arka plan dokusu gibi CSS içinden çağrılan yerel dosyalar (url("/...")) gerçekten sunulur.
    html = (STATIC / page).read_text(encoding="utf-8")
    sheets = [html] + [(STATIC / css).read_text(encoding="utf-8") for css in re.findall(r'href="/(css/[^"]+)"', html)]
    urls = {u for text in sheets for u in re.findall(r'url\("/([^"]+)"\)', text)}
    missing = [u for u in urls if not (STATIC / u).is_file()]
    assert not missing, f"{page}: eksik yerel dosya(lar) {missing}"


def test_landing_scene_has_fallback_and_reduced_motion():
    # Sahne yalnız süs: WebGL yoksa yedek ışıma görünür, hareket azaltma tercihine uyulur.
    html = (STATIC / "landing.html").read_text(encoding="utf-8")
    js = (STATIC / "js" / "neuro-scene.js").read_text(encoding="utf-8")
    css = (STATIC / "css" / "landing.css").read_text(encoding="utf-8")
    assert 'id="neuroScene"' in html and 'src="/js/neuro-scene.js"' in html
    assert "no-webgl" in js and ".no-webgl .scene-fallback" in css
    assert "prefers-reduced-motion" in js and "prefers-reduced-motion" in css
