from playwright.sync_api import sync_playwright
import pathlib
p=pathlib.Path("poster.html").resolve()
with sync_playwright() as pw:
    b=pw.chromium.launch(); pg=b.new_page(viewport={"width":3179,"height":4494})
    pg.goto(p.as_uri()); pg.wait_for_timeout(800)
    h=pg.evaluate("document.querySelector('.page').scrollHeight"); print("content px",h,"page px",4494)
    pg.pdf(path="poster_A0.pdf",width="841mm",height="1189mm",print_background=True,prefer_css_page_size=True)
    pg.set_viewport_size({"width":1060,"height":1499}); pg.evaluate("document.body.style.zoom=1060/3179")
    pg.screenshot(path="preview.png",full_page=False); b.close()
