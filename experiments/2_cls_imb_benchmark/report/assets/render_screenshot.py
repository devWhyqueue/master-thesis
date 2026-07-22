from playwright.sync_api import sync_playwright
import os

html_path = os.path.abspath('index.html')

with sync_playwright() as p:
    browser = p.chromium.launch()
    # device_scale_factor=3 captures Ultra-HD 3x resolution (300+ DPI print quality)
    page = browser.new_page(
        viewport={'width': 1620, 'height': 1080},
        device_scale_factor=3
    )
    page.goto('file:///' + html_path)
    page.wait_for_selector('#diagram')
    page.wait_for_timeout(300)
    
    diagram = page.query_selector('#diagram')
    diagram.screenshot(path='rendered_diagram.png')
    diagram.screenshot(path='cls_imb_benchmark.png')
    
    browser.close()

print("High-DPI 3x screenshot saved successfully.")
