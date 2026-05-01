"""E2E browser tests for the subtitle-generator web app using Playwright.

Supports both local and deployed modes via BASE_URL env var:
  python tests/test_e2e.py                          # local (localhost:8742)
  BASE_URL=https://subtitlegenst.z13.web.core.windows.net python tests/test_e2e.py
"""

import asyncio
import os
import time
from urllib.parse import urlparse
from playwright.async_api import async_playwright

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8742")
AI_TRACK_PATH = "/v2/track"
AI_INGEST_HOSTS = (
    "applicationinsights.azure.com",
    "dc.services.visualstudio.com",
)


def is_app_insights_track_url(url: str) -> bool:
    return AI_TRACK_PATH in url and any(host in url for host in AI_INGEST_HOSTS)


async def wait_for_telemetry(predicate, label: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.25)
    raise AssertionError(f"Timed out waiting for App Insights telemetry: {label}")


async def test():
    host = urlparse(BASE_URL).hostname
    is_local = host in {"localhost", "127.0.0.1", "::1"}
    print(f"Testing against: {BASE_URL} ({'local' if is_local else 'deployed'})\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        telemetry_posts: list[str] = []
        telemetry_statuses: list[int] = []

        def capture_telemetry_request(request):
            if request.method == "POST" and is_app_insights_track_url(request.url):
                telemetry_posts.append(request.post_data or "")

        def capture_telemetry_response(response):
            if is_app_insights_track_url(response.url):
                telemetry_statuses.append(response.status)

        page.on("request", capture_telemetry_request)
        page.on("response", capture_telemetry_response)

        # 1. Load page
        print("TEST 1: Load page")
        await page.goto(BASE_URL)
        title = await page.title()
        assert "Subtitle Generator" in title, f"Bad title: {title}"
        print(f"  PASS: title = {title}")

        # 2. Check mode badge appears
        print("TEST 2: Mode detection")
        await page.wait_for_function(
            "() => document.querySelector('.mode-badge')?.textContent?.includes('Mode')",
            timeout=15000,
        )
        badge_text = await page.locator(".mode-badge").text_content()
        expected_mode = "Local" if is_local else "Web"
        assert expected_mode in badge_text, f"Expected {expected_mode} mode, got: {badge_text}"
        print(f"  PASS: {badge_text}")

        if not is_local:
            print("TEST 2b: App Insights page view telemetry")
            await wait_for_telemetry(
                lambda: any("PageviewData" in post for post in telemetry_posts)
                and any(200 <= status < 300 for status in telemetry_statuses),
                "page view",
            )
            print("  PASS: page view telemetry posted")

        # 3. Click Generate (with cold-start retry)
        print("TEST 3: Generate subtitle")
        gen_btn = page.locator("button:has-text('Generate')").first
        for gen_attempt in range(3):
            await gen_btn.click()
            try:
                await page.wait_for_function(
                    "() => document.querySelectorAll('.slot').length >= 4",
                    timeout=60000,
                )
                break
            except Exception:
                if gen_attempt < 2:
                    print(f"  Cold start timeout, retrying ({gen_attempt + 1}/3)...")
                    # Dismiss any alert dialog
                    page.on("dialog", lambda d: d.dismiss())
                    await page.wait_for_timeout(5000)
                else:
                    raise
        slots = await page.locator(".slot").count()
        assert slots >= 4, f"Expected >= 4 slots, got {slots}"

        slot_texts = []
        for i in range(slots):
            txt = await page.locator(".slot").nth(i).text_content()
            slot_texts.append(txt)
        print(f"  Slots ({slots}): {slot_texts}")
        print("  PASS: subtitle generated")

        print("TEST 3b: Rating quality tags")
        interesting_btn = page.locator('[data-testid="tag-interesting"]')
        realistic_btn = page.locator('[data-testid="tag-realistic"]')
        assert await interesting_btn.is_visible(), "Interesting tag should be visible"
        assert await realistic_btn.is_visible(), "Realistic tag should be visible"
        await interesting_btn.click()
        interesting_class = await interesting_btn.get_attribute("class")
        assert "active" in interesting_class, f"Expected Interesting tag active, got: {interesting_class}"
        print("  PASS: realistic/interesting tags visible and toggle")

        if not is_local:
            print("TEST 3c: App Insights generate telemetry")
            await wait_for_telemetry(
                lambda: any(
                    "GenerateSuccess" in post or "GenerateDuration" in post
                    for post in telemetry_posts
                ),
                "GenerateSuccess/GenerateDuration",
                timeout=30.0,
            )
            print("  PASS: generate telemetry posted")

        # 4. Check sources appear
        print("TEST 4: Sources panel")
        await page.wait_for_function(
            "() => document.querySelectorAll('.source-line').length >= 4",
            timeout=15000,
        )
        source_lines = await page.locator(".source-line").count()
        assert source_lines >= 4, f"Expected >= 4 source lines, got {source_lines}"
        print(f"  PASS: {source_lines} source lines shown")

        # 5. Build Prompt (jacket dry_run)
        print("TEST 5: Build Prompt")
        prompt_btn = page.locator("button:has-text('Build Prompt')")
        await prompt_btn.click()
        await page.wait_for_function(
            "() => document.querySelector('.prompt-text')?.textContent?.length > 100",
            timeout=30000,
        )
        prompt_text = await page.locator(".prompt-text").text_content()
        assert len(prompt_text) > 100, f"Prompt too short: {len(prompt_text)} chars"
        print(f"  PASS: prompt built ({len(prompt_text)} chars)")

        # 6. Copy button exists
        print("TEST 6: Copy button exists")
        copy_btn = page.locator("button:has-text('Copy')").first
        assert await copy_btn.is_visible(), "No Copy button"
        print("  PASS: Copy button present")

        # 7. Settings (mode-dependent)
        print("TEST 7: Settings")
        tone_select = page.locator("select").first
        await tone_select.select_option("pop")
        if is_local:
            await page.wait_for_function(
                "() => document.querySelectorAll('select').length >= 2",
                timeout=15000,
            )
            model_selects = await page.locator("select").count()
            assert model_selects >= 2, f"Expected >= 2 selects (tone + model), got {model_selects}"
            print("  PASS: model picker visible (local mode)")
        else:
            # In web mode, model picker should be hidden (not visible)
            await page.wait_for_timeout(2000)
            model_visible = await page.locator("select").nth(1).is_visible()
            assert not model_visible, "Model picker should be hidden in web mode"
            print("  PASS: model picker hidden (web mode)")

        # 8. Generate with a hard tier filter
        print("TEST 8: Generate with tone=pop")
        await gen_btn.click()
        await page.wait_for_function(
            "() => document.querySelectorAll('.slot').length >= 4",
            timeout=60000,
        )
        slots2 = await page.locator(".slot").count()
        assert slots2 >= 4, f"Expected >= 4 slots, got {slots2}"
        print("  PASS: regenerated with tier filter")

        # 9. GitHub link in footer
        print("TEST 9: GitHub link")
        gh_link = page.locator("a[href*='github.com/Arithmomaniac/subtitle-generator']")
        assert await gh_link.count() > 0, "No GitHub link in footer"
        print("  PASS: GitHub link present")

        # 10. Generate until remix (sub-parts visible)
        print("TEST 10: Generate until remix")
        await tone_select.select_option("")
        got_remix = False
        # Use a more specific locator to avoid matching hidden "Generate Jacket"
        generate_btn = page.locator("button.btn-primary:has-text('Generate')")
        for attempt in range(30):
            await generate_btn.click()
            await page.wait_for_function(
                "() => document.querySelectorAll('.slot').length >= 4",
                timeout=60000,
            )
            subparts = await page.locator(".slot-subpart").count()
            if subparts >= 2:
                parts = []
                for i in range(subparts):
                    parts.append(await page.locator(".slot-subpart").nth(i).text_content())
                print(f"  Remix found on attempt {attempt + 1}: {parts}")
                # Verify remix similarity line appears
                sim_line = page.locator(".remix-info")
                if await sim_line.count() > 0:
                    sim_text = await sim_line.text_content()
                    print(f"  {sim_text}")
                got_remix = True
                break
        assert got_remix, "No remix after 30 attempts (remix_prob=0.8, expected ~80%)"
        print("  PASS: remix sub-parts rendered")

        # 11. Mobile layout should not create horizontal overflow
        print("TEST 11: Mobile spacing")
        await page.set_viewport_size({"width": 390, "height": 844})
        await page.wait_for_timeout(500)
        has_horizontal_overflow = await page.evaluate(
            "() => document.documentElement.scrollWidth > window.innerWidth"
        )
        assert not has_horizontal_overflow, "Mobile layout should not horizontally overflow"
        print("  PASS: mobile layout stays within viewport")

        print()
        print(f"ALL 11 TESTS PASSED ({BASE_URL})")
        await browser.close()


asyncio.run(test())
