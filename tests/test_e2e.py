"""E2E browser tests for the subtitle-generator web app using Playwright.

Supports both local and deployed modes via BASE_URL env var:
  python tests/test_e2e.py                          # local (localhost:8742)
  BASE_URL=https://subtitlegenst.z13.web.core.windows.net python tests/test_e2e.py
"""

import asyncio
import os
from playwright.async_api import async_playwright

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8742")


async def test():
    is_local = "localhost" in BASE_URL
    print(f"Testing against: {BASE_URL} ({'local' if is_local else 'deployed'})\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

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

        # 8. Generate with tone bias
        print("TEST 8: Generate with tone=pop")
        await gen_btn.click()
        await page.wait_for_function(
            "() => document.querySelectorAll('.slot').length >= 4",
            timeout=60000,
        )
        slots2 = await page.locator(".slot").count()
        assert slots2 >= 4, f"Expected >= 4 slots, got {slots2}"
        print("  PASS: regenerated with tone bias")

        # 9. GitHub link in footer
        print("TEST 9: GitHub link")
        gh_link = page.locator("a[href*='github.com/Arithmomaniac/subtitle-generator']")
        assert await gh_link.count() > 0, "No GitHub link in footer"
        print("  PASS: GitHub link present")

        # 10. Generate until remix (sub-parts visible)
        print("TEST 10: Generate until remix")
        got_remix = False
        for attempt in range(30):
            await gen_btn.click()
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

        print()
        print(f"ALL 10 TESTS PASSED ({BASE_URL})")
        await browser.close()


asyncio.run(test())
