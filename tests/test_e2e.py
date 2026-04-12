"""E2E browser tests for the subtitle-generator web app using Playwright."""

import asyncio
from playwright.async_api import async_playwright


async def test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # 1. Load page
        print("TEST 1: Load page")
        await page.goto("http://localhost:8742")
        title = await page.title()
        assert "Subtitle Generator" in title, f"Bad title: {title}"
        print(f"  PASS: title = {title}")

        # 2. Check mode badge appears
        print("TEST 2: Mode detection")
        # Wait for Alpine to detect mode (async health check)
        await page.wait_for_function(
            "() => document.querySelector('.mode-badge')?.textContent?.includes('Mode')",
            timeout=10000,
        )
        badge_text = await page.locator(".mode-badge").text_content()
        print(f"  Badge text: {badge_text}")
        assert "Local" in badge_text, f"Expected Local mode, got: {badge_text}"
        print("  PASS: local mode detected")

        # 3. Click Generate
        print("TEST 3: Generate subtitle")
        gen_btn = page.locator("button:has-text('Generate')").first
        await gen_btn.click()
        # Wait for Alpine to render slots
        await page.wait_for_function(
            "() => document.querySelectorAll('.slot').length >= 4",
            timeout=15000,
        )
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
        # Wait for Alpine to render sources (x-show transition)
        await page.wait_for_function(
            "() => document.querySelectorAll('.source-line').length >= 4",
            timeout=10000,
        )
        source_lines = await page.locator(".source-line").count()
        assert source_lines >= 4, f"Expected >= 4 source lines, got {source_lines}"
        print(f"  PASS: {source_lines} source lines shown")

        # 5. Build Prompt (jacket dry_run)
        print("TEST 5: Build Prompt")
        prompt_btn = page.locator("button:has-text('Build Prompt')")
        await prompt_btn.click()
        prompt_section = page.locator(".prompt-section")
        await page.wait_for_function(
            "() => document.querySelector('.prompt-text')?.textContent?.length > 100",
            timeout=10000,
        )
        prompt_text = await page.locator(".prompt-text").text_content()
        assert len(prompt_text) > 100, f"Prompt too short: {len(prompt_text)} chars"
        print(f"  PASS: prompt built ({len(prompt_text)} chars)")

        # 6. Copy button exists
        print("TEST 6: Copy button exists")
        copy_btn = page.locator("button:has-text('Copy')").first
        assert await copy_btn.is_visible(), "No Copy button"
        print("  PASS: Copy button present")

        # 7. Settings visibility in local mode
        print("TEST 7: Settings")
        tone_select = page.locator("select").first
        await tone_select.select_option("pop")
        # In local mode, remix prob input should be visible
        remix_prob = page.locator("input[type='number']").first
        is_visible = await remix_prob.is_visible()
        assert is_visible, "remixProb should be visible in local mode"
        print("  PASS: settings interactive, local-only visible")

        print()
        print("ALL 7 TESTS PASSED")
        await browser.close()


asyncio.run(test())
