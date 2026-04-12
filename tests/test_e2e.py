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
        badge = page.locator("#modeBadge")
        # Wait for mode detection to complete (async fetch)
        await page.wait_for_function(
            "() => !document.getElementById('modeBadge').textContent.includes('detecting')",
            timeout=10000,
        )
        badge_text = await badge.text_content()
        print(f"  Badge text: {badge_text}")
        assert "Local" in badge_text, f"Expected Local mode, got: {badge_text}"
        print("  PASS: local mode detected")

        # 3. Click Generate
        print("TEST 3: Generate subtitle")
        gen_btn = page.locator("#generateBtn")
        await gen_btn.click()
        slot = page.locator(".slot").first
        await slot.wait_for(timeout=15000)
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
        sources = page.locator("#sources")
        is_visible = await sources.is_visible()
        assert is_visible, "Sources panel not visible"
        source_lines = await page.locator(".source-line").count()
        assert source_lines >= 4, f"Expected >= 4 source lines, got {source_lines}"
        print(f"  PASS: {source_lines} source lines shown")

        # 5. Build Prompt (jacket dry_run)
        print("TEST 5: Build Prompt")
        prompt_btn = page.locator("#promptBtn")
        await prompt_btn.click()
        prompt_section = page.locator("#promptSection")
        await prompt_section.wait_for(state="visible", timeout=10000)
        prompt_text = await page.locator("#promptText").text_content()
        assert len(prompt_text) > 100, f"Prompt too short: {len(prompt_text)} chars"
        print(f"  PASS: prompt built ({len(prompt_text)} chars)")

        # 6. Copy button exists
        print("TEST 6: Copy button exists")
        copy_btn = page.locator("button:has-text('Copy')")
        assert await copy_btn.first.is_visible(), "No Copy button"
        print("  PASS: Copy button present")

        # 7. Settings visibility in local mode
        print("TEST 7: Settings")
        tone_select = page.locator("#tone")
        await tone_select.select_option("pop")
        remix_prob = page.locator("#remixProb")
        is_visible = await remix_prob.is_visible()
        assert is_visible, "remixProb should be visible in local mode"
        print("  PASS: settings interactive, local-only visible")

        print()
        print("ALL 7 TESTS PASSED")
        await browser.close()


asyncio.run(test())
