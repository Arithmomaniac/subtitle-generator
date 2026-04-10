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
            txt = await page.locator(".slot").nth(i).locator(".slot-text").text_content()
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

        # 5. Lock a slot
        print("TEST 5: Lock a slot")
        first_slot = page.locator(".slot").first
        first_text = await first_slot.locator(".slot-text").text_content()
        lock_icon = first_slot.locator(".lock-icon")
        await lock_icon.click()
        has_locked = await first_slot.evaluate("el => el.classList.contains('locked')")
        assert has_locked, "Slot not locked after click"
        print(f'  PASS: locked "{first_text}"')

        # 6. Regenerate with lock
        print("TEST 6: Regenerate with locked slot")
        await gen_btn.click()
        await page.wait_for_timeout(3000)
        new_first = await page.locator(".slot").first.locator(".slot-text").text_content()
        assert new_first == first_text, f"Locked slot changed: {first_text} -> {new_first}"
        still_locked = await page.locator(".slot").first.evaluate(
            "el => el.classList.contains('locked')"
        )
        assert still_locked, "Lock state lost after regenerate"
        print(f'  PASS: "{first_text}" preserved after regenerate')

        # 7. Click-to-edit
        print("TEST 7: Click-to-edit")
        second_slot = page.locator(".slot").nth(1)
        second_text_el = second_slot.locator(".slot-text")
        await second_text_el.click()
        edit_input = second_slot.locator(".slot-edit")
        await edit_input.wait_for(timeout=3000)
        await edit_input.fill("Cats")
        await edit_input.press("Enter")
        await page.wait_for_timeout(500)
        new_val = await second_slot.locator(".slot-text").text_content()
        assert new_val == "Cats", f"Edit failed: got {new_val}"
        is_now_locked = await second_slot.evaluate("el => el.classList.contains('locked')")
        assert is_now_locked, "Edited slot not auto-locked"
        print('  PASS: edited to "Cats", auto-locked')

        # 8. Clear locks
        print("TEST 8: Clear locks")
        await page.locator("text=Clear Locks").click()
        locked_count = await page.locator(".slot.locked").count()
        assert locked_count == 0, f"Still {locked_count} locked after clear"
        print("  PASS: all locks cleared")

        # 9. Build Prompt (jacket dry_run)
        print("TEST 9: Build Prompt")
        prompt_btn = page.locator("#promptBtn")
        await prompt_btn.click()
        prompt_section = page.locator("#promptSection")
        await prompt_section.wait_for(state="visible", timeout=10000)
        prompt_text = await page.locator("#promptText").text_content()
        assert len(prompt_text) > 100, f"Prompt too short: {len(prompt_text)} chars"
        print(f"  PASS: prompt built ({len(prompt_text)} chars)")

        # 10. Copy button exists
        print("TEST 10: Copy button exists")
        copy_btn = page.locator("button:has-text('Copy')")
        assert await copy_btn.first.is_visible(), "No Copy button"
        print("  PASS: Copy button present")

        # 11. Settings visibility in local mode
        print("TEST 11: Settings")
        tone_select = page.locator("#tone")
        await tone_select.select_option("pop")
        remix_prob = page.locator("#remixProb")
        is_visible = await remix_prob.is_visible()
        assert is_visible, "remixProb should be visible in local mode"
        print("  PASS: settings interactive, local-only visible")

        print()
        print("ALL 11 TESTS PASSED")
        await browser.close()


asyncio.run(test())
