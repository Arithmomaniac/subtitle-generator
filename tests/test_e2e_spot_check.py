"""E2E browser tests for the spot-check page using Playwright.

Local-only — the spot-check endpoints are not deployed to Azure.

Usage:
  python tests/test_e2e_spot_check.py                    # default localhost:8742
  BASE_URL=http://localhost:9000 python tests/test_e2e_spot_check.py
"""

import asyncio
import os
from playwright.async_api import async_playwright

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8742")


async def test():
    print(f"Testing spot-check against: {BASE_URL}\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Intercept /api/spot-check/rate calls to verify payloads
        rate_requests: list[dict] = []

        async def capture_rate(route):
            request = route.request
            body = request.post_data_json
            rate_requests.append(body)
            await route.continue_()

        await page.route("**/api/spot-check/rate", capture_rate)

        # ── TEST 1: Load page ──
        print("TEST 1: Load spot-check page")
        await page.goto(BASE_URL + "/spot-check.html")
        title = await page.title()
        assert "Spot Check" in title, f"Bad title: {title}"
        header = await page.locator("h1").text_content()
        assert "Spot Check" in header, f"Bad header: {header}"
        print(f"  PASS: title = {title}")

        # ── TEST 2: Load batch ──
        print("TEST 2: Load batch")
        # Set samples to 1 per tier for faster tests
        await page.locator('[data-testid="samples-select"]').select_option("1")
        await page.locator('[data-testid="load-batch-btn"]').click()

        # Wait for rating phase
        await page.wait_for_function(
            "() => document.querySelector('[data-testid=\"subtitle-card\"]') !== null",
            timeout=30000,
        )
        progress = await page.locator('[data-testid="progress-text"]').text_content()
        assert "1 / 3" in progress, f"Expected '1 / 3', got: {progress}"
        print(f"  PASS: batch loaded, progress = {progress}")

        # ── TEST 3: Click tier button ──
        print("TEST 3: Click tier button (Pop)")
        rate_requests.clear()
        await page.locator('[data-testid="btn-pop"]').click()

        # Should enter reveal phase
        await page.wait_for_function(
            "() => document.querySelector('[data-testid=\"reveal-panel\"]') !== null",
            timeout=10000,
        )
        reveal_text = await page.locator('[data-testid="reveal-panel"]').text_content()
        assert "target was" in reveal_text.lower() or "match" in reveal_text.lower() or "mismatch" in reveal_text.lower(), \
            f"Reveal panel doesn't show result: {reveal_text}"
        # Verify rate request was sent
        assert len(rate_requests) == 1, f"Expected 1 rate request, got {len(rate_requests)}"
        assert rate_requests[0].get("felt_tier") == "pop", f"Expected felt_tier=pop, got {rate_requests[0]}"
        print(f"  PASS: rated as pop, reveal shown")

        # ── TEST 4: Advance with Next button ──
        print("TEST 4: Click Next to advance")
        await page.locator('[data-testid="btn-next"]').click()
        await page.wait_for_function(
            "() => document.querySelector('[data-testid=\"tier-section\"]') !== null",
            timeout=10000,
        )
        progress = await page.locator('[data-testid="progress-text"]').text_content()
        assert "2 / 3" in progress, f"Expected '2 / 3', got: {progress}"
        print(f"  PASS: advanced to item 2")

        # ── TEST 5: Keyboard shortcut (press 'n' for niche) ──
        print("TEST 5: Keyboard shortcut (n for niche)")
        rate_requests.clear()
        await page.keyboard.press("n")
        await page.wait_for_function(
            "() => document.querySelector('[data-testid=\"reveal-panel\"]') !== null",
            timeout=10000,
        )
        assert len(rate_requests) == 1, f"Expected 1 rate request, got {len(rate_requests)}"
        assert rate_requests[0].get("felt_tier") == "niche", f"Expected felt_tier=niche, got {rate_requests[0]}"
        print(f"  PASS: rated as niche via keyboard")

        # ── TEST 6: Toggle tags during reveal ──
        print("TEST 6: Toggle tags during reveal")
        # Press 'f' for funny tag
        await page.keyboard.press("f")
        funny_btn = page.locator('[data-testid="tag-funny"]')
        funny_class = await funny_btn.get_attribute("class")
        assert "active" in funny_class, f"Expected funny tag active, got class: {funny_class}"
        # Toggle off
        await page.keyboard.press("f")
        funny_class = await funny_btn.get_attribute("class")
        assert "active" not in funny_class, f"Expected funny tag inactive after toggle"
        # Toggle on again for the submission
        await page.locator('[data-testid="tag-funny"]').click()
        print(f"  PASS: tags toggle correctly")

        # Advance to next
        await page.keyboard.press("Enter")
        await page.wait_for_function(
            "() => document.querySelector('[data-testid=\"tier-section\"]') !== null",
            timeout=10000,
        )

        # ── TEST 7: Skip ──
        print("TEST 7: Skip a subtitle")
        rate_requests.clear()
        await page.keyboard.press("s")
        await page.wait_for_function(
            "() => document.querySelector('[data-testid=\"reveal-panel\"]') !== null",
            timeout=10000,
        )
        assert len(rate_requests) == 1, f"Expected 1 rate request, got {len(rate_requests)}"
        assert rate_requests[0].get("skipped") is True, f"Expected skipped=true, got {rate_requests[0]}"
        skip_text = await page.locator('[data-testid="reveal-skip"]').text_content()
        assert "Skipped" in skip_text, f"Expected skip reveal, got: {skip_text}"
        print(f"  PASS: skipped, reveal shows target")

        # Advance — should go to summary
        await page.keyboard.press("Enter")

        # ── TEST 8: Summary ──
        print("TEST 8: Batch summary")
        await page.wait_for_function(
            "() => document.querySelector('[data-testid=\"summary-panel\"]') !== null",
            timeout=10000,
        )
        accuracy_text = await page.locator('[data-testid="batch-accuracy"]').text_content()
        session_text = await page.locator('[data-testid="session-total"]').text_content()
        assert "%" in accuracy_text or "—" in accuracy_text, f"Bad accuracy: {accuracy_text}"
        assert "rated" in session_text.lower(), f"Bad session total: {session_text}"
        print(f"  PASS: summary shown — accuracy={accuracy_text}, session={session_text}")

        # ── TEST 9: Load More ──
        print("TEST 9: Load More")
        await page.locator('[data-testid="load-more-btn"]').click()
        await page.wait_for_function(
            "() => document.querySelector('[data-testid=\"subtitle-card\"]') !== null",
            timeout=30000,
        )
        progress = await page.locator('[data-testid="progress-text"]').text_content()
        assert "1 / 3" in progress, f"Expected '1 / 3' after load more, got: {progress}"
        print(f"  PASS: new batch loaded, progress reset")

        # ── TEST 10: Keyboard hints visible ──
        print("TEST 10: Keyboard hints")
        hints = page.locator('[data-testid="keyboard-hints"]')
        assert await hints.is_visible(), "Keyboard hints not visible"
        hints_text = await hints.text_content()
        assert "Pop" in hints_text and "Mainstream" in hints_text, f"Missing hint text: {hints_text}"
        print(f"  PASS: keyboard hints visible")

        # ── TEST 11: Back link ──
        print("TEST 11: Back link")
        back_link = page.locator("a.back-link")
        assert await back_link.is_visible(), "No back link"
        href = await back_link.get_attribute("href")
        assert href == "/", f"Back link should go to /, got: {href}"
        print(f"  PASS: back link present")

        print()
        print(f"ALL 11 TESTS PASSED ({BASE_URL})")
        await browser.close()


asyncio.run(test())
