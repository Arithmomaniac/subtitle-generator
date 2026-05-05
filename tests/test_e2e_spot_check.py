"""Browser e2e smoke test for the local spot-check page.

The spot-check endpoints are local-only and are intentionally not deployed to
Azure.

Run:
    uv run python tests/test_e2e_spot_check.py
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

from playwright.async_api import Page, async_playwright

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8742")


@dataclass
class RateRequestCapture:
    payloads: list[dict] = field(default_factory=list)

    async def route(self, route) -> None:
        self.payloads.append(route.request.post_data_json)
        await route.continue_()

    def clear(self) -> None:
        self.payloads.clear()

    def assert_single(self, **expected) -> None:
        assert len(self.payloads) == 1, f"Expected 1 rate request, got {len(self.payloads)}"
        payload = self.payloads[0]
        for key, value in expected.items():
            assert payload.get(key) == value, f"Expected {key}={value!r}, got {payload}"


async def load_spot_check_page(page: Page) -> None:
    print("TEST 1: Load spot-check page")
    await page.goto(BASE_URL + "/spot-check.html")
    title = await page.title()
    assert "Spot Check" in title, f"Bad title: {title}"
    header = await page.locator("h1").text_content()
    assert "Spot Check" in header, f"Bad header: {header}"
    print(f"  PASS: title = {title}")


async def load_batch(page: Page) -> None:
    print("TEST 2: Load batch")
    await page.locator('[data-testid="samples-select"]').select_option("1")
    await page.locator('[data-testid="load-batch-btn"]').click()
    await page.wait_for_function(
        "() => document.querySelector('[data-testid=\"subtitle-card\"]') !== null",
        timeout=30000,
    )
    progress = await page.locator('[data-testid="progress-text"]').text_content()
    assert "1 / 3" in progress, f"Expected '1 / 3', got: {progress}"
    print(f"  PASS: batch loaded, progress = {progress}")


async def rate_pop_and_reveal(page: Page, capture: RateRequestCapture) -> None:
    print("TEST 3: Click tier button (Pop)")
    capture.clear()
    await page.locator('[data-testid="btn-pop"]').click()
    await page.wait_for_function(
        "() => document.querySelector('[data-testid=\"reveal-panel\"]') !== null",
        timeout=10000,
    )
    reveal_text = await page.locator('[data-testid="reveal-panel"]').text_content()
    reveal_lower = reveal_text.lower()
    assert (
        "target was" in reveal_lower
        or "match" in reveal_lower
        or "mismatch" in reveal_lower
    ), f"Reveal panel does not show result: {reveal_text}"
    capture.assert_single(felt_tier="pop")
    print("  PASS: rated as pop, reveal shown")


async def advance_to_next_item(page: Page) -> None:
    print("TEST 4: Click Next to advance")
    await page.locator('[data-testid="btn-next"]').click()
    await page.wait_for_function(
        "() => document.querySelector('[data-testid=\"tier-section\"]') !== null",
        timeout=10000,
    )
    progress = await page.locator('[data-testid="progress-text"]').text_content()
    assert "2 / 3" in progress, f"Expected '2 / 3', got: {progress}"
    print("  PASS: advanced to item 2")


async def rate_niche_with_keyboard(page: Page, capture: RateRequestCapture) -> None:
    print("TEST 5: Keyboard shortcut (n for niche)")
    capture.clear()
    await page.keyboard.press("n")
    await page.wait_for_function(
        "() => document.querySelector('[data-testid=\"reveal-panel\"]') !== null",
        timeout=10000,
    )
    capture.assert_single(felt_tier="niche")
    print("  PASS: rated as niche via keyboard")


async def toggle_reveal_tags(page: Page) -> None:
    print("TEST 6: Toggle tags during reveal")
    await page.keyboard.press("f")
    funny_button = page.locator('[data-testid="tag-funny"]')
    funny_class = await funny_button.get_attribute("class")
    assert "active" in funny_class, f"Expected funny tag active, got class: {funny_class}"

    await page.keyboard.press("f")
    funny_class = await funny_button.get_attribute("class")
    assert "active" not in funny_class, "Expected funny tag inactive after toggle"

    await funny_button.click()
    await page.keyboard.press("i")
    interesting_class = await page.locator('[data-testid="tag-interesting"]').get_attribute("class")
    assert "active" in interesting_class, f"Expected interesting tag active, got: {interesting_class}"

    await page.keyboard.press("l")
    realistic_class = await page.locator('[data-testid="tag-realistic"]').get_attribute("class")
    assert "active" in realistic_class, f"Expected realistic tag active, got: {realistic_class}"
    print("  PASS: tags toggle correctly")


async def advance_with_enter(page: Page) -> None:
    await page.keyboard.press("Enter")
    await page.wait_for_function(
        "() => document.querySelector('[data-testid=\"tier-section\"]') !== null",
        timeout=10000,
    )


async def skip_final_item(page: Page, capture: RateRequestCapture) -> None:
    print("TEST 7: Skip a subtitle")
    capture.clear()
    await page.keyboard.press("s")
    await page.wait_for_function(
        "() => document.querySelector('[data-testid=\"reveal-panel\"]') !== null",
        timeout=10000,
    )
    capture.assert_single(skipped=True)
    skip_text = await page.locator('[data-testid="reveal-skip"]').text_content()
    assert "Skipped" in skip_text, f"Expected skip reveal, got: {skip_text}"
    print("  PASS: skipped, reveal shows target")


async def verify_summary(page: Page) -> None:
    print("TEST 8: Batch summary")
    await page.keyboard.press("Enter")
    await page.wait_for_function(
        "() => document.querySelector('[data-testid=\"summary-panel\"]') !== null",
        timeout=10000,
    )
    accuracy_text = await page.locator('[data-testid="batch-accuracy"]').text_content()
    session_text = await page.locator('[data-testid="session-total"]').text_content()
    assert "%" in accuracy_text or "-" in accuracy_text, f"Bad accuracy: {accuracy_text}"
    assert "rated" in session_text.lower(), f"Bad session total: {session_text}"
    print(f"  PASS: summary shown - accuracy={accuracy_text}, session={session_text}")


async def load_more_and_verify_chrome(page: Page) -> None:
    print("TEST 9: Load More")
    await page.locator('[data-testid="load-more-btn"]').click()
    await page.wait_for_function(
        "() => document.querySelector('[data-testid=\"subtitle-card\"]') !== null",
        timeout=30000,
    )
    progress = await page.locator('[data-testid="progress-text"]').text_content()
    assert "1 / 3" in progress, f"Expected '1 / 3' after load more, got: {progress}"
    print("  PASS: new batch loaded, progress reset")

    print("TEST 10: Keyboard hints")
    hints = page.locator('[data-testid="keyboard-hints"]')
    assert await hints.is_visible(), "Keyboard hints not visible"
    hints_text = await hints.text_content()
    assert "Pop" in hints_text and "Mainstream" in hints_text, f"Missing hint text: {hints_text}"
    print("  PASS: keyboard hints visible")

    print("TEST 11: Back link")
    back_link = page.locator("a.back-link")
    assert await back_link.is_visible(), "No back link"
    href = await back_link.get_attribute("href")
    assert href == "/", f"Back link should go to /, got: {href}"
    print("  PASS: back link present")


async def run_spot_check_e2e() -> None:
    print(f"Testing spot-check against: {BASE_URL}\n")
    capture = RateRequestCapture()

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.route("**/api/spot-check/rate", capture.route)
        try:
            await load_spot_check_page(page)
            await load_batch(page)
            await rate_pop_and_reveal(page, capture)
            await advance_to_next_item(page)
            await rate_niche_with_keyboard(page, capture)
            await toggle_reveal_tags(page)
            await advance_with_enter(page)
            await skip_final_item(page, capture)
            await verify_summary(page)
            await load_more_and_verify_chrome(page)
            print()
            print(f"ALL 11 TESTS PASSED ({BASE_URL})")
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(run_spot_check_e2e())
