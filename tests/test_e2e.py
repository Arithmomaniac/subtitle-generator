"""Browser e2e smoke test for the subtitle-generator home page.

Run locally:
    uv run python tests/test_e2e.py

Run against deployment:
    $env:BASE_URL = "https://subtitlegenst.z13.web.core.windows.net"
    uv run python tests/test_e2e.py
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlparse

from playwright.async_api import Page, async_playwright

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8742")
AI_TRACK_PATH = "/v2/track"
AI_INGEST_HOSTS = (
    "applicationinsights.azure.com",
    "dc.services.visualstudio.com",
)


@dataclass
class TelemetryCapture:
    posts: list[str] = field(default_factory=list)
    statuses: list[int] = field(default_factory=list)

    def is_track_url(self, url: str) -> bool:
        return AI_TRACK_PATH in url and any(host in url for host in AI_INGEST_HOSTS)

    def capture_request(self, request) -> None:
        if request.method == "POST" and self.is_track_url(request.url):
            self.posts.append(request.post_data or "")

    def capture_response(self, response) -> None:
        if self.is_track_url(response.url):
            self.statuses.append(response.status)

    async def wait_for(
        self,
        predicate: Callable[[], bool],
        label: str,
        timeout: float = 20.0,
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.25)
        raise AssertionError(f"Timed out waiting for App Insights telemetry: {label}")


def is_local_url(url: str) -> bool:
    host = urlparse(url).hostname
    return host in {"localhost", "127.0.0.1", "::1"}


async def generate_until_slots(page: Page, *, attempts: int = 3) -> list[str]:
    generate_button = page.locator("button.btn-primary:has-text('Generate')")
    for attempt in range(attempts):
        await generate_button.click()
        try:
            await page.wait_for_function(
                "() => document.querySelectorAll('.slot').length >= 4",
                timeout=60000,
            )
            break
        except Exception:
            if attempt == attempts - 1:
                raise
            print(f"  Cold start timeout, retrying ({attempt + 1}/{attempts})...")
            await page.wait_for_timeout(5000)

    slot_count = await page.locator(".slot").count()
    assert slot_count >= 4, f"Expected >= 4 slots, got {slot_count}"
    return [
        await page.locator(".slot").nth(index).text_content()
        for index in range(slot_count)
    ]


async def test_load_and_mode(page: Page, *, is_local: bool, telemetry: TelemetryCapture) -> None:
    print("TEST 1: Load page")
    await page.goto(BASE_URL)
    title = await page.title()
    assert "Subtitle Generator" in title, f"Bad title: {title}"
    print(f"  PASS: title = {title}")

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
        await telemetry.wait_for(
            lambda: any("PageviewData" in post for post in telemetry.posts)
            and any(200 <= status < 300 for status in telemetry.statuses),
            "page view",
        )
        print("  PASS: page view telemetry posted")


async def test_generation(page: Page, *, is_local: bool, telemetry: TelemetryCapture) -> None:
    print("TEST 3: Generate subtitle")
    slot_texts = await generate_until_slots(page)
    print(f"  Slots ({len(slot_texts)}): {slot_texts}")
    print("  PASS: subtitle generated")

    print("TEST 3b: Rating quality tags")
    interesting_button = page.locator('[data-testid="tag-interesting"]')
    realistic_button = page.locator('[data-testid="tag-realistic"]')
    assert await interesting_button.is_visible(), "Interesting tag should be visible"
    assert await realistic_button.is_visible(), "Realistic tag should be visible"
    await interesting_button.click()
    interesting_class = await interesting_button.get_attribute("class")
    assert "active" in interesting_class, f"Expected Interesting tag active, got: {interesting_class}"
    print("  PASS: realistic/interesting tags visible and toggle")

    if not is_local:
        print("TEST 3c: App Insights generate telemetry")
        await telemetry.wait_for(
            lambda: any(
                "GenerateSuccess" in post or "GenerateDuration" in post
                for post in telemetry.posts
            ),
            "GenerateSuccess/GenerateDuration",
            timeout=30.0,
        )
        print("  PASS: generate telemetry posted")


async def test_sources_and_prompt(page: Page) -> None:
    print("TEST 4: Sources panel")
    await page.wait_for_function(
        "() => document.querySelectorAll('.source-line').length >= 4",
        timeout=15000,
    )
    source_lines = await page.locator(".source-line").count()
    assert source_lines >= 4, f"Expected >= 4 source lines, got {source_lines}"
    print(f"  PASS: {source_lines} source lines shown")

    print("TEST 5: Build Prompt")
    await page.locator("button:has-text('Build Prompt')").click()
    await page.wait_for_function(
        "() => document.querySelector('.prompt-text')?.textContent?.length > 100",
        timeout=30000,
    )
    prompt_text = await page.locator(".prompt-text").text_content()
    assert len(prompt_text) > 100, f"Prompt too short: {len(prompt_text)} chars"
    print(f"  PASS: prompt built ({len(prompt_text)} chars)")

    print("TEST 6: Copy button exists")
    copy_button = page.locator("button:has-text('Copy')").first
    assert await copy_button.is_visible(), "No Copy button"
    print("  PASS: Copy button present")


async def test_settings_and_tier_filter(page: Page, *, is_local: bool) -> None:
    print("TEST 7: Settings")
    tone_select = page.locator("select").first
    await tone_select.select_option("pop")
    if is_local:
        await page.wait_for_function(
            "() => document.querySelectorAll('select').length >= 2",
            timeout=15000,
        )
        select_count = await page.locator("select").count()
        assert select_count >= 2, f"Expected >= 2 selects (tone + model), got {select_count}"
        print("  PASS: model picker visible (local mode)")
    else:
        await page.wait_for_timeout(2000)
        assert not await page.locator("select").nth(1).is_visible(), (
            "Model picker should be hidden in web mode"
        )
        print("  PASS: model picker hidden (web mode)")

    print("TEST 8: Generate with tone=pop")
    await generate_until_slots(page)
    print("  PASS: regenerated with tier filter")


async def test_footer_remix_and_mobile(page: Page) -> None:
    print("TEST 9: GitHub link")
    github_link = page.locator("a[href*='github.com/Arithmomaniac/subtitle-generator']")
    assert await github_link.count() > 0, "No GitHub link in footer"
    print("  PASS: GitHub link present")

    print("TEST 10: Generate until remix")
    await page.locator("select").first.select_option("")
    generate_button = page.locator("button.btn-primary:has-text('Generate')")
    got_remix = False
    for attempt in range(30):
        await generate_button.click()
        await page.wait_for_function(
            "() => document.querySelectorAll('.slot').length >= 4",
            timeout=60000,
        )
        subpart_count = await page.locator(".slot-subpart").count()
        if subpart_count >= 2:
            parts = [
                await page.locator(".slot-subpart").nth(index).text_content()
                for index in range(subpart_count)
            ]
            print(f"  Remix found on attempt {attempt + 1}: {parts}")
            if await page.locator(".remix-info").count() > 0:
                print(f"  {await page.locator('.remix-info').text_content()}")
            got_remix = True
            break
    assert got_remix, "No remix after 30 attempts (remix_prob=0.8, expected ~80%)"
    print("  PASS: remix sub-parts rendered")

    print("TEST 11: Mobile spacing")
    await page.set_viewport_size({"width": 390, "height": 844})
    await page.wait_for_timeout(500)
    has_horizontal_overflow = await page.evaluate(
        "() => document.documentElement.scrollWidth > window.innerWidth"
    )
    assert not has_horizontal_overflow, "Mobile layout should not horizontally overflow"
    print("  PASS: mobile layout stays within viewport")


async def run_home_e2e() -> None:
    is_local = is_local_url(BASE_URL)
    print(f"Testing against: {BASE_URL} ({'local' if is_local else 'deployed'})\n")
    telemetry = TelemetryCapture()

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        page.on("dialog", lambda dialog: asyncio.create_task(dialog.dismiss()))
        page.on("request", telemetry.capture_request)
        page.on("response", telemetry.capture_response)
        try:
            await test_load_and_mode(page, is_local=is_local, telemetry=telemetry)
            await test_generation(page, is_local=is_local, telemetry=telemetry)
            await test_sources_and_prompt(page)
            await test_settings_and_tier_filter(page, is_local=is_local)
            await test_footer_remix_and_mobile(page)
            print()
            print(f"ALL 11 TESTS PASSED ({BASE_URL})")
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(run_home_e2e())
