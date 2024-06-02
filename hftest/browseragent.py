# browseragent.py

import asyncio
from playwright.async_api import async_playwright

class BrowserAgent:
    def __init__(self):
        pass
    
    async def start_browser(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
    
    async def navigate_to(self, url):
        await self.page.goto(url)
    
    async def fill_form(self, selector, text):
        await self.page.fill(selector, text)
    
    async def click_button(self, selector):
        await self.page.click(selector)
    
    async def close_browser(self):
        await self.browser.close()
        await self.playwright.stop()

    async def run_task(self, task):
        await self.start_browser()
        await task(self.page)
        await self.close_browser()
