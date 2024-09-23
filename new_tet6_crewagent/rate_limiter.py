# rate_limiter.py
import asyncio
import time
from collections import deque

class RateLimiter:
    def __init__(self, calls_per_minute, burst_limit=None):
        self.calls_per_minute = calls_per_minute
        self.interval = 60 / calls_per_minute
        self.burst_limit = burst_limit or calls_per_minute
        self.call_times = deque(maxlen=self.burst_limit)
        self.queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.task = None

    async def wait(self):
        async with self.lock:
            now = time.time()
            if len(self.call_times) == self.burst_limit:
                oldest_call = self.call_times[0]
                if now - oldest_call < 60:
                    wait_time = 60 - (now - oldest_call)
                    await asyncio.sleep(wait_time)
            
            self.call_times.append(time.time())

    async def run(self, coroutine):
        future = asyncio.Future()
        await self.queue.put((coroutine, future))
        
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._process_queue())
        
        return await future

    async def _process_queue(self):
        while not self.queue.empty():
            coroutine, future = await self.queue.get()
            await self.wait()
            try:
                result = await coroutine
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self.queue.task_done()