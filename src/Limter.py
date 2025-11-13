import asyncio
import time

# 模拟 LLM 接口的 token 消耗估算（可根据模型类型调整）
R_PER_TOKENS = 700


# 动态限速控制器
class RateLimiter:
    def __init__(self, rpm_limit=1000, tpm_limit=50000, safety_margin=0.8):
        self.rpm_limit = int(rpm_limit * safety_margin)
        self.tpm_limit = int(tpm_limit * safety_margin)
        self.reset_time = time.time()
        self.req_count = 0
        self.token_count = 0

    async def check_limit(self):
        now = time.time()
        elapsed = now - self.reset_time

        # 每分钟重置
        if elapsed > 60:
            self.reset_time = now
            self.req_count = 0
            self.token_count = 0
            return

        # 如果请求或token超限，就sleep直到窗口重置
        if self.req_count >= self.rpm_limit or self.token_count >= self.tpm_limit:
            wait = 60 - elapsed
            print(
                f"⚠️ 达到速率限制（req={self.req_count},rpm_limit={self.rpm_limit}, tokens={self.token_count}, tpm_limit={self.tpm_limit})），等待 {wait:.1f} 秒...")
            await asyncio.sleep(wait)
            self.reset_time = time.time()
            self.req_count = 0
            self.token_count = 0

    async def record_call(self, token_used: int):
        self.req_count += 1
        self.token_count += token_used
        await self.check_limit()
