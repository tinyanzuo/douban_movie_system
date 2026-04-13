"""
豆瓣电影 MCP 客户端 - 使用 FastMCP Client
"""
from fastmcp import Client
import asyncio
import json
from typing import Dict, Optional, Any


class DoubanMovieMCPClient:
    """豆瓣电影MCP客户端"""

    def __init__(self, server_url="http://127.0.0.1:8765/mcp"):
        self.server_url = server_url
        self._client = None
        self._connected = False

    async def connect(self):
        """连接服务器"""
        self._client = Client(self.server_url)
        await self._client.__aenter__()
        self._connected = True
        return self

    async def close(self):
        """关闭连接"""
        if self._connected and self._client:
            await self._client.__aexit__(None, None, None)
            self._connected = False

    async def _call_tool(self, tool_name: str, arguments: Dict) -> Any:
        """调用工具并解析返回值"""
        if not self._connected or self._client is None:
            await self.connect()

        result = await self._client.call_tool(tool_name, arguments)

        # 解析 TextContent 返回值
        if hasattr(result, 'content') and result.content:
            for content in result.content:
                if hasattr(content, 'text'):
                    try:
                        return json.loads(content.text)
                    except json.JSONDecodeError:
                        return {"result": content.text}
        return result

    async def search_movie(self, movie_name: str) -> Dict:
        """搜索电影"""
        return await self._call_tool("search_movie", {"movie_name": movie_name})

    async def get_recommendations(self, user_id: str = "user_001", top_n: int = 8) -> Dict:
        """获取推荐"""
        return await self._call_tool("get_movie_recommendations", {
            "user_id": user_id,
            "top_n": top_n
        })

    async def analyze_sentiment(self, movie_name: str, review_text: str) -> Dict:
        """情感分析"""
        return await self._call_tool("analyze_sentiment", {
            "movie_name": movie_name,
            "review_text": review_text
        })

    async def get_movies_by_genre(self, genre: str, limit: int = 10) -> Dict:
        """按类型获取电影"""
        return await self._call_tool("get_movie_by_genre", {
            "genre": genre,
            "limit": limit
        })

    async def compare_movies(self, movie1: str, movie2: str) -> Dict:
        """比较电影"""
        return await self._call_tool("compare_movies", {
            "movie1": movie1,
            "movie2": movie2
        })

    async def get_top_rated_movies(self, limit: int = 10) -> Dict:
        """获取高分电影"""
        return await self._call_tool("get_top_rated_movies", {"limit": limit})

    async def get_movie_stats(self, movie_name: str) -> Dict:
        """获取电影统计"""
        return await self._call_tool("get_movie_stats", {"movie_name": movie_name})

    async def get_user_watch_history(self, user_id: str = "user_001") -> Dict:
        """获取观影历史"""
        return await self._call_tool("get_user_watch_history", {"user_id": user_id})


# 全局客户端实例
_mcp_client = None

def get_mcp_client():
    """获取MCP客户端单例"""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = DoubanMovieMCPClient()
    return _mcp_client


# 同步包装器
class SyncMCPClient:
    def __init__(self, server_url="http://127.0.0.1:8765/mcp"):
        self.server_url = server_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = DoubanMovieMCPClient(self.server_url)
        return self._client

    def _run_async(self, coro):
        """运行异步函数"""
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果循环已在运行，使用线程池
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # 没有事件循环，创建新的
            return asyncio.run(coro)

    def search_movie(self, movie_name: str):
        return self._run_async(self._get_client().search_movie(movie_name))

    def get_recommendations(self, user_id: str = "user_001", top_n: int = 8):
        return self._run_async(self._get_client().get_recommendations(user_id, top_n))

    def analyze_sentiment(self, movie_name: str, review_text: str):
        return self._run_async(self._get_client().analyze_sentiment(movie_name, review_text))

    def get_movies_by_genre(self, genre: str, limit: int = 10):
        return self._run_async(self._get_client().get_movies_by_genre(genre, limit))

    def compare_movies(self, movie1: str, movie2: str):
        return self._run_async(self._get_client().compare_movies(movie1, movie2))

    def get_top_rated_movies(self, limit: int = 10):
        return self._run_async(self._get_client().get_top_rated_movies(limit))

    def get_movie_stats(self, movie_name: str):
        return self._run_async(self._get_client().get_movie_stats(movie_name))

    def get_user_watch_history(self, user_id: str = "user_001"):
        return self._run_async(self._get_client().get_user_watch_history(user_id))


# 创建全局同步客户端实例
sync_mcp_client = SyncMCPClient()