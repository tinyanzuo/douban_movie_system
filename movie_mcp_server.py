"""
豆瓣电影 MCP 服务器 - 使用 FastMCP
"""
from fastmcp import FastMCP
import sys
import os
import json

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 创建 FastMCP 服务器
mcp = FastMCP("Douban Movie MCP Server")

# 全局变量
_data_manager = None
_LOCAL_MOVIE_CACHE = {}


def set_data_manager(dm):
    """设置数据管理器"""
    global _data_manager
    _data_manager = dm
    print("✅ MCP服务器已绑定数据管理器")


def set_movie_cache(cache):
    """设置电影缓存"""
    global _LOCAL_MOVIE_CACHE
    _LOCAL_MOVIE_CACHE = cache
    print(f"✅ MCP服务器已加载 {len(cache)} 部电影缓存")


# ==================== 工具1：搜索电影 ====================
@mcp.tool(
    name="search_movie",
    description="搜索电影信息，返回电影的详细信息（评分、导演、演员、简介等）"
)
def search_movie(movie_name: str) -> dict:
    """搜索电影"""
    if _data_manager is None:
        return {"error": "数据管理器未初始化，请先启动Flask应用"}

    result = _data_manager.search_movie_info(movie_name)
    if result["success"]:
        return result["data"]
    return {"error": f"未找到电影「{movie_name}」"}


# ==================== 工具2：电影推荐 ====================
@mcp.tool(
    name="get_movie_recommendations",
    description="根据用户ID获取个性化电影推荐。当用户要求推荐电影时使用此工具。"
)
def get_movie_recommendations(user_id: str = "user_001", top_n: int = 8) -> dict:
    """获取个性化电影推荐"""
    if _data_manager is None:
        return {"error": "数据管理器未初始化，请先启动Flask应用"}

    # 获取所有电影
    all_movies = _data_manager.movies.copy()
    for title, info in _LOCAL_MOVIE_CACHE.items():
        if not any(m.get('title') == title for m in all_movies):
            all_movies.append({
                'id': info.get('id', title),
                'title': title,
                'director': info.get('director', '未知'),
                'actors': info.get('actors', []),
                'genre': info.get('genre', '未知'),
                'year': int(info.get('year', 2000)) if str(info.get('year', '2000')).isdigit() else 2000,
                'rating': float(info.get('rating', 8.0)),
                'description': info.get('description', '')
            })

    user = _data_manager.users.get(user_id, {"watched": []})
    watched_list = user.get('watched', [])

    recommendations = _data_manager.deep_recommender.get_recommendation(
        user_id=user_id,
        all_movies=all_movies,
        top_n=top_n,
        exclude_watched=True,
        watched_list=watched_list
    )

    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "method": "MCP协议推荐 | 召回层(向量检索) + 排序层(NCF)",
        "total": len(recommendations)
    }


# ==================== 工具3：情感分析 ====================
@mcp.tool(
    name="analyze_sentiment",
    description="分析电影评论的情感（正面/中性/负面），使用BERT模型"
)
def analyze_sentiment(movie_name: str, review_text: str) -> dict:
    """情感分析"""
    if _data_manager is None:
        return {"error": "数据管理器未初始化，请先启动Flask应用"}

    sentiment, confidence = _data_manager.sentiment_analyzer.predict_sentiment(review_text)

    sentiment_cn = {"positive": "正面", "neutral": "中性", "negative": "负面"}.get(sentiment, "中性")
    sentiment_icon = {"positive": "😊", "neutral": "😐", "negative": "😞"}.get(sentiment, "🎬")

    return {
        "movie": movie_name,
        "review": review_text[:200] + "..." if len(review_text) > 200 else review_text,
        "sentiment": sentiment,
        "sentiment_cn": sentiment_cn,
        "sentiment_icon": sentiment_icon,
        "confidence": round(confidence, 4),
        "confidence_percent": f"{round(confidence * 100, 1)}%",
        "model": "BERT-base-chinese"
    }


# ==================== 工具4：按类型获取电影 ====================
@mcp.tool(
    name="get_movie_by_genre",
    description="根据类型获取电影列表。当用户说想看某类电影（如科幻片、喜剧片）时使用此工具。"
)
def get_movie_by_genre(genre: str, limit: int = 10) -> dict:
    """按类型获取电影"""
    if _data_manager is None:
        return {"error": "数据管理器未初始化，请先启动Flask应用"}

    movies = []
    for movie in _data_manager.movies:
        if genre in movie.get('genre', ''):
            movies.append({
                "title": movie['title'],
                "rating": movie['rating'],
                "year": movie['year'],
                "director": movie['director'],
                "genre": movie['genre'],
                "description": movie.get('description', '')
            })

    for title, info in _LOCAL_MOVIE_CACHE.items():
        if genre in info.get('genre', '') and not any(m['title'] == title for m in movies):
            movies.append({
                "title": title,
                "rating": float(info.get('rating', 8.0)),
                "year": info.get('year', '未知'),
                "director": info.get('director', '未知'),
                "genre": info.get('genre', '未知'),
                "description": info.get('description', '')
            })

    movies.sort(key=lambda x: x['rating'], reverse=True)

    return {
        "genre": genre,
        "count": len(movies[:limit]),
        "total_available": len(movies),
        "movies": movies[:limit]
    }


# ==================== 工具5：比较电影 ====================
@mcp.tool(
    name="compare_movies",
    description="比较两部电影的评分和特点"
)
def compare_movies(movie1: str, movie2: str) -> dict:
    """比较两部电影"""
    if _data_manager is None:
        return {"error": "数据管理器未初始化，请先启动Flask应用"}

    def get_movie_info(name):
        result = _data_manager.search_movie_info(name)
        if result["success"]:
            return result["data"]
        return None

    m1 = get_movie_info(movie1)
    m2 = get_movie_info(movie2)

    if not m1 or not m2:
        missing = []
        if not m1: missing.append(movie1)
        if not m2: missing.append(movie2)
        return {"error": f"无法找到电影：{', '.join(missing)}"}

    rating1 = float(m1['rating']) if str(m1['rating']).replace('.', '').isdigit() else 0
    rating2 = float(m2['rating']) if str(m2['rating']).replace('.', '').isdigit() else 0

    return {
        "movie1": {
            "title": m1['title'],
            "rating": m1['rating'],
            "year": m1['year'],
            "genre": m1['genre'],
            "director": m1['director'],
            "description": m1.get('description', '')[:100]
        },
        "movie2": {
            "title": m2['title'],
            "rating": m2['rating'],
            "year": m2['year'],
            "genre": m2['genre'],
            "director": m2['director'],
            "description": m2.get('description', '')[:100]
        },
        "comparison": {
            "rating_diff": round(rating1 - rating2, 1),
            "winner": m1['title'] if rating1 > rating2 else m2['title'],
            "recommendation": f"推荐观看《{m1['title'] if rating1 > rating2 else m2['title']}》"
        }
    }


# ==================== 工具6：高分电影排行榜 ====================
@mcp.tool(
    name="get_top_rated_movies",
    description="获取豆瓣高分电影排行榜"
)
def get_top_rated_movies(limit: int = 10) -> dict:
    """获取高分电影排行榜"""
    if _data_manager is None:
        return {"error": "数据管理器未初始化，请先启动Flask应用"}

    all_movies = []
    for movie in _data_manager.movies:
        all_movies.append({
            "title": movie['title'],
            "rating": movie['rating'],
            "year": movie['year'],
            "director": movie['director'],
            "genre": movie['genre']
        })

    for title, info in _LOCAL_MOVIE_CACHE.items():
        if not any(m['title'] == title for m in all_movies):
            all_movies.append({
                "title": title,
                "rating": float(info.get('rating', 8.0)),
                "year": info.get('year', '未知'),
                "director": info.get('director', '未知'),
                "genre": info.get('genre', '未知')
            })

    all_movies.sort(key=lambda x: x['rating'], reverse=True)

    return {
        "title": "豆瓣电影排行榜",
        "total": len(all_movies[:limit]),
        "movies": all_movies[:limit]
    }


# ==================== 启动服务器 ====================
def run_mcp_server(host="127.0.0.1", port=8765):
    """运行MCP服务器"""
    print("=" * 60)
    print("🚀 启动豆瓣电影 MCP 服务器 (FastMCP)")
    print(f"📡 监听地址: http://{host}:{port}")
    print("🔧 可用工具:")
    print("   - search_movie: 搜索电影")
    print("   - get_movie_recommendations: 电影推荐")
    print("   - analyze_sentiment: 情感分析")
    print("   - get_movie_by_genre: 按类型获取电影")
    print("   - compare_movies: 比较电影")
    print("   - get_top_rated_movies: 高分排行榜")
    print("=" * 60)
    print()
    print("⚠️  注意: 此服务器需要与Flask应用配合使用")
    print("   请确保先启动Flask应用，MCP服务器会自动连接")
    print("=" * 60)

    mcp.run(
        transport="http",
        host=host,
        port=port
    )


if __name__ == "__main__":
    run_mcp_server()