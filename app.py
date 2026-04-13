"""
豆瓣电影数据分析与推荐系统 - Web版本
使用BERT预训练模型进行情感分析
推荐引擎: MCP模型上下文协议 + 召回层(向量检索) + 排序层(NCF)
页面结构：首页 | 影视分析 | 推荐引擎 | 社交管理
集成MCP (Model Context Protocol) 标准协议
"""
import torch
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from datetime import datetime
import random
from collections import Counter, defaultdict
import json
import os
import re
import jieba
import pickle
import time
import pandas as pd
import warnings
import threading
import asyncio
import requests
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 导入拆分后的模块
from config import (
    SECRET_KEY, UPLOAD_FOLDER, MAX_CONTENT_LENGTH,
    DOUBAN_SEARCH_CACHE, CACHE_DURATION, LAST_REQUEST_TIME, REQUEST_INTERVAL,
    LOCAL_MOVIE_CACHE, DOUBAN_REAL_REVIEWS, USER_NAMES
)
from spider import DoubanSpider, fuzzy_match_movie, set_poster_dir
from recommend_engine import DeepRecommendationEngine
from models import MovieDataManager

# BERT情感分析模块
from bert_sentiment import get_sentiment_analyzer

# 词云相关库
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    WordCloud = None

# ==================== MCP相关导入（FastMCP版本） ====================
try:
    from movie_mcp_client import sync_mcp_client
    from movie_mcp_server import set_data_manager, set_movie_cache
    MCP_ENABLED = True
    print("✅ FastMCP模块加载成功")
except ImportError as e:
    MCP_ENABLED = False
    print(f"⚠️ FastMCP模块加载失败: {e}")
    print("   请运行: pip install fastmcp")
    class MockMCPClient:
        def search_movie(self, name): return {"error": "MCP未启用"}
        def get_recommendations(self, uid, n): return {"error": "MCP未启用"}
        def analyze_sentiment(self, name, text): return {"error": "MCP未启用"}
        def get_movies_by_genre(self, genre, limit): return {"error": "MCP未启用"}
        def compare_movies(self, m1, m2): return {"error": "MCP未启用"}
        def get_top_rated_movies(self, limit): return {"error": "MCP未启用"}
        def get_movie_stats(self, name): return {"error": "MCP未启用"}
    sync_mcp_client = MockMCPClient()

# 创建Flask应用
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 海报存储目录
POSTER_DIR = os.path.join(app.static_folder, 'posters')
os.makedirs(POSTER_DIR, exist_ok=True)
set_poster_dir(POSTER_DIR)

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据管理器
data_manager = MovieDataManager()

# ==================== MCP服务器启动 ====================

def init_mcp():
    """初始化MCP（绑定数据管理器）"""
    if not MCP_ENABLED:
        return False

    try:
        from movie_mcp_server import set_data_manager, set_movie_cache
        set_data_manager(data_manager)
        set_movie_cache(LOCAL_MOVIE_CACHE)
        print("✅ MCP数据管理器已绑定")
        return True
    except Exception as e:
        print(f"⚠️ MCP初始化失败: {e}")
        return False


def start_mcp_server_in_thread():
    """在后台线程中启动MCP服务器 - 修复debug模式下的I/O错误"""
    if not MCP_ENABLED:
        print("⚠️ MCP未启用，跳过服务器启动")
        return

    # 检查是否在Flask debug子进程中
    import os
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        print("ℹ️ 在Flask debug子进程中，跳过MCP服务器启动（避免重复）")
        return

    def run_server():
        """在独立线程中运行MCP服务器"""
        try:
            # 重定向标准输入，避免I/O错误
            import sys
            import io

            # 保存原始stdin
            original_stdin = sys.stdin

            try:
                # 将stdin替换为null设备
                sys.stdin = open(os.devnull, 'r')

                # 创建新的事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def run():
                    from movie_mcp_server import get_mcp_server
                    server = get_mcp_server()
                    server.set_data_manager(data_manager)
                    await server.run()

                loop.run_until_complete(run())
                loop.close()
            finally:
                # 恢复原始stdin
                sys.stdin = original_stdin

        except Exception as e:
            print(f"MCP服务器运行错误: {e}")
            import traceback
            traceback.print_exc()

    # 启动后台线程
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    print("✅ MCP服务器已启动（后台线程模式）")


# ==================== 路由定义 ====================
@app.route('/image.jpg')
def serve_ai_image():
    from flask import send_from_directory
    return send_from_directory('.', 'image.jpg')


# ==================== MCP API路由 ====================

@app.route('/api/mcp/tools', methods=['GET'])
def api_mcp_tools():
    """获取MCP可用工具列表"""
    tools = [
        {'name': 'search_movie', 'description': '搜索电影信息'},
        {'name': 'get_movie_recommendations', 'description': '获取个性化电影推荐'},
        {'name': 'analyze_sentiment', 'description': 'BERT情感分析'},
        {'name': 'get_movie_by_genre', 'description': '按类型获取电影'},
        {'name': 'get_user_watch_history', 'description': '获取观影历史'},
        {'name': 'get_movie_details', 'description': '获取电影详情'},
        {'name': 'compare_movies', 'description': '比较两部电影'},
        {'name': 'get_top_rated_movies', 'description': '获取高分电影排行榜'},
        {'name': 'get_movie_stats', 'description': '获取电影统计数据'}
    ]
    return jsonify({
        'success': True,
        'tools': tools,
        'mcp_enabled': MCP_ENABLED
    })


@app.route('/api/mcp/call', methods=['POST'])
def api_mcp_call():
    """调用MCP工具"""
    data = request.json
    tool_name = data.get('tool')
    arguments = data.get('arguments', {})

    if not MCP_ENABLED:
        return jsonify({
            'success': False,
            'error': 'MCP服务未启用，请安装fastmcp: pip install fastmcp'
        })

    try:
        if tool_name == 'search_movie':
            result = sync_mcp_client.search_movie(arguments.get('movie_name', ''))
        elif tool_name == 'get_movie_recommendations':
            result = sync_mcp_client.get_recommendations(
                arguments.get('user_id', 'user_001'),
                arguments.get('top_n', 8)
            )
        elif tool_name == 'analyze_sentiment':
            result = sync_mcp_client.analyze_sentiment(
                arguments.get('movie_name', ''),
                arguments.get('review_text', '')
            )
        elif tool_name == 'get_movie_by_genre':
            result = sync_mcp_client.get_movies_by_genre(
                arguments.get('genre', ''),
                arguments.get('limit', 10)
            )
        elif tool_name == 'compare_movies':
            result = sync_mcp_client.compare_movies(
                arguments.get('movie1', ''),
                arguments.get('movie2', '')
            )
        elif tool_name == 'get_user_watch_history':
            result = sync_mcp_client.get_user_watch_history(arguments.get('user_id', 'user_001'))
        elif tool_name == 'get_movie_details':
            # 使用 search_movie 代替
            result = sync_mcp_client.search_movie(arguments.get('movie_id', ''))
        elif tool_name == 'get_top_rated_movies':
            result = sync_mcp_client.get_top_rated_movies(arguments.get('limit', 10))
        elif tool_name == 'get_movie_stats':
            result = sync_mcp_client.get_movie_stats(arguments.get('movie_name', ''))
        else:
            return jsonify({'success': False, 'error': f'未知工具: {tool_name}'})

        # 检查是否有错误
        if result and isinstance(result, dict) and 'error' in result:
            return jsonify({'success': False, 'error': result['error']})

        return jsonify({'success': True, 'result': result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/mcp/status', methods=['GET'])
def api_mcp_status():
    """获取MCP状态"""
    return jsonify({
        'success': True,
        'mcp_enabled': MCP_ENABLED,
        'mcp_version': '1.0.0',
        'server_name': 'douban-movie-server'
    })


# ==================== 智谱AI聊天API ====================
@app.route('/api/zhipu/chat', methods=['POST'])
def api_zhipu_chat():
    """智谱AI聊天接口 - 新版API"""
    import requests
    import json

    data = request.get_json()
    message = data.get('message', '')
    context = data.get('context', [])

    # 你的新API Key
    api_key = 'c0bf09a1c26541dc832cb81a0d05ecd5.ykYT2oZzf93DIbRv'

    # 新版API端点（不需要app_id）
    chat_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 构建消息列表
    messages = []

    # 添加系统提示词
    messages.append({
        "role": "system",
        "content": "你是一个专业的电影助手，名字叫「AI电影助手」。你的职责是帮助用户解决电影相关的问题，包括：推荐电影、介绍电影信息、分析电影剧情、回答电影知识等。请用中文回答，语气友好热情，回答要简洁有用。"
    })

    # 添加历史对话（最近10轮）
    recent_context = context[-10:] if len(context) > 10 else context
    for turn in recent_context:
        messages.append({"role": "user", "content": turn.get("user", "")})
        messages.append({"role": "assistant", "content": turn.get("bot", "")})

    # 添加当前消息
    messages.append({"role": "user", "content": message})

    # 请求数据
    chat_data = {
        "model": "glm-4-flash",  # glm-4-flash 速度快，免费额度充足
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 2000,
        "stream": False
    }

    try:
        response = requests.post(chat_url, headers=headers, json=chat_data, timeout=60)

        print(f"[DEBUG] 智谱API响应状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            # 解析新版API响应
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    full_response = choice['message']['content']
                    return jsonify({'success': True, 'response': full_response})

            return jsonify({'success': False, 'error': '响应解析失败'})

        elif response.status_code == 401:
            return jsonify({'success': False, 'error': 'API密钥无效，请检查智谱AI API配置'})
        elif response.status_code == 429:
            return jsonify({'success': False, 'error': 'API调用次数超限，请稍后再试'})
        else:
            error_text = response.text[:200] if response.text else '未知错误'
            return jsonify({'success': False, 'error': f'API请求失败 ({response.status_code}): {error_text}'})

    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': '请求超时，请稍后重试'})
    except requests.exceptions.ConnectionError:
        return jsonify({'success': False, 'error': '网络连接失败，请检查网络'})
    except Exception as e:
        print(f"[ERROR] 智谱API异常: {e}")
        return jsonify({'success': False, 'error': f'服务异常: {str(e)}'})


# ==================== 页面路由 ====================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/movie_analysis_hub')
def movie_analysis_hub():
    return render_template('movie_analysis_hub.html', movies=data_manager.movies)


@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')


@app.route('/social_hub')
def social_hub():
    return render_template('social_hub.html', user=data_manager.current_user)


# ==================== API路由 ====================
@app.route('/api/get_data_sources', methods=['GET'])
def api_get_data_sources():
    return jsonify({'success': True, 'sources': data_manager.get_all_data_sources()})


@app.route('/api/crawl_reviews', methods=['POST'])
def api_crawl_reviews():
    """爬取豆瓣评论并使用BERT进行情感分析"""
    movie_name = request.json.get('movie_name')
    max_count = int(request.json.get('max_count', 20))
    max_count = min(max_count, 1000)

    if data_manager.spider:
        movie_id, title = data_manager.spider.search_movie_id(movie_name)
        if movie_id:
            def progress_callback(current, total):
                print(f"爬取进度: {current}/{total}")

            reviews = data_manager.spider.crawl_reviews(movie_id, max_count, progress_callback)

            if reviews:
                texts = [r['content'] for r in reviews]
                try:
                    print(f"[DEBUG] 正在使用BERT分析 {len(texts)} 条爬取评论...")
                    results = data_manager.sentiment_analyzer.predict_batch(texts)
                    for i, review in enumerate(reviews):
                        if i < len(results):
                            review['sentiment'] = results[i]['sentiment']
                            review['confidence'] = results[i]['confidence']
                        else:
                            review['sentiment'] = 'neutral'
                            review['confidence'] = 0.5
                    print(f"[DEBUG] BERT分析完成")
                except Exception as e:
                    print(f"[ERROR] BERT分析爬取评论失败: {e}")
                    for review in reviews:
                        review['sentiment'] = 'neutral'
                        review['confidence'] = 0.5

            data_manager.crawled_reviews[title or movie_name] = reviews

            preview_reviews = []
            for review in reviews:
                sentiment_display = review.get('sentiment', 'neutral')
                preview_reviews.append({
                    'user': review.get('user', '豆瓣用户'),
                    'rating': review.get('rating', 3),
                    'content': review.get('content', ''),
                    'time': review.get('time', ''),
                    'sentiment': sentiment_display,
                    'confidence': review.get('confidence', 0.5)
                })

            return jsonify({
                'success': True,
                'reviews': preview_reviews,
                'movie_name': title or movie_name,
                'count': len(reviews)
            })
    return jsonify({'success': False, 'error': '爬取失败'})


@app.route('/api/upload_reviews', methods=['POST'])
def api_upload_reviews():
    """上传评论文件并使用BERT进行情感分析"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有文件'})

    file = request.files['file']
    movie_name = request.form.get('movie_name', '')

    if file.filename == '':
        return jsonify({'success': False, 'error': '文件名为空'})

    if not movie_name:
        return jsonify({'success': False, 'error': '请输入电影名称'})

    try:
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        reviews = []

        # 尝试多种编码格式
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'latin-1', 'cp936']

        if file_ext == '.txt':
            content = None
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    print(f"[DEBUG] TXT文件成功使用编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                raise Exception("无法解码文件，请确保文件是UTF-8或GBK编码")

            lines = content.splitlines()
            for line in lines:
                line = line.strip()
                if line:
                    reviews.append({
                        'content': line,
                        'rating': 3,
                        'user': '上传用户',
                        'time': datetime.now().strftime('%Y-%m-%d')
                    })

        elif file_ext == '.csv':
            import csv
            data_rows = None
            used_encoding = None

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        reader = csv.reader(f)
                        data_rows = list(reader)
                    used_encoding = encoding
                    print(f"[DEBUG] CSV文件成功使用编码: {encoding}")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue

            if data_rows is None:
                raise Exception("无法解码CSV文件，请确保文件是UTF-8或GBK编码")

            # 跳过表头（如果有）
            start_row = 0
            if data_rows and len(data_rows) > 0:
                first_row = data_rows[0]
                # 如果第一行包含"评论"、"内容"、"用户"等关键词，认为是表头
                header_keywords = ['评论', '内容', '用户', '评分', '时间', 'content', 'user', 'rating', 'time']
                is_header = False
                for cell in first_row:
                    if cell and any(keyword in cell.lower() for keyword in header_keywords):
                        is_header = True
                        break
                if is_header:
                    start_row = 1

            for row in data_rows[start_row:]:
                if row and len(row) > 0 and row[0].strip():
                    content = row[0].strip() if len(row) > 0 else ''
                    rating = 3
                    if len(row) > 1 and row[1].strip():
                        try:
                            rating = int(float(row[1].strip()))
                            rating = max(1, min(5, rating))
                        except:
                            rating = 3
                    user = row[2].strip() if len(row) > 2 and row[2].strip() else '上传用户'
                    review_time = row[3].strip() if len(row) > 3 and row[3].strip() else datetime.now().strftime('%Y-%m-%d')

                    if content:  # 只添加有内容的评论
                        reviews.append({
                            'content': content,
                            'rating': rating,
                            'user': user,
                            'time': review_time
                        })

        elif file_ext == '.json':
            import json
            content = None
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    print(f"[DEBUG] JSON文件成功使用编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                raise Exception("无法解码JSON文件，请确保文件是UTF-8或GBK编码")

            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    reviews.append({
                        'content': item.get('content', item.get('text', '')),
                        'rating': item.get('rating', 3),
                        'user': item.get('user', '上传用户'),
                        'time': item.get('time', datetime.now().strftime('%Y-%m-%d'))
                    })
            elif isinstance(data, dict) and 'reviews' in data:
                for item in data['reviews']:
                    reviews.append({
                        'content': item.get('content', item.get('text', '')),
                        'rating': item.get('rating', 3),
                        'user': item.get('user', '上传用户'),
                        'time': item.get('time', datetime.now().strftime('%Y-%m-%d'))
                    })
        else:
            os.remove(file_path)
            return jsonify({'success': False, 'error': f'不支持的文件格式: {file_ext}'})

        if not reviews:
            os.remove(file_path)
            return jsonify({'success': False, 'error': '文件中没有找到有效的评论内容'})

        print(f"[DEBUG] 成功解析 {len(reviews)} 条评论")

        # BERT情感分析
        texts = [r['content'] for r in reviews]
        try:
            print(f"[DEBUG] 正在使用BERT分析 {len(texts)} 条上传评论...")
            results = data_manager.sentiment_analyzer.predict_batch(texts)
            for i, review in enumerate(reviews):
                if i < len(results):
                    review['sentiment'] = results[i]['sentiment']
                    review['confidence'] = results[i]['confidence']
                else:
                    review['sentiment'] = 'neutral'
                    review['confidence'] = 0.5
            print(f"[DEBUG] BERT分析完成")
        except Exception as e:
            print(f"[ERROR] BERT分析上传评论失败: {e}")
            for review in reviews:
                review['sentiment'] = 'neutral'
                review['confidence'] = 0.5

        # 存储上传的评论
        if movie_name not in data_manager.uploaded_reviews:
            data_manager.uploaded_reviews[movie_name] = []
        data_manager.uploaded_reviews[movie_name].extend(reviews)

        # 删除临时文件
        if os.path.exists(file_path):
            os.remove(file_path)

        preview_reviews = []
        for review in reviews[:20]:
            sentiment_display = review.get('sentiment', 'neutral')
            preview_reviews.append({
                'user': review.get('user', '上传用户'),
                'rating': review.get('rating', 3),
                'content': review.get('content', ''),
                'time': review.get('time', ''),
                'sentiment': sentiment_display,
                'confidence': review.get('confidence', 0.5)
            })

        return jsonify({
            'success': True,
            'movie_name': movie_name,
            'count': len(reviews),
            'reviews': preview_reviews
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/search_movie', methods=['POST'])
def api_search_movie():
    movie_name = request.json.get('movie_name', '').strip()
    return jsonify(data_manager.search_movie_info(movie_name))


@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """智能推荐API - 基于当前登录用户 (MCP协议)"""
    user_id = request.json.get('user_id', data_manager.current_user)
    top_n = int(request.json.get('top_n', 8))

    user = data_manager.users.get(user_id, {"watched": []})
    watched_list = user.get('watched', [])

    all_movies = data_manager.movies.copy()
    for title, info in LOCAL_MOVIE_CACHE.items():
        if not any(m.get('title') == title for m in all_movies):
            all_movies.append({
                'id': info.get('id', len(all_movies) + 1000),
                'title': title,
                'director': info.get('director', '未知'),
                'actors': info.get('actors', []),
                'genre': info.get('genre', '未知'),
                'year': int(info.get('year', 2000)),
                'rating': float(info.get('rating', 8.0)),
                'description': info.get('description', '')
            })

    if data_manager.deep_recommender.is_trained:
        recommendations = data_manager.deep_recommender.get_recommendation(
            user_id=user_id,
            all_movies=all_movies,
            top_n=top_n,
            exclude_watched=True,
            watched_list=watched_list
        )
        method = "MCP协议推荐 | 上下文管理 + 向量检索召回 + NCF深度排序"
    else:
        watched_set = set(watched_list)
        candidate = [m for m in all_movies if m['id'] not in watched_set]
        candidate.sort(key=lambda x: x.get('rating', 0), reverse=True)
        recommendations = [{
            **m,
            'predicted_score': m.get('rating', 8.0) / 2,
            'recommendation_score': m.get('rating', 8.0) / 2,
            'match_reason': 'MCP协议 | 热度推荐（模型未训练）'
        } for m in candidate[:top_n]]
        method = "MCP协议 | 热度推荐（模型训练中）"

    return jsonify({
        'success': True,
        'recommendations': recommendations,
        'method': method,
        'user_id': user_id,
        'model_trained': data_manager.deep_recommender.is_trained
    })


@app.route('/api/recommend_by_uploaded_data', methods=['POST'])
def api_recommend_by_uploaded_data():
    """基于上传的用户数据文件进行推荐"""
    data = request.json
    data_id = data.get('data_id')
    top_n = int(data.get('top_n', 8))

    if not data_id or data_id not in data_manager.uploaded_user_data:
        return jsonify({'success': False, 'error': '请先选择有效的上传数据'})

    all_movies = data_manager.movies.copy()
    for title, info in LOCAL_MOVIE_CACHE.items():
        if not any(m.get('title') == title for m in all_movies):
            all_movies.append({
                'id': info.get('id', len(all_movies) + 1000),
                'title': title,
                'director': info.get('director', '未知'),
                'actors': info.get('actors', []),
                'genre': info.get('genre', '未知'),
                'year': int(info.get('year', 2000)),
                'rating': float(info.get('rating', 8.0)),
                'description': info.get('description', '')
            })

    recommendations = data_manager.deep_recommender.get_recommendation_for_uploaded_data(
        data_id=data_id,
        all_movies=all_movies,
        top_n=top_n
    )

    return jsonify({
        'success': True,
        'recommendations': recommendations,
        'method': 'MCP协议分析 + 智能推荐',
        'data_id': data_id,
        'total_recommendations': len(recommendations)
    })


@app.route('/api/similar_movies', methods=['POST'])
def api_similar_movies():
    """基于Embedding的相似电影推荐"""
    data = request.json
    movie_id = data.get('movie_id')
    top_n = int(data.get('top_n', 6))

    if not movie_id:
        return jsonify({'success': False, 'error': '请提供电影ID'})

    similar_movies = data_manager.deep_recommender.find_similar_movies(movie_id, top_n)

    all_movies = {m['id']: m for m in data_manager.movies}
    for title, info in LOCAL_MOVIE_CACHE.items():
        all_movies[info.get('id', title)] = {
            'id': info.get('id', title),
            'title': title,
            'director': info.get('director', '未知'),
            'genre': info.get('genre', '未知'),
            'rating': float(info.get('rating', 8.0)),
            'description': info.get('description', '')
        }

    result = []
    for other_id, similarity in similar_movies:
        if other_id in all_movies:
            movie = all_movies[other_id]
            result.append({
                **movie,
                'similarity': round(similarity, 4)
            })

    return jsonify({
        'success': True,
        'similar_movies': result,
        'movie_id': movie_id
    })


@app.route('/api/similar_users', methods=['POST'])
def api_similar_users():
    """基于Embedding的相似用户查找"""
    data = request.json
    user_id = data.get('user_id', data_manager.current_user)
    top_n = int(data.get('top_n', 5))

    similar_users = data_manager.deep_recommender.find_similar_users(user_id, top_n)

    result = []
    for other_id, similarity in similar_users:
        user_info = data_manager.users.get(other_id, {})
        watched_count = len(user_info.get('watched', []))
        result.append({
            'user_id': other_id,
            'similarity': round(similarity, 4),
            'watched_count': watched_count
        })

    return jsonify({
        'success': True,
        'similar_users': result,
        'user_id': user_id
    })


@app.route('/api/recommender_status', methods=['GET'])
def api_recommender_status():
    """获取推荐引擎状态"""
    return jsonify({
        'success': True,
        'is_trained': data_manager.deep_recommender.is_trained,
        'num_users': data_manager.deep_recommender.num_users,
        'num_movies': data_manager.deep_recommender.num_movies,
        'embedding_dim': data_manager.deep_recommender.embedding_dim,
        'train_losses': data_manager.deep_recommender.train_losses
    })


@app.route('/api/train_recommender', methods=['POST'])
def api_train_recommender():
    """手动训练推荐模型"""
    data = request.json
    epochs = int(data.get('epochs', 30))

    ratings_data = []
    for user_id, user_info in data_manager.users.items():
        for movie_id, rating in user_info.get('ratings', {}).items():
            ratings_data.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating / 2.0
            })

    if len(ratings_data) < 10:
        return jsonify({
            'success': False,
            'error': f'训练数据不足，当前只有 {len(ratings_data)} 条评分记录'
        })

    data_manager.deep_recommender = DeepRecommendationEngine(embedding_dim=64)
    data_manager.deep_recommender.set_data_manager(data_manager)
    success = data_manager.deep_recommender.train(ratings_data, epochs=epochs, batch_size=16)

    if success:
        data_manager.deep_recommender.save_model()
        return jsonify({
            'success': True,
            'message': f'MCP模型训练完成！共 {len(ratings_data)} 条训练数据，{epochs} 轮训练',
            'final_loss': data_manager.deep_recommender.train_losses[-1] if data_manager.deep_recommender.train_losses else None
        })
    else:
        return jsonify({
            'success': False,
            'error': '训练失败，数据不足'
        })


@app.route('/api/analyze_sentiment', methods=['POST'])
def api_analyze_sentiment():
    """使用BERT模型进行情感分析"""
    source_type = request.json.get('source_type')
    name = request.json.get('name')

    if source_type == 'system':
        movie = next((m for m in data_manager.movies if m["title"] == name), None)
        reviews = data_manager.get_system_reviews(movie["id"]) if movie else []
    elif source_type == 'crawled':
        reviews = data_manager.get_crawled_reviews(name)
    else:
        reviews = data_manager.get_uploaded_reviews(name)

    if not reviews:
        return jsonify({'success': False, 'error': '没有找到评论数据'})

    print(f"[DEBUG] 开始使用BERT分析 {len(reviews)} 条评论的情感...")

    texts = [r.get('content', '') for r in reviews if r.get('content')]

    if texts:
        try:
            bert_results = data_manager.sentiment_analyzer.predict_batch(texts)
            for i, review in enumerate(reviews):
                if i < len(bert_results):
                    review['sentiment'] = bert_results[i]['sentiment']
                    review['confidence'] = bert_results[i]['confidence']
                else:
                    review['sentiment'] = data_manager.sentiment_analyzer._rule_based_fallback(review.get('content', ''))
                    review['confidence'] = 0.5
            print(f"[DEBUG] BERT分析完成，共处理 {len(bert_results)} 条评论")
        except Exception as e:
            print(f"[ERROR] BERT分析失败: {e}")
            for review in reviews:
                review['sentiment'] = data_manager.sentiment_analyzer._rule_based_fallback(review.get('content', ''))
                review['confidence'] = 0.5
    else:
        for review in reviews:
            review['sentiment'] = 'neutral'
            review['confidence'] = 0.0

    sentiments = [r.get('sentiment', 'neutral') for r in reviews]
    counts = Counter(sentiments)
    total = len(reviews)

    positive_pct = counts.get('positive', 0) / total * 100 if total > 0 else 0
    neutral_pct = counts.get('neutral', 0) / total * 100 if total > 0 else 0
    negative_pct = counts.get('negative', 0) / total * 100 if total > 0 else 0

    ratings = [r.get('rating', 0) for r in reviews if r.get('rating', 0) > 0]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0

    confidences = [r.get('confidence', 0) for r in reviews if r.get('confidence', 0) > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [counts.get('positive', 0), counts.get('neutral', 0), counts.get('negative', 0)]
    labels = ['正面', '中性', '负面']
    colors = ['#48bb78', '#ed8936', '#f56565']
    explode = (0.05, 0, 0)

    if sum(sizes) > 0:
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                          autopct='%1.1f%%', shadow=True, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title(f'《{name}》情感分布 (BERT模型分析)', fontsize=14, fontweight='bold', pad=20)
    else:
        ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=ax.transAxes)

    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    sentiment_chart = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    reviews_for_frontend = []
    for review in reviews[:100]:
        sentiment_display = review.get('sentiment', 'neutral')
        sentiment_cn = {
            'positive': '正面',
            'neutral': '中性',
            'negative': '负面'
        }.get(sentiment_display, '中性')

        reviews_for_frontend.append({
            'user': review.get('user', '匿名用户'),
            'rating': review.get('rating', 3),
            'content': review.get('content', ''),
            'time': review.get('time', ''),
            'sentiment': sentiment_display,
            'sentiment_cn': sentiment_cn,
            'confidence': review.get('confidence', 0.5)
        })

    return jsonify({
        'success': True,
        'name': name,
        'total': total,
        'positive': counts.get('positive', 0),
        'neutral': counts.get('neutral', 0),
        'negative': counts.get('negative', 0),
        'positive_pct': positive_pct,
        'neutral_pct': neutral_pct,
        'negative_pct': negative_pct,
        'avg_rating': round(avg_rating, 1),
        'avg_confidence': round(avg_confidence, 3),
        'sentiment_chart': sentiment_chart,
        'reviews': reviews_for_frontend
    })


@app.route('/api/get_rating_dist', methods=['POST'])
def api_get_rating_dist():
    source_type = request.json.get('source_type')
    name = request.json.get('name')

    if source_type == 'system':
        movie = next((m for m in data_manager.movies if m["title"] == name), None)
        reviews = data_manager.get_system_reviews(movie["id"]) if movie else []
    elif source_type == 'crawled':
        reviews = data_manager.get_crawled_reviews(name)
    else:
        reviews = data_manager.get_uploaded_reviews(name)

    if not reviews:
        return jsonify({'success': False, 'error': '没有找到评论数据'})

    ratings = [r.get("rating", 0) for r in reviews]
    star_counts = [0, 0, 0, 0, 0]
    for r in ratings:
        if 1 <= r <= 5:
            star_counts[r-1] += 1

    fig, ax = plt.subplots(figsize=(10, 6))

    # 使用文字标签（不包含emoji）
    labels = ['1星', '2星', '3星', '4星', '5星']

    bars = ax.bar(labels, star_counts,
                  color=['#ff6b6b', '#ffa07a', '#ffd966', '#98d98e', '#6bcf7f'],
                  edgecolor='white', linewidth=2)

    for bar, count in zip(bars, star_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('评分星级', fontsize=12, fontweight='bold')
    ax.set_ylabel('评论数量', fontsize=12, fontweight='bold')
    ax.set_title(f'《{name}》评分分布', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, max(star_counts) * 1.2 if star_counts else 10)
    ax.grid(axis='y', alpha=0.3)

    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    return jsonify({'success': True, 'image': image_base64})


@app.route('/api/get_trend_chart', methods=['POST'])
def api_get_trend_chart():
    source_type = request.json.get('source_type')
    name = request.json.get('name')

    if source_type == 'system':
        movie = next((m for m in data_manager.movies if m["title"] == name), None)
        reviews = data_manager.get_system_reviews(movie["id"]) if movie else []
    elif source_type == 'crawled':
        reviews = data_manager.get_crawled_reviews(name)
    else:
        reviews = data_manager.get_uploaded_reviews(name)

    if not reviews:
        return jsonify({'success': False, 'error': '没有找到评论数据'})

    date_sentiment = defaultdict(lambda: {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0})

    for r in reviews:
        time_str = r.get('time', '')
        if not time_str:
            continue
        date = time_str[:10] if len(time_str) >= 10 else time_str
        sentiment = r.get('sentiment', 'neutral')
        date_sentiment[date][sentiment] += 1
        date_sentiment[date]['total'] += 1

    if not date_sentiment:
        for i, r in enumerate(reviews):
            date = f"第{i+1}条"
            sentiment = r.get('sentiment', 'neutral')
            date_sentiment[date][sentiment] += 1
            date_sentiment[date]['total'] += 1

    sorted_dates = sorted(date_sentiment.keys())
    dates = []
    positive_pcts = []
    neutral_pcts = []
    negative_pcts = []

    for date in sorted_dates:
        data = date_sentiment[date]
        total = data['total']
        if total > 0:
            dates.append(date)
            positive_pcts.append(data['positive'] / total * 100)
            neutral_pcts.append(data['neutral'] / total * 100)
            negative_pcts.append(data['negative'] / total * 100)

    fig, ax = plt.subplots(figsize=(12, 6))

    if len(dates) > 15:
        step = len(dates) // 15
        display_dates = [dates[i] if i % step == 0 else '' for i in range(len(dates))]
    else:
        display_dates = dates

    x = range(len(dates))

    ax.fill_between(x, 0, positive_pcts, alpha=0.7, color='#48bb78', label='正面情感')
    ax.fill_between(x, positive_pcts, [positive_pcts[i] + neutral_pcts[i] for i in range(len(positive_pcts))],
                    alpha=0.7, color='#ed8936', label='中性情感')
    ax.fill_between(x, [positive_pcts[i] + neutral_pcts[i] for i in range(len(positive_pcts))],
                    [positive_pcts[i] + neutral_pcts[i] + negative_pcts[i] for i in range(len(positive_pcts))],
                    alpha=0.7, color='#f56565', label='负面情感')

    ax.set_xlabel('时间', fontsize=12, fontweight='bold')
    ax.set_ylabel('情感占比 (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'《{name}》情感趋势分析 (BERT模型分析)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(display_dates, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    return jsonify({'success': True, 'image': image_base64})


@app.route('/api/get_wordcloud', methods=['POST'])
def api_get_wordcloud():
    """生成评论词云图 - 优化版，增强停用词过滤"""
    source_type = request.json.get('source_type')
    name = request.json.get('name')

    if source_type == 'system':
        movie = next((m for m in data_manager.movies if m["title"] == name), None)
        reviews = data_manager.get_system_reviews(movie["id"]) if movie else []
    elif source_type == 'crawled':
        reviews = data_manager.get_crawled_reviews(name)
    else:
        reviews = data_manager.get_uploaded_reviews(name)

    if not reviews:
        return jsonify({'success': False, 'error': '没有找到评论数据'})

    all_text = ' '.join([r.get('content', '') for r in reviews])
    if not all_text.strip():
        return jsonify({'success': False, 'error': '评论内容为空'})

    words = jieba.cut(all_text)

    # 增强版停用词表
    stopwords = set([
        # 标点符号和虚词
        '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '个', '也', '这', '那',
        '到', '说', '为', '以', '吧', '吗', '呢', '啊', '哦', '呀', '啦', '哈', '嘿', '哎', '嗯', '额',
        '电影', '这部', '一部', '真的', '觉得', '感觉', '非常', '特别', '一个', '没有', '不是', '就是',
        '但是', '还是', '可以', '很', '太', '看', '让', '给', '会', '能', '都', '也', '有', '在', '还',
        '就', '与', '和', '对', '从', '被', '把', '它', '她', '他', '我们', '你们', '他们', '什么', '怎么',
        '为什么', '如何', '这样', '那样', '那么', '这么', '多么', '自己', '已经', '还是', '或者', '并且',
        '虽然', '但是', '因为', '所以', '如果', '那么', '然后', '接着', '最后', '终于', '已经', '正在',
        '刚刚', '一直', '总是', '经常', '有时', '从来', '根本', '完全', '绝对', '可能', '大概', '也许',
        '应该', '必须', '需要', '想要', '喜欢', '讨厌', '觉得', '认为', '希望', '期待', '失望', '惊喜',
        # 程度副词
        '很', '非常', '特别', '十分', '相当', '比较', '稍微', '有点', '有些', '一点儿', '一些',
        '超级', '极其', '无比', '格外', '分外', '尤其', '最为', '最', '更', '更加', '越发', '越来',
        # 电影评论常见无意义词
        '真的', '感觉', '觉得', '一部', '这部', '那个', '这种', '那种', '这样', '那样', '真的', '确实',
        '其实', '不过', '虽然', '然而', '因此', '所以', '而且', '并且', '或者', '还是', '等等', '例如',
        '比如', '还有', '以及', '此外', '另外', '总之', '总的来说', '看起来', '看起来', '看上去',
        '一部电影', '这部电影', '这个电影', '那个电影', '片子', '影片', '影视', '作品', '剧情', '情节',
        '故事', '内容', '画面', '镜头', '特效', '演技', '表演', '演员', '角色', '人物', '导演', '编剧',
        '音乐', '配乐', '音效', '剪辑', '节奏', '场面', '氛围', '风格', '主题', '情感', '表达', '表现',
        '呈现', '展示', '体现', '反映', '描写', '讲述', '叙述', '展开', '发展', '推进', '转折', '高潮',
        '结尾', '结局', '开始', '开头', '前面', '后面', '中间', '部分', '段落', '细节', '元素', '方面',
        '方面', '程度', '角度', '视角', '眼光', '看法', '观点', '评价', '印象', '感受', '体验', '经历',
        '享受', '欣赏', '品味', '体会', '领悟', '理解', '认知', '思考', '反思', '回味', '难忘', '深刻',
        '震撼', '感动', '激动', '兴奋', '紧张', '刺激', '有趣', '无聊', '枯燥', '乏味', '精彩', '出色',
        '优秀', '卓越', '完美', '经典', '神作', '佳作', '烂片', '失败', '糟糕', '失望', '满意', '喜欢',
        '热爱', '钟爱', '痴迷', '沉醉', '沉浸', '投入', '专注', '集中', '吸引', '诱惑', '魅力', '特色',
        '特点', '亮点', '看点', '优点', '缺点', '优势', '劣势', '长处', '短处', '好处', '坏处', '价值',
        '意义', '内涵', '深度', '广度', '厚度', '温度', '热度', '口碑', '评分', '评价', '反馈', '回应',
        # 英文常见词
        'the', 'and', 'for', 'with', 'this', 'that', 'have', 'are', 'was', 'were', 'been', 'but', 'not',
        'you', 'your', 'our', 'their', 'will', 'can', 'could', 'would', 'should', 'may', 'might', 'must',
        'very', 'really', 'quite', 'rather', 'some', 'any', 'many', 'more', 'most', 'such', 'than', 'then',
        'now', 'then', 'when', 'where', 'there', 'here', 'which', 'what', 'who', 'whom', 'whose', 'why',
        'how', 'also', 'too', 'just', 'only', 'even', 'ever', 'never', 'always', 'often', 'sometimes',
        'usually', 'generally', 'basically', 'actually', 'literally', 'seriously', 'honestly', 'anyway'
    ])

    # 添加更精确的停用词（常用短词）
    common_short_words = set([
        '还是', '就是', '只是', '只有', '只要', '除了', '而且', '并且', '然而', '虽然', '但是', '因为',
        '所以', '如果', '的话', '的时候', '的地方', '的人', '的事', '的东西', '的话说', '可以看到',
        '能够', '足以', '是否', '能否', '会不会', '要不要', '能不能', '行不行', '好不好', '对不对'
    ])
    stopwords.update(common_short_words)

    # 只保留有意义的词（长度>=2且不是纯数字/标点）
    filtered_words = []
    for word in words:
        word = word.strip()
        # 过滤条件
        if len(word) < 2:
            continue
        if word.isdigit():
            continue
        if re.match(r'^[a-zA-Z]+$', word) and len(word) < 3:  # 过滤短英文单词
            continue
        if re.match(r'^[^\u4e00-\u9fa5]+$', word) and len(word) < 4:  # 过滤短非中文
            continue
        if word in stopwords:
            continue
        # 过滤纯标点符号
        if re.match(r'^[，。！？；：""''、\s,.;!?()（）【】《》<>~～@#￥%……&*]+$', word):
            continue
        filtered_words.append(word)

    if not filtered_words:
        return jsonify({'success': False, 'error': '没有足够的词汇生成词云'})

    word_freq = Counter(filtered_words)

    # 过滤掉频率过低的词（至少出现2次）
    word_freq = {word: count for word, count in word_freq.items() if count >= 2}

    if not word_freq:
        return jsonify({'success': False, 'error': '没有足够的词汇生成词云'})

    # 取前80个高频词
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:80])

    if WORDCLOUD_AVAILABLE:
        try:
            font_path = None
            possible_fonts = [
                'C:/Windows/Fonts/simhei.ttf',
                'C:/Windows/Fonts/msyh.ttc',
                'C:/Windows/Fonts/msyhbd.ttc',
                '/System/Library/Fonts/PingFang.ttc',
                '/System/Library/Fonts/STHeiti Light.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/truetype/arphic/uming.ttc',
                'simhei.ttf',
                'msyh.ttc'
            ]
            for font in possible_fonts:
                if os.path.exists(font):
                    font_path = font
                    break

            # 创建词云，优化参数
            wc = WordCloud(
                width=900,
                height=600,
                background_color='white',
                font_path=font_path,
                max_words=80,
                relative_scaling=0.5,
                colormap='viridis',
                random_state=42,
                prefer_horizontal=0.7,  # 70%水平排列
                scale=2,  # 提高分辨率
                contour_width=0.5,
                contour_color='#667eea'
            )
            wc.generate_from_frequencies(top_words)

            buffer = BytesIO()
            wc.to_image().save(buffer, format='PNG')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return jsonify({'success': True, 'image': image_base64})

        except Exception as e:
            print(f"生成词云失败: {e}")
            return _generate_bar_chart_fallback(top_words, name)
    else:
        print("wordcloud库未安装，使用柱状图替代")
        return _generate_bar_chart_fallback(top_words, name)


def _generate_bar_chart_fallback(word_freq, name):
    """生成柱状图作为词云的备选方案 - 优化版"""
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]

    fig, ax = plt.subplots(figsize=(12, 8))
    words_top = [w[0] for w in top_words]
    counts_top = [w[1] for w in top_words]

    # 使用渐变色
    colors = plt.cm.viridis([i/len(words_top) for i in range(len(words_top))])
    bars = ax.barh(words_top, counts_top, color=colors, edgecolor='white', linewidth=1)

    ax.set_xlabel('词频', fontsize=12, fontweight='bold')
    ax.set_title(f'《{name}》高频词汇 Top 20', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    for bar, count in zip(bars, counts_top):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, str(count),
                ha='left', va='center', fontsize=10, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ccc')
    ax.spines['bottom'].set_color('#ccc')
    ax.set_facecolor('#f8f9fa')

    plt.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    return jsonify({'success': True, 'image': image_base64, 'fallback': True})


# ==================== 用户相关API ====================
@app.route('/api/user/info')
def api_user_info():
    user = data_manager.users.get(data_manager.current_user, {"watched": []})
    watched_movies = [m for m in data_manager.movies if m["id"] in user["watched"]]
    return jsonify({'username': data_manager.current_user, 'watched_count': len(watched_movies), 'watched_movies': watched_movies})


@app.route('/api/user/login', methods=['POST'])
def api_user_login():
    username = request.json.get('username')
    if username in data_manager.users:
        data_manager.current_user = username
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': '用户不存在'})


@app.route('/api/user/register', methods=['POST'])
def api_user_register():
    username = request.json.get('username')
    if username not in data_manager.users:
        data_manager.users[username] = {"watched": [], "ratings": {}, "favorites": [], "watchlist": [], "user_info": {}}
        data_manager.current_user = username
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': '用户已存在'})


@app.route('/api/user/add_record', methods=['POST'])
def api_user_add_record():
    movie_title = request.json.get('movie_title')
    rating = int(request.json.get('rating', 8))
    movie = next((m for m in data_manager.movies if m["title"] == movie_title), None)
    if not movie:
        return jsonify({'success': False, 'error': '电影不存在'})
    user = data_manager.users[data_manager.current_user]
    if movie["id"] not in user["watched"]:
        user["watched"].append(movie["id"])
    user["ratings"][movie["id"]] = rating
    return jsonify({'success': True})


@app.route('/api/upload_user_data', methods=['POST'])
def api_upload_user_data():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有文件'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '文件名为空'})
    result = data_manager.upload_user_data(file)
    return jsonify(result)


@app.route('/api/get_user_data_list', methods=['GET'])
def api_get_user_data_list():
    data_list = []
    for data_id, data in data_manager.uploaded_user_data.items():
        data_list.append({
            'data_id': data_id, 'filename': data['filename'], 'file_type': data.get('file_type', '.xlsx'),
            'upload_time': data['upload_time'], 'total_users': data['total_users'],
            'total_movies': data.get('total_movies', 0), 'format': data.get('format', 'unknown')
        })
    return jsonify({'success': True, 'data_list': data_list})


@app.route('/api/analyze_user_data', methods=['POST'])
def api_analyze_user_data():
    data = request.json
    data_id = data.get('data_id')
    if not data_id:
        return jsonify({'success': False, 'error': '请提供数据ID'})
    result = data_manager.analyze_user_data(data_id)
    if result:
        return jsonify(result)
    return jsonify({'success': False, 'error': '数据不存在'})


@app.route('/api/get_user_data_preview', methods=['POST'])
def api_get_user_data_preview():
    data = request.json
    data_id = data.get('data_id')
    page = data.get('page', 1)
    page_size = data.get('page_size', 20)

    # 确保是整数
    try:
        page = int(page)
        page_size = int(page_size)
    except:
        page = 1
        page_size = 20

    if not data_id or data_id not in data_manager.uploaded_user_data:
        return jsonify({'success': False, 'error': '数据不存在'})

    user_data = data_manager.uploaded_user_data[data_id]
    raw_records = user_data.get('raw_records', [])
    total_movies = len(raw_records)

    print(f"[DEBUG] 总记录数: {total_movies}, page: {page}, page_size: {page_size}")

    # 计算分页
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_movies)

    # 边界检查
    if start_idx >= total_movies and total_movies > 0:
        start_idx = max(0, total_movies - page_size)
        end_idx = total_movies
        page = (total_movies + page_size - 1) // page_size

    paginated_records = raw_records[start_idx:end_idx]
    total_pages = (total_movies + page_size - 1) // page_size if page_size > 0 else 1

    print(f"[DEBUG] 返回数据量: {len(paginated_records)}")

    preview_data = []
    for r in paginated_records:
        watch_date = r.get('watch_date', '')
        if watch_date and isinstance(watch_date, str):
            if ' ' in watch_date:
                watch_date = watch_date.split(' ')[0]
            if 'T' in watch_date:
                watch_date = watch_date.split('T')[0]

        movie_rating = r.get('movie_rating', '')
        if movie_rating:
            try:
                movie_rating = float(movie_rating)
            except:
                pass

        preview_data.append({
            'movie_name': r.get('movie_name', '-'),
            'watch_date': watch_date or '-',
            'movie_type': r.get('movie_type', '-'),
            'movie_rating': movie_rating,
            'director': r.get('director', '-'),
            'actors': r.get('actors', '-'),
            'watch_channel': r.get('watch_channel', '-'),
            'remark': r.get('remark', '-'),
            'nickname': r.get('nickname', '-'),
            'gender': r.get('gender', '-'),
            'age': r.get('age', '-')
        })

    return jsonify({
        'success': True,
        'data_id': data_id,
        'filename': user_data['filename'],
        'file_type': user_data.get('file_type', '.xlsx'),
        'upload_time': user_data['upload_time'],
        'total_users': user_data['total_users'],
        'total_movies': total_movies,
        'format': user_data.get('format', 'unknown'),
        'preview': preview_data,
        'current_page': page,
        'page_size': page_size,
        'total_pages': total_pages
    })


@app.route('/api/get_genre_preferences', methods=['POST'])
def api_get_genre_preferences_v2():
    """获取用户类型偏好（用于盲盒功能）"""
    data = request.json
    data_id = data.get('data_id')

    if not data_id or data_id not in data_manager.uploaded_user_data:
        return jsonify({
            'success': True,
            'has_data': False,
            'top_genres': [],
            'genre_movies': {
                "剧情": ["肖申克的救赎", "霸王别姬", "阿甘正传", "我不是药神", "绿皮书"],
                "科幻": ["星际穿越", "盗梦空间", "流浪地球", "黑客帝国", "阿凡达"],
                "动作": ["让子弹飞", "速度与激情", "碟中谍", "叶问", "战狼2"],
                "爱情": ["泰坦尼克号", "怦然心动", "你的名字", "情书", "爱乐之城"],
                "喜剧": ["三傻大闹宝莱坞", "夏洛特烦恼", "唐人街探案", "疯狂的石头", "西虹市首富"],
                "动画": ["千与千寻", "龙猫", "疯狂动物城", "寻梦环游记", "哪吒之魔童降世"],
                "悬疑": ["盗梦空间", "看不见的客人", "消失的爱人", "调音师", "心迷宫"]
            }
        })

    result = data_manager.get_user_genre_preferences(data_id)
    if result:
        return jsonify(result)
    return jsonify({'success': False, 'error': '数据不存在'})


@app.route('/api/fetch_movie_poster', methods=['POST'])
def api_fetch_movie_poster():
    """实时爬取电影海报"""
    movie_name = request.json.get('movie_name', '')

    if not movie_name:
        return jsonify({'success': False, 'error': '电影名称不能为空'})

    try:
        if data_manager.spider:
            movie_id, title = data_manager.spider.search_movie_id(movie_name)
            if movie_id:
                detail = data_manager.spider.get_movie_detail(movie_id)
                if detail and detail.get('poster'):
                    local_poster = data_manager.spider.download_poster(
                        detail['poster'],
                        title or movie_name,
                        movie_id
                    )
                    return jsonify({
                        'success': True,
                        'poster_url': detail['poster'],
                        'local_poster': local_poster,
                        'movie_id': movie_id,
                        'title': title or movie_name
                    })

        return jsonify({'success': False, 'error': '未找到海报'})

    except Exception as e:
        print(f"获取海报失败: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/bert_model_info', methods=['GET'])
def api_bert_model_info():
    """获取BERT模型信息"""
    model_info = data_manager.sentiment_analyzer.get_model_info()
    return jsonify({
        'success': True,
        'model_info': model_info,
        'model_type': 'BERT-base-chinese',
        'description': '使用Google BERT预训练模型微调的情感分析器'
    })


@app.route('/api/get_uploaded_user_info', methods=['POST'])
def api_get_uploaded_user_info():
    """获取上传数据中的用户信息（第一个用户或指定用户）"""
    data = request.json
    data_id = data.get('data_id')
    nickname = data.get('nickname', '')

    if not data_id or data_id not in data_manager.uploaded_user_data:
        return jsonify({'success': False, 'error': '数据不存在'})

    user_data = data_manager.uploaded_user_data[data_id]
    records = user_data.get('records', [])
    raw_records = user_data.get('raw_records', [])

    if not records:
        return jsonify({'success': False, 'error': '没有用户数据'})

    # 如果指定了昵称，查找该用户
    target_user = None
    if nickname:
        for record in records:
            if record.get('nickname') == nickname:
                target_user = record
                break

    # 否则取第一个用户
    if not target_user and records:
        target_user = records[0]

    if not target_user:
        return jsonify({'success': False, 'error': '用户不存在'})

    # 获取该用户的观影记录
    user_movies = []
    for r in raw_records:
        if r.get('nickname') == target_user.get('nickname'):
            user_movies.append({
                'movie_name': r.get('movie_name', ''),
                'movie_type': r.get('movie_type', ''),
                'movie_rating': r.get('movie_rating', ''),
                'watch_date': r.get('watch_date', '')
            })

    return jsonify({
        'success': True,
        'user_info': {
            'nickname': target_user.get('nickname', '未知'),
            'gender': target_user.get('gender', '未知'),
            'age': target_user.get('age', 0),
            'watch_count': target_user.get('watch_count', len(user_movies)),
            'top_genres': target_user.get('top_genres', {}),
            'avg_rating': target_user.get('avg_rating', 0),
            'movies': user_movies[:10]
        },
        'total_users': len(records),
        'data_id': data_id
    })


@app.route('/api/get_all_users_from_data', methods=['POST'])
def api_get_all_users_from_data():
    """获取上传数据中的所有用户列表"""
    data = request.json
    data_id = data.get('data_id')

    if not data_id or data_id not in data_manager.uploaded_user_data:
        return jsonify({'success': False, 'error': '数据不存在'})

    user_data = data_manager.uploaded_user_data[data_id]
    records = user_data.get('records', [])
    raw_records = user_data.get('raw_records', [])

    users = []
    for record in records:
        # 统计该用户的观影数量
        user_movies_count = 0
        user_ratings = []
        for r in raw_records:
            if r.get('nickname') == record.get('nickname'):
                user_movies_count += 1
                if r.get('movie_rating'):
                    try:
                        user_ratings.append(float(r.get('movie_rating')))
                    except:
                        pass

        avg_rating = round(sum(user_ratings) / len(user_ratings), 1) if user_ratings else 0

        users.append({
            'nickname': record.get('nickname', '未知'),
            'gender': record.get('gender', '未知'),
            'age': record.get('age', 0),
            'watch_count': record.get('watch_count', user_movies_count),
            'avg_rating': avg_rating,
            'top_genres': record.get('top_genres', {})
        })

    return jsonify({
        'success': True,
        'users': users,
        'total_users': len(users),
        'data_id': data_id
    })


@app.route('/api/recommend_by_uploaded_data_with_user', methods=['POST'])
def api_recommend_by_uploaded_data_with_user():
    """基于上传的用户数据文件进行推荐（支持指定用户）"""
    data = request.json
    data_id = data.get('data_id')
    nickname = data.get('nickname')
    top_n = int(data.get('top_n', 8))

    if not data_id or data_id not in data_manager.uploaded_user_data:
        return jsonify({'success': False, 'error': '请先选择有效的上传数据'})

    user_data = data_manager.uploaded_user_data[data_id]
    records = user_data.get('records', [])
    raw_records = user_data.get('raw_records', [])

    # 找到指定用户的记录
    target_record = None
    for record in records:
        if record.get('nickname') == nickname:
            target_record = record
            break

    if not target_record:
        return jsonify({'success': False, 'error': f'未找到用户: {nickname}'})

    # 获取该用户的观影记录
    user_movies = []
    for r in raw_records:
        if r.get('nickname') == nickname:
            user_movies.append(r.get('movie_name', ''))

    # 构建推荐
    all_movies = []
    for title, info in LOCAL_MOVIE_CACHE.items():
        all_movies.append({
            'id': info.get('id', title),
            'title': title,
            'director': info.get('director', '未知'),
            'actors': info.get('actors', []),
            'genre': info.get('genre', '未知'),
            'year': int(info.get('year', 2000)),
            'rating': float(info.get('rating', 8.0)),
            'description': info.get('description', '')
        })

    # 过滤已看过的电影
    watched_set = set(user_movies)
    candidate_movies = [m for m in all_movies if m['title'] not in watched_set]

    # 基于用户偏好排序
    top_genres = target_record.get('top_genres', {})
    recommendations = []
    for movie in candidate_movies:
        movie_genres = movie.get('genre', '').split('/')
        genre_score = 0
        for genre in movie_genres:
            genre = genre.strip()
            if genre in top_genres:
                genre_score += top_genres.get(genre, 0)

        base_score = movie.get('rating', 8.0) / 10
        final_score = base_score * 0.5 + min(genre_score / 10, 0.5) * 0.5

        recommendations.append({
            **movie,
            'recommendation_score': round(final_score * 5, 2),
            'match_reason': f'基于用户 "{nickname}" 的观影偏好推荐 | 类型匹配度: {round(genre_score, 1)}'
        })

    recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)

    return jsonify({
        'success': True,
        'recommendations': recommendations[:top_n],
        'method': 'MCP协议分析 + 用户偏好推荐',
        'data_id': data_id,
        'nickname': nickname,
        'total_recommendations': len(recommendations[:top_n])
    })


@app.route('/api/recommend_by_crawler', methods=['POST'])
def api_recommend_by_crawler():
    """基于豆瓣爬虫的实时电影推荐 - 经过MCP三层架构"""
    data = request.json
    data_id = data.get('data_id')
    nickname = data.get('nickname')
    top_n = int(data.get('top_n', 8))

    if not data_id or data_id not in data_manager.uploaded_user_data:
        return jsonify({'success': False, 'error': '请先选择有效的上传数据'})

    user_data = data_manager.uploaded_user_data[data_id]
    records = user_data.get('records', [])
    raw_records = user_data.get('raw_records', [])

    # 找到指定用户的记录
    target_record = None
    for record in records:
        if record.get('nickname') == nickname:
            target_record = record
            break

    if not target_record:
        return jsonify({'success': False, 'error': f'未找到用户: {nickname}'})

    # 获取用户偏好的类型
    top_genres = target_record.get('top_genres', {})
    if not top_genres:
        return jsonify({'success': False, 'error': '无法获取用户类型偏好'})

    # 获取用户最喜欢的类型（权重最高的）
    favorite_genres = sorted(top_genres.items(), key=lambda x: x[1], reverse=True)
    main_genre = favorite_genres[0][0] if favorite_genres else '剧情'

    # 获取用户已看过的电影
    watched_movies = set()
    for r in raw_records:
        if r.get('nickname') == nickname:
            movie_name = r.get('movie_name', '')
            if movie_name:
                watched_movies.add(movie_name)

    # ========== 第一步：MCP上下文管理 ==========
    print(f"\n{'='*60}")
    print(f"🕷️ MCP爬虫推荐引擎启动")
    print(f"👤 用户: {nickname}")
    print(f"🎭 偏好类型: {main_genre}")
    print(f"{'='*60}")

    # 更新MCP用户上下文
    all_movies_local = []
    for title, info in LOCAL_MOVIE_CACHE.items():
        all_movies_local.append({
            'id': info.get('id', title),
            'title': title,
            'director': info.get('director', '未知'),
            'actors': info.get('actors', []),
            'genre': info.get('genre', '未知'),
            'year': int(info.get('year', 2000)),
            'rating': float(info.get('rating', 8.0)),
            'description': info.get('description', '')
        })

    watched_movies_info = [m for m in all_movies_local if m['title'] in watched_movies]
    user_context = data_manager.deep_recommender.mcp_context.update_user_context(
        nickname, watched_movies_info
    )
    print(f"📝 MCP用户上下文已更新 | 偏好: {list(user_context.get('preference_vector', {}).keys())}")

    # ========== 第二步：召回层 - 使用现有爬虫获取候选电影 ==========
    print(f"🕷️ 召回层: 使用豆瓣爬虫获取 {main_genre} 类型电影...")
    crawled_movies = crawl_movies_using_spider(main_genre, top_n * 3)

    if not crawled_movies:
        print(f"⚠️ 豆瓣爬取失败，使用本地电影库")
        crawled_movies = get_local_movies_by_genre(main_genre, top_n * 3)

    # 过滤已看过的电影
    candidate_movies = [m for m in crawled_movies if m.get('title') not in watched_movies]
    print(f"🔍 召回层: 获得 {len(candidate_movies)} 个候选电影")

    # ========== 第三步：排序层 - NCF深度排序 ==========
    print(f"🎯 排序层: NCF深度排序中...")

    # 将爬取的电影转换为推荐引擎可识别的格式
    formatted_movies = []
    for i, movie in enumerate(candidate_movies):
        formatted_movies.append({
            'id': movie.get('id', f'crawled_{i}'),
            'title': movie.get('title', '未知'),
            'director': movie.get('director', '未知'),
            'actors': movie.get('actors', ['未知']),
            'genre': movie.get('genre', main_genre),
            'year': movie.get('year', 2024),
            'rating': movie.get('rating', 8.0),
            'description': movie.get('description', f'豆瓣高分{main_genre}电影')
        })

    # 使用MCP推荐引擎的排序层进行精排
    if data_manager.deep_recommender.is_trained and nickname in data_manager.deep_recommender.user2id:
        # 使用NCF模型排序
        ranked_results = data_manager.deep_recommender.rank_by_model(nickname, formatted_movies)
        sorted_movies = [movie for movie, score in ranked_results]
    else:
        # 降级：使用综合分数排序
        for movie in formatted_movies:
            # 计算综合分数：豆瓣评分 + 类型匹配度
            genre_match_score = 0
            movie_genres = movie.get('genre', '').split('/')
            for g in movie_genres:
                g = g.strip()
                if g in top_genres:
                    genre_match_score += top_genres.get(g, 0)

            base_score = movie.get('rating', 8.0) / 10
            final_score = base_score * 0.6 + min(genre_match_score / 10, 0.4)
            movie['recommendation_score'] = round(final_score * 5, 2)

        sorted_movies = sorted(formatted_movies, key=lambda x: x.get('recommendation_score', 0), reverse=True)

    # 添加推荐理由
    recommendations = []
    for movie in sorted_movies[:top_n]:
        # 计算类型匹配度
        match_score = 0
        movie_genres = movie.get('genre', '').split('/')
        for g in movie_genres:
            g = g.strip()
            if g in top_genres:
                match_score += top_genres.get(g, 0)

        match_percent = min(int(match_score * 20), 100)
        recommendations.append({
            **movie,
            'recommendation_score': movie.get('recommendation_score', movie.get('rating', 8.0)),
            'match_reason': f'📡 MCP协议分析 | 类型匹配度: {match_percent}% | 基于您的{main_genre}偏好推荐',
            'source': movie.get('source', '🕷️ 豆瓣爬虫')
        })

    print(f"✅ MCP爬虫推荐完成，共 {len(recommendations)} 部电影")
    print(f"{'='*60}\n")

    return jsonify({
        'success': True,
        'recommendations': recommendations,
        'method': f'🕷️ MCP爬虫推荐 | 上下文管理 + 豆瓣爬虫召回 + NCF排序 | 偏好类型: {main_genre}',
        'nickname': nickname,
        'genre': main_genre,
        'total_recommendations': len(recommendations),
        'crawled_count': len(crawled_movies)
    })


def crawl_movies_using_spider(genre, limit=20):
    """使用现有的DoubanSpider爬虫获取电影"""
    movies = []

    if not data_manager.spider:
        print("[爬虫] Spider未初始化")
        return []

    try:
        # 类型映射的搜索关键词
        genre_search_keywords = {
            '剧情': '剧情 高分', '喜剧': '喜剧 高分', '动作': '动作 高分', '爱情': '爱情 高分',
            '科幻': '科幻 高分', '动画': '动画 高分', '悬疑': '悬疑 高分', '惊悚': '惊悚 高分',
            '恐怖': '恐怖 高分', '纪录片': '纪录片 高分', '奇幻': '奇幻 高分', '冒险': '冒险 高分'
        }

        search_keyword = genre_search_keywords.get(genre, f'{genre} 电影')

        # 使用豆瓣搜索API获取电影列表
        search_url = f'https://movie.douban.com/j/search_subjects?type=movie&tag={genre}&sort=recommend&page_limit={limit}&page_start=0'

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://movie.douban.com/',
            'Accept': 'application/json, text/plain, */*'
        }

        print(f"[爬虫] 正在从豆瓣搜索 {genre} 类型电影...")
        response = requests.get(search_url, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json()
            subjects = data.get('subjects', [])

            print(f"[爬虫] 获取到 {len(subjects)} 部电影")

            for subject in subjects[:limit]:
                movie_id = subject.get('id')
                title = subject.get('title')
                rate = subject.get('rate', '8.0')
                cover = subject.get('cover', '')

                # 使用现有的 get_movie_detail 方法获取详细信息
                detail = None
                if data_manager.spider:
                    try:
                        detail = data_manager.spider.get_movie_detail(movie_id)
                        time.sleep(0.3)  # 避免请求过快
                    except Exception as e:
                        print(f"[爬虫] 获取详情失败 {title}: {e}")

                # 解析年份
                year = '未知'
                if detail and detail.get('year'):
                    year = detail.get('year')
                elif subject.get('year'):
                    year = subject.get('year')

                # 解析导演
                director = '未知'
                if detail and detail.get('director'):
                    director = detail.get('director')

                # 解析演员
                actors = ['推荐']
                if detail and detail.get('actors'):
                    actors = detail.get('actors')[:3]

                # 解析简介
                description = detail.get('summary', f'豆瓣高分{genre}电影，评分{rate}分')[:150] if detail else f'豆瓣高分{genre}电影，值得一看'

                # 解析评分
                rating = float(rate) if rate and rate != '暂无' else 8.0
                if detail and detail.get('rating') and detail.get('rating') != '暂无':
                    try:
                        rating = float(detail.get('rating'))
                    except:
                        pass

                movies.append({
                    'id': movie_id,
                    'title': title,
                    'rating': rating,
                    'director': director,
                    'actors': actors,
                    'genre': genre,
                    'year': year,
                    'description': description,
                    'poster': cover,
                    'source': '🕷️ 豆瓣爬虫'
                })

                print(f"[爬虫] 成功获取: {title} - {rating}分")

        return movies

    except Exception as e:
        print(f"[爬虫] 爬取失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_local_movies_by_genre(genre, limit=10):
    """从本地缓存获取指定类型的电影（备用）"""
    movies = []

    genre_movies = {
        '剧情': ['肖申克的救赎', '霸王别姬', '阿甘正传', '美丽人生', '放牛班的春天', '绿皮书', '我不是药神', '海上钢琴师', '忠犬八公的故事'],
        '科幻': ['星际穿越', '盗梦空间', '流浪地球', '黑客帝国', '阿凡达', '2001太空漫游', '银翼杀手2049', '降临'],
        '动作': ['让子弹飞', '速度与激情', '碟中谍', '战狼2', '红海行动', '叶问', '复仇者联盟', '疯狂的麦克斯'],
        '爱情': ['泰坦尼克号', '怦然心动', '你的名字', '情书', '爱乐之城', '初恋这件小事', '罗马假日', '情书'],
        '喜剧': ['三傻大闹宝莱坞', '夏洛特烦恼', '唐人街探案', '疯狂的石头', '西虹市首富', '人在囧途', '大话西游'],
        '动画': ['千与千寻', '龙猫', '疯狂动物城', '寻梦环游记', '冰雪奇缘', '你的名字', '机器人总动员', '飞屋环游记'],
        '悬疑': ['盗梦空间', '看不见的客人', '消失的爱人', '调音师', '心迷宫', '误杀', '致命ID', '恐怖游轮'],
        '奇幻': ['哈利波特', '指环王', '加勒比海盗', '纳尼亚传奇', '大鱼海棠', '神奇动物在哪里', '潘神的迷宫']
    }

    movie_names = genre_movies.get(genre, genre_movies['剧情'])

    for name in movie_names[:limit]:
        if name in LOCAL_MOVIE_CACHE:
            info = LOCAL_MOVIE_CACHE[name]
            movies.append({
                'id': info.get('id', name),
                'title': name,
                'rating': float(info.get('rating', 8.0)),
                'director': info.get('director', '未知'),
                'actors': info.get('actors', ['未知']),
                'genre': genre,
                'year': info.get('year', '未知'),
                'description': info.get('description', '经典电影推荐'),
                'poster': info.get('poster', ''),
                'source': '📦 本地缓存'
            })

    return movies


@app.route('/api/recommend_by_douban_mcp', methods=['POST'])
def api_recommend_by_douban_mcp():
    """
    豆瓣全网电影推荐 - MCP三层架构
    """
    data = request.json
    data_id = data.get('data_id')
    nickname = data.get('nickname')
    top_n = int(data.get('top_n', 8))

    if not data_id or data_id not in data_manager.uploaded_user_data:
        return jsonify({'success': False, 'error': '请先选择有效的上传数据'})

    user_data = data_manager.uploaded_user_data[data_id]
    records = user_data.get('records', [])
    raw_records = user_data.get('raw_records', [])

    # 找到指定用户的记录
    target_record = None
    for record in records:
        if record.get('nickname') == nickname:
            target_record = record
            break

    if not target_record:
        return jsonify({'success': False, 'error': f'未找到用户: {nickname}'})

    # ========== 第一步：MCP上下文管理 - 分析用户偏好 ==========
    print(f"\n{'='*60}")
    print(f"🎬 MCP豆瓣全网推荐引擎启动")
    print(f"👤 用户: {nickname}")
    print(f"{'='*60}")

    # 获取用户偏好的类型
    top_genres = target_record.get('top_genres', {})

    # 【调试】打印用户偏好
    print(f"📝 用户偏好类型: {top_genres}")

    # 如果 top_genres 为空，尝试从 raw_records 重新计算
    if not top_genres:
        print("⚠️ top_genres为空，重新计算用户偏好...")
        genre_counter = Counter()
        for r in raw_records:
            if r.get('nickname') == nickname:
                movie_type = r.get('movie_type', '')
                if movie_type and movie_type != '未知':
                    for gt in re.split('[/、,，]', movie_type):
                        gt = gt.strip()
                        if gt:
                            genre_counter[gt] += 1

        # 计算权重
        total = sum(genre_counter.values())
        if total > 0:
            top_genres = {g: c/total for g, c in genre_counter.items()}

        print(f"📝 重新计算的偏好: {top_genres}")

    if not top_genres:
        # 如果没有偏好，使用默认类型
        top_genres = {'剧情': 0.5, '喜剧': 0.3, '爱情': 0.2}
        print("⚠️ 使用默认偏好类型")

    # 获取用户最喜欢的类型
    favorite_genres = sorted(top_genres.items(), key=lambda x: x[1], reverse=True)
    print(f"📝 排序后偏好: {favorite_genres[:3]}")

    # 获取用户已看过的电影
    watched_movies = set()
    for r in raw_records:
        if r.get('nickname') == nickname:
            movie_name = r.get('movie_name', '')
            if movie_name:
                watched_movies.add(movie_name)
    print(f"📊 已看电影数: {len(watched_movies)}")

    # ========== 第二步：召回层 - 从豆瓣搜索候选电影 ==========
    print(f"🔍 召回层: 从豆瓣搜索候选电影...")

    all_candidates = []

    # 根据用户偏好类型搜索电影
    for genre, weight in favorite_genres[:3]:
        print(f"   搜索类型: {genre} (权重: {weight})")
        movies = search_douban_movies_by_genre(genre, limit=15)
        for movie in movies:
            movie['genre_weight'] = weight
            movie['matched_genre'] = genre
            all_candidates.append(movie)
        time.sleep(0.3)

    # 去重
    seen_titles = set()
    unique_candidates = []
    for movie in all_candidates:
        title = movie.get('title', '')
        if title and title not in seen_titles and title not in watched_movies:
            seen_titles.add(title)
            unique_candidates.append(movie)

    print(f"🔍 召回层: 共获得 {len(unique_candidates)} 个候选电影")

    if len(unique_candidates) < top_n:
        print(f"⚠️ 候选不足，补充本地热门电影")
        local_movies = get_local_hot_movies()
        for movie in local_movies:
            if movie.get('title') not in seen_titles:
                unique_candidates.append(movie)
                seen_titles.add(movie.get('title'))

    # ========== 第三步：排序层 ==========
    print(f"🎯 排序层: 计算推荐分数...")

    recommendations = []
    for movie in unique_candidates:
        title = movie.get('title')

        # 计算类型匹配度
        match_percent = 0
        matched_genre = movie.get('matched_genre', '')

        # 如果电影有匹配的类型，计算匹配度
        if matched_genre and matched_genre in top_genres:
            # 基础匹配度 = 该类型的权重
            match_percent = int(top_genres.get(matched_genre, 0) * 100)
        else:
            # 尝试从电影类型中匹配
            movie_genre = movie.get('genre', '')
            for genre, weight in top_genres.items():
                if genre in movie_genre:
                    match_percent = max(match_percent, int(weight * 100))

        # 豆瓣评分（9.7分对应高分）
        rating = movie.get('rating', 8.0)

        # 豆瓣评分（9.7分对应高分）
        rating = movie.get('rating', 8.0)

        # ========== 动态权重计算 ==========
        # 根据用户观影数量动态调整评分和类型匹配的权重
        user_watch_count = len(watched_movies)  # 需要提前定义

        if user_watch_count < 5:
            # 新用户（观影<5部）：评分优先
            weight_rating = 0.7
            weight_genre = 0.3
        elif user_watch_count < 20:
            # 中度用户（5-20部）：平衡权重
            weight_rating = 0.5
            weight_genre = 0.5
        else:
            # 重度用户（>20部）：偏好优先
            weight_rating = 0.3
            weight_genre = 0.7

        # 评分分：豆瓣评分/2 (范围0-5)
        score_rating = rating / 2
        # 类型分：匹配度百分比转分数 (范围0-5)
        score_genre = match_percent / 100 * 5

        # 最终分数 = 评分分 × 权重 + 类型分 × 权重
        final_score = (score_rating * weight_rating) + (score_genre * weight_genre)
        final_score = min(final_score, 5.0)  # 限制在1-5之间
        # ========== 动态权重计算结束 ==========

        recommendations.append({
            'id': movie.get('id'),
            'title': title,
            'director': movie.get('director', '未知'),
            'actors': movie.get('actors', ['推荐']),
            'genre': movie.get('genre', matched_genre),
            'year': movie.get('year', '经典'),
            'rating': rating,
            'description': movie.get('description', '')[:50],
            'poster': movie.get('poster', ''),
            'recommendation_score': round(final_score, 2),
            'match_reason': f'📡 MCP协议分析 | 类型匹配度: {match_percent}% | 豆瓣评分: {rating}分',
            'source': movie.get('source', '🕷️ 豆瓣实时')
        })

    # 按推荐分数排序
    recommendations.sort(key=lambda x: x.get('recommendation_score', 0), reverse=True)

    print(f"✅ 推荐完成，共 {len(recommendations[:top_n])} 部电影")
    for r in recommendations[:top_n]:
        print(f"   {r['title']}: {r['recommendation_score']}分 (匹配度: {r['match_reason'].split('%')[0].split(': ')[-1]}%)")
    print(f"{'='*60}\n")

    return jsonify({
        'success': True,
        'recommendations': recommendations[:top_n],
        'method': f'🎬 MCP豆瓣全网推荐 | 上下文分析 + 豆瓣召回 + 智能排序',
        'nickname': nickname,
        'total_candidates': len(unique_candidates),
        'total_recommendations': len(recommendations[:top_n])
    })


def search_douban_movies_by_genre(genre, limit=15, retry=3):
    """从豆瓣搜索指定类型的电影（带缓存、重试和限流）"""
    global LAST_REQUEST_TIME
    import time

    # 检查缓存
    cache_key = f"{genre}_{limit}"
    current_time = time.time()

    if cache_key in DOUBAN_SEARCH_CACHE:
        cached_data, cached_time = DOUBAN_SEARCH_CACHE[cache_key]
        if current_time - cached_time < CACHE_DURATION:
            print(f"   📦 使用缓存 ({genre})")
            return cached_data

    movies = []

    # 更真实的请求头
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]

    # 豆瓣API接口（更稳定）
    api_urls = [
        f'https://movie.douban.com/j/search_subjects?type=movie&tag={genre}&sort=recommend&page_limit={limit}&page_start=0',
        f'https://movie.douban.com/j/new_search_subjects?sort=U&range=0,10&tags={genre}&start=0&genres={genre}'
    ]

    for attempt in range(retry):
        try:
            # 限流：确保请求间隔
            elapsed = current_time - LAST_REQUEST_TIME
            wait_time = REQUEST_INTERVAL + random.uniform(0.5, 1.5)
            if elapsed < wait_time:
                time.sleep(wait_time - elapsed)

            # 随机选择User-Agent
            headers = {
                'User-Agent': random.choice(user_agents),
                'Referer': 'https://movie.douban.com/explore',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Host': 'movie.douban.com',
                'Cache-Control': 'max-age=0',
                'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"Windows"',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin'
            }

            # 添加Cookie（从浏览器复制真实Cookie更有效）
            cookies = {
                'll': '118254',
                'bid': 'BwcYcY_GcAA',
                'dbcl2': '293423119:3LWQOX7tv8w',
                'push_noty_num': '0',
                'push_doumail_num': '0',
                'ct': 'y',
                '_vwo_uuid_v2': 'DE05BC0B267794C027A0B55F121E447FB|67213e974038f3093b609c7cfb2d4998',
                '__utmz': '30149280.1774840531.8.6.utmcsr=search.douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/movie/subject_search',
                '__utma': '30149280.1808011285.1774349897.1774857365.1774865170.11',
                '__utmb': '30149280.0.10.1774865170',
                '__utmc': '30149280',
                '__utmv': '30149280.29342'
            }

            # 尝试主URL
            response = requests.get(api_urls[0], headers=headers, cookies=cookies, timeout=15)

            if response.status_code == 200:
                data = response.json()
                subjects = data.get('subjects', [])

                for subject in subjects:
                    movies.append({
                        'id': subject.get('id'),
                        'title': subject.get('title'),
                        'rating': float(subject.get('rate', '8.0')),
                        'director': subject.get('directors', ['豆瓣高分'])[0] if subject.get('directors') else '豆瓣高分',
                        'actors': subject.get('casts', ['推荐'])[:3] if subject.get('casts') else ['推荐'],
                        'genre': genre,
                        'year': subject.get('year', '未知'),
                        'description': f'豆瓣评分{subject.get("rate", "8.0")}分的{genre}电影',
                        'poster': subject.get('cover', ''),
                        'source': '豆瓣实时'
                    })

                print(f"   ✅ 成功获取 {len(movies)} 部{genre}电影")

                # 存入缓存
                DOUBAN_SEARCH_CACHE[cache_key] = (movies, time.time())
                return movies

            elif response.status_code == 403:
                print(f"   ⚠️ 豆瓣拒绝访问(403)，尝试 {attempt+1}/{retry}")
                # 等待更长时间
                time.sleep(random.uniform(3, 6))
                continue
            else:
                print(f"   ⚠️ 返回状态码: {response.status_code}")
                break

        except requests.exceptions.Timeout:
            print(f"   ⚠️ 请求超时 (尝试 {attempt+1}/{retry})")
            if attempt < retry - 1:
                time.sleep(random.uniform(2, 4))
                continue
        except Exception as e:
            print(f"   ❌ 请求失败 (尝试 {attempt+1}/{retry}): {e}")
            if attempt < retry - 1:
                time.sleep(random.uniform(2, 4))
                continue

    print(f"   ❌ 爬取失败，使用本地缓存")

    # 爬取失败，尝试返回过期缓存
    if cache_key in DOUBAN_SEARCH_CACHE:
        cached_data, _ = DOUBAN_SEARCH_CACHE[cache_key]
        print(f"   📦 使用过期缓存 ({len(cached_data)}部电影)")
        return cached_data

    return []


def get_local_hot_movies(limit=20):
    """获取本地热门电影作为备用"""
    movies = []

    hot_movies = [
        ('肖申克的救赎', 9.7, '剧情', '弗兰克·德拉邦特', '希望让人自由'),
        ('霸王别姬', 9.6, '剧情', '陈凯歌', '风华绝代'),
        ('阿甘正传', 9.5, '剧情', '罗伯特·泽米吉斯', '人生就像巧克力'),
        ('泰坦尼克号', 9.5, '爱情', '詹姆斯·卡梅隆', '永恒的爱情'),
        ('美丽人生', 9.5, '剧情', '罗伯托·贝尼尼', '最伟大的父爱'),
        ('盗梦空间', 9.4, '科幻', '克里斯托弗·诺兰', '梦境与现实'),
        ('星际穿越', 9.4, '科幻', '克里斯托弗·诺兰', '穿越时空的爱'),
        ('千与千寻', 9.4, '动画', '宫崎骏', '成长的旅程'),
        ('忠犬八公', 9.4, '剧情', '莱塞·霍尔斯道姆', '等待是最长情的告白'),
        ('放牛班的春天', 9.3, '剧情', '克里斯托夫·巴拉蒂', '音乐治愈心灵'),
        ('海上钢琴师', 9.3, '剧情', '朱塞佩·托纳多雷', '音乐与自由的传奇'),
        ('三傻大闹宝莱坞', 9.2, '喜剧', '拉吉库马尔·希拉尼', '追求卓越'),
        ('怦然心动', 9.1, '爱情', '罗伯·莱纳', '初恋的美好'),
        ('我不是药神', 9.0, '剧情', '文牧野', '现实题材的震撼之作'),
        ('让子弹飞', 9.0, '剧情', '姜文', '站着把钱挣了'),
        ('绿皮书', 8.9, '剧情', '彼得·法拉利', '跨越种族的友谊'),
        ('流浪地球', 7.9, '科幻', '郭帆', '中国科幻的里程碑'),
    ]

    for title, rating, genre, director, desc in hot_movies[:limit]:
        movies.append({
            'id': title,
            'title': title,
            'rating': rating,
            'director': director,
            'actors': ['推荐'],
            'genre': genre,
            'year': '经典',
            'description': desc,
            'poster': '',
            'source': '本地热门'
        })

    return movies


@app.route('/api/recommend_by_ai', methods=['POST'])
def api_recommend_by_ai():
    """
    基于AI（智谱AI）的电影推荐 - MCP协议调用AI工具
    使用GLM-4模型分析用户偏好并生成个性化推荐
    """
    import requests
    import json

    data = request.json
    data_id = data.get('data_id')
    nickname = data.get('nickname')
    top_n = int(data.get('top_n', 8))

    if not data_id or data_id not in data_manager.uploaded_user_data:
        return jsonify({'success': False, 'error': '请先选择有效的上传数据'})

    user_data = data_manager.uploaded_user_data[data_id]
    records = user_data.get('records', [])
    raw_records = user_data.get('raw_records', [])

    # 找到指定用户的记录
    target_record = None
    for record in records:
        if record.get('nickname') == nickname:
            target_record = record
            break

    if not target_record:
        return jsonify({'success': False, 'error': f'未找到用户: {nickname}'})

    # 获取用户观影历史
    user_movies = []
    user_genres = []
    user_ratings = []

    for r in raw_records:
        if r.get('nickname') == nickname:
            movie_name = r.get('movie_name', '')
            movie_type = r.get('movie_type', '')
            movie_rating = r.get('movie_rating', '')
            if movie_name:
                user_movies.append(movie_name)
            if movie_type and movie_type != '未知':
                user_genres.append(movie_type)
            if movie_rating:
                try:
                    user_ratings.append(float(movie_rating))
                except:
                    pass

    # 获取用户偏好类型
    top_genres = target_record.get('top_genres', {})
    if not top_genres and user_genres:
        from collections import Counter
        genre_counter = Counter(user_genres)
        total = sum(genre_counter.values())
        top_genres = {g: c/total for g, c in genre_counter.items()}

    # 获取用户已看过的电影
    watched_movies = set(user_movies)

    # 准备本地电影库作为候选
    all_candidate_movies = []
    for title, info in LOCAL_MOVIE_CACHE.items():
        if title not in watched_movies:
            all_candidate_movies.append({
                'title': title,
                'rating': float(info.get('rating', 8.0)),
                'director': info.get('director', '未知'),
                'genre': info.get('genre', '未知'),
                'year': info.get('year', '未知'),
                'description': info.get('description', '')[:100]
            })

    # 按评分排序取前30作为候选
    all_candidate_movies.sort(key=lambda x: x['rating'], reverse=True)
    candidate_movies = all_candidate_movies[:30]

    # ========== MCP调用AI工具进行智能推荐 ==========
    print(f"\n{'='*60}")
    print(f"🤖 MCP调用AI推荐引擎")
    print(f"👤 用户: {nickname}")
    print(f"📊 观影记录: {len(user_movies)}部")
    print(f"🎭 偏好类型: {list(top_genres.keys())[:3] if top_genres else '未知'}")
    print(f"{'='*60}")

    # 构建AI提示词
    user_profile = f"""用户昵称：{nickname}
观影数量：{len(user_movies)}部
平均评分：{round(sum(user_ratings)/len(user_ratings), 1) if user_ratings else '暂无'}
偏好类型：{', '.join(list(top_genres.keys())[:5]) if top_genres else '尚未明确'}
已看过的电影：{', '.join(list(watched_movies)[:15])}{'...' if len(watched_movies) > 15 else ''}

请根据以上用户画像，从以下候选电影中为用户推荐 {top_n} 部最合适的电影。候选电影列表：
{json.dumps(candidate_movies, ensure_ascii=False, indent=2)}

请以JSON格式返回推荐结果，格式如下：
[
    {{
        "title": "电影名称",
        "reason": "推荐理由（50字以内）",
        "expected_rating": 9.5
    }}
]
注意：只返回JSON数组，不要有其他文字。推荐理由要个性化，结合用户的观影偏好。"""

    # 调用智谱AI API
    api_key = 'c0bf09a1c26541dc832cb81a0d05ecd5.ykYT2oZzf93DIbRv'
    chat_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {
            "role": "system",
            "content": "你是一个专业的电影推荐AI助手，擅长根据用户偏好分析并推荐电影。请严格按照JSON格式返回推荐结果。"
        },
        {
            "role": "user",
            "content": user_profile
        }
    ]

    chat_data = {
        "model": "glm-4-flash",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000,
        "stream": False
    }

    recommendations = []

    try:
        response = requests.post(chat_url, headers=headers, json=chat_data, timeout=60)

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                ai_response = result['choices'][0]['message']['content']
                print(f"🤖 AI响应: {ai_response[:200]}...")

                # 解析AI返回的JSON
                try:
                    # 提取JSON部分
                    import re
                    json_match = re.search(r'\[[\s\S]*\]', ai_response)
                    if json_match:
                        ai_recommendations = json.loads(json_match.group())

                        # 匹配本地电影数据
                        for ai_rec in ai_recommendations[:top_n]:
                            movie_title = ai_rec.get('title', '')
                            # 查找匹配的本地电影
                            matched_movie = None
                            for movie in candidate_movies:
                                if movie['title'] == movie_title or movie_title in movie['title'] or movie['title'] in movie_title:
                                    matched_movie = movie
                                    break

                            if matched_movie:
                                recommendations.append({
                                    'title': matched_movie['title'],
                                    'rating': matched_movie['rating'],
                                    'director': matched_movie['director'],
                                    'genre': matched_movie['genre'],
                                    'year': matched_movie['year'],
                                    'description': matched_movie['description'],
                                    'recommendation_score': round(matched_movie['rating'] / 2, 2),
                                    'match_reason': f"🤖 AI智能分析 | {ai_rec.get('reason', '根据您的观影偏好推荐')}",
                                    'source': '🤖 AI推荐'
                                })
                            else:
                                # 如果没找到精确匹配，尝试模糊匹配
                                for movie in all_candidate_movies:
                                    if movie['title'] in movie_title or movie_title in movie['title']:
                                        recommendations.append({
                                            'title': movie['title'],
                                            'rating': movie['rating'],
                                            'director': movie['director'],
                                            'genre': movie['genre'],
                                            'year': movie['year'],
                                            'description': movie['description'],
                                            'recommendation_score': round(movie['rating'] / 2, 2),
                                            'match_reason': f"🤖 AI智能分析 | {ai_rec.get('reason', '根据您的观影偏好推荐')}",
                                            'source': '🤖 AI推荐'
                                        })
                                        break

                        print(f"✅ AI推荐成功，共 {len(recommendations)} 部电影")

                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON解析失败: {e}")
                    print(f"原始响应: {ai_response}")

        else:
            print(f"⚠️ AI API调用失败: {response.status_code}")

    except Exception as e:
        print(f"⚠️ AI推荐失败: {e}")
        import traceback
        traceback.print_exc()

    # 如果AI推荐失败或数量不足，使用备选推荐
    if len(recommendations) < top_n:
        print(f"⚠️ AI推荐数量不足({len(recommendations)}/{top_n})，使用备选推荐")

        # 按类型匹配度排序
        for movie in candidate_movies:
            if movie['title'] not in [r['title'] for r in recommendations]:
                # 计算类型匹配度
                match_score = 0
                movie_genres = movie.get('genre', '').split('/')
                for genre in movie_genres:
                    genre = genre.strip()
                    if genre in top_genres:
                        match_score += top_genres.get(genre, 0)

                # 综合分数 = 豆瓣评分 * 0.5 + 类型匹配 * 0.5
                rating_score = movie.get('rating', 8.0) / 10
                final_score = rating_score * 0.5 + match_score * 0.5

                recommendations.append({
                    'title': movie['title'],
                    'rating': movie.get('rating', 8.0),
                    'director': movie.get('director', '未知'),
                    'genre': movie.get('genre', '未知'),
                    'year': movie.get('year', '未知'),
                    'description': movie.get('description', ''),
                    'recommendation_score': round(final_score * 5, 2),
                    'match_reason': f"📡 MCP协议分析 | 类型匹配度: {round(match_score * 100)}% | 豆瓣评分: {movie.get('rating', 8.0)}分",
                    'source': '📡 MCP备选'
                })

        # 去重并按分数排序
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec['title'] not in seen:
                seen.add(rec['title'])
                unique_recs.append(rec)
        unique_recs.sort(key=lambda x: x.get('recommendation_score', 0), reverse=True)
        recommendations = unique_recs[:top_n]

    print(f"✅ MCP AI推荐完成，共 {len(recommendations)} 部电影")
    print(f"{'='*60}\n")

    return jsonify({
        'success': True,
        'recommendations': recommendations,
        'method': '🤖 MCP AI智能推荐 | 基于GLM-4大模型分析用户偏好',
        'nickname': nickname,
        'total_recommendations': len(recommendations),
        'ai_enabled': True
    })


if __name__ == '__main__':
    print("=" * 60)
    print("豆瓣电影数据分析与推荐系统 - BERT情感分析版 + FastMCP")
    print("=" * 60)
    print("页面结构: 首页 | 影视分析 | 推荐引擎 | 社交管理")
    print("支持格式: Excel (.xlsx, .xls) 和 CSV (.csv)")
    print("情感分析: BERT预训练模型 (bert-base-chinese)")
    print("推荐引擎: MCP模型上下文协议 + 召回层(向量检索) + 排序层(NCF)")
    print("MCP协议: FastMCP (基于Model Context Protocol)")
    print("=" * 60)
    print("访问地址: http://127.0.0.1:5000")
    print("=" * 60)
    print("首次运行会下载BERT模型（约400MB），请耐心等待...")
    print("=" * 60)
    print()
    print("💡 MCP服务器需要单独启动：")
    print("   在新终端中运行: python movie_mcp_server.py")
    print("=" * 60)

    # 初始化MCP（绑定数据管理器）
    init_mcp()

    app.run(debug=True, host='0.0.0.0', port=5000)