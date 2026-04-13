"""
数据模型模块 - MovieDataManager 类
"""
import os
import re
import random
import base64
from datetime import datetime
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import jieba

# 导入配置中的常量
from config import DOUBAN_REAL_REVIEWS, USER_NAMES, LOCAL_MOVIE_CACHE
from spider import fuzzy_match_movie
from recommend_engine import DeepRecommendationEngine
from bert_sentiment import get_sentiment_analyzer

# 爬虫相关
REQUESTS_AVAILABLE = True
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    REQUESTS_AVAILABLE = False
    BeautifulSoup = None

# 词云相关
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    WordCloud = None


class MovieDataManager:
    def __init__(self):
        self.movies = self._generate_movies()
        self.reviews = self._generate_reviews()
        self.users = self._generate_users()
        self.current_user = "user_001"
        self.crawled_reviews = {}
        self.uploaded_reviews = {}
        self.uploaded_user_data = {}

        # 使用微调版BERT情感分析器（会自动训练）
        print("\n" + "="*50)
        print("正在初始化BERT情感分析器（微调版）...")
        print("="*50)
        self.sentiment_analyzer = get_sentiment_analyzer()

        # 打印模型信息
        model_info = self.sentiment_analyzer.get_model_info()
        print(f"✅ BERT情感分析器已就绪")
        print(f"   模型类型: {model_info.get('model_type', 'fine_tuned_bert')}")
        print(f"   设备: {model_info.get('device', 'CPU')}")

        # 深度学习推荐引擎
        self.deep_recommender = DeepRecommendationEngine(embedding_dim=64)
        self.deep_recommender.set_data_manager(self)

        if os.path.exists('deep_recommend_model.pth'):
            self.deep_recommender.load_model()
        else:
            self._train_recommender()

        # 爬虫
        self.spider = None
        if REQUESTS_AVAILABLE:
            try:
                from spider import DoubanSpider
                self.spider = DoubanSpider()
                print("✅ 豆瓣爬虫初始化完成")
            except:
                pass

    def _train_recommender(self):
        """训练推荐模型"""
        ratings_data = []
        for user_id, user_info in self.users.items():
            for movie_id, rating in user_info.get('ratings', {}).items():
                ratings_data.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating / 2.0
                })

        if len(ratings_data) >= 10:
            print(f"📊 准备训练MCP模型数据: {len(ratings_data)} 条评分记录")
            self.deep_recommender.train(ratings_data, epochs=30, batch_size=16)
            self.deep_recommender.save_model()
        else:
            print(f"⚠️ 训练数据不足({len(ratings_data)}条)，使用基础推荐")

    def _generate_movies(self):
        return [
            {"id": 1, "title": "肖申克的救赎", "director": "弗兰克·德拉邦特", "actors": ["蒂姆·罗宾斯", "摩根·弗里曼"], "genre": "剧情", "year": 1994, "rating": 9.7, "description": "希望让人自由"},
            {"id": 2, "title": "霸王别姬", "director": "陈凯歌", "actors": ["张国荣", "张丰毅", "巩俐"], "genre": "剧情", "year": 1993, "rating": 9.6, "description": "风华绝代"},
            {"id": 3, "title": "阿甘正传", "director": "罗伯特·泽米吉斯", "actors": ["汤姆·汉克斯"], "genre": "剧情", "year": 1994, "rating": 9.5, "description": "人生就像巧克力"},
            {"id": 4, "title": "盗梦空间", "director": "克里斯托弗·诺兰", "actors": ["莱昂纳多"], "genre": "科幻/悬疑", "year": 2010, "rating": 9.4, "description": "梦境与现实"},
            {"id": 5, "title": "星际穿越", "director": "克里斯托弗·诺兰", "actors": ["马修·麦康纳"], "genre": "科幻", "year": 2014, "rating": 9.4, "description": "穿越时空的爱"},
            {"id": 6, "title": "千与千寻", "director": "宫崎骏", "actors": ["柊瑠美"], "genre": "动画", "year": 2001, "rating": 9.4, "description": "成长的旅程"},
            {"id": 7, "title": "让子弹飞", "director": "姜文", "actors": ["姜文", "葛优", "周润发"], "genre": "剧情/喜剧", "year": 2010, "rating": 9.0, "description": "站着把钱挣了"},
            {"id": 8, "title": "我不是药神", "director": "文牧野", "actors": ["徐峥", "王传君", "周一围"], "genre": "剧情", "year": 2018, "rating": 9.0, "description": "现实题材的震撼之作"},
        ]

    def _generate_reviews(self):
        """生成系统内置评论（带情感标签）"""
        reviews = []
        for movie in self.movies:
            for _ in range(random.randint(10, 15)):
                sentiment = random.choice(["positive", "neutral", "negative"])
                text = random.choice(DOUBAN_REAL_REVIEWS[sentiment])
                rating = 5 if sentiment == "positive" else (3 if sentiment == "neutral" else 2)
                reviews.append({
                    "movie_id": movie["id"],
                    "movie_name": movie["title"],
                    "user": random.choice(USER_NAMES),
                    "content": text,
                    "rating": rating,
                    "sentiment": sentiment,
                    "time": datetime.now().strftime("%Y-%m-%d")
                })
        return reviews

    def _generate_users(self):
        users = {}
        for i in range(1, 21):
            user_id = f"user_{i:03d}"
            watched = random.sample(self.movies, random.randint(3, 5))
            users[user_id] = {"watched": [m["id"] for m in watched], "ratings": {m["id"]: random.randint(6, 10) for m in watched}, "favorites": [], "watchlist": [], "user_info": {}}
        return users

    def get_system_reviews(self, movie_id):
        return [r for r in self.reviews if r["movie_id"] == movie_id]

    def get_system_reviews_by_title(self, movie_title):
        """根据电影标题获取系统评论"""
        movie = next((m for m in self.movies if m["title"] == movie_title), None)
        if movie:
            return self.get_system_reviews(movie["id"])
        return []

    def get_crawled_reviews(self, movie_name):
        return self.crawled_reviews.get(movie_name, [])

    def get_uploaded_reviews(self, movie_name):
        return self.uploaded_reviews.get(movie_name, [])

    def search_movie_info(self, movie_name):
        """搜索电影信息 - 增强版：支持豆瓣爬虫获取详情"""
        result = {"success": False, "data": None, "message": "", "suggestions": []}
        detail = None

        print(f"[DEBUG] 开始搜索电影: {movie_name}")
        print(f"[DEBUG] Spider是否可用: {self.spider is not None}")

        if self.spider:
            try:
                print(f"[DEBUG] 正在通过豆瓣搜索电影ID...")
                movie_id, title = self.spider.search_movie_id(movie_name)
                print(f"[DEBUG] 搜索结果 - movie_id: {movie_id}, title: {title}")

                if movie_id and title:
                    print(f"[DEBUG] 正在获取电影详情...")
                    detail = self.spider.get_movie_detail(movie_id)
                    print(f"[DEBUG] 详情获取成功: {detail is not None}")

                    if detail:
                        result["success"] = True
                        result["data"] = {
                            "title": title,
                            "year": detail.get('year', '未知'),
                            "rating": detail.get('rating', '暂无'),
                            "director": detail.get('director', '未知'),
                            "actors": detail.get('actors', ['未知']),
                            "genre": ' / '.join(detail.get('genres', ['未知'])),
                            "genres": detail.get('genres', ['未知']),
                            "country": detail.get('country', '未知'),
                            "language": detail.get('language', '未知'),
                            "duration": detail.get('duration', '未知'),
                            "description": detail.get('summary', '暂无简介'),
                            "poster": detail.get('poster', ''),
                            "local_poster": detail.get('local_poster', ''),
                            "release_date": detail.get('release_date', '未知'),
                            "tags": detail.get('tags', []),
                            "wish_count": detail.get('wish_count', '0'),
                            "collect_count": detail.get('collect_count', '0'),
                            "votes": detail.get('votes', '0'),
                            "source": "豆瓣爬虫"
                        }
                        print(f"[DEBUG] 返回豆瓣爬虫数据，标题: {title}")
                        return result
                else:
                    print(f"[DEBUG] 豆瓣搜索未找到结果，尝试本地缓存")
            except Exception as e:
                print(f"[DEBUG] 豆瓣爬虫搜索失败，尝试本地缓存: {e}")

        # 本地缓存匹配
        if movie_name in LOCAL_MOVIE_CACHE:
            cached = LOCAL_MOVIE_CACHE[movie_name]
            result["success"] = True
            result["data"] = {
                "title": cached["title"],
                "year": cached["year"],
                "rating": cached["rating"],
                "director": cached["director"],
                "actors": cached["actors"],
                "genre": cached["genre"],
                "genres": cached.get("genres", [cached["genre"]]),
                "country": cached.get("country", "未知"),
                "language": cached.get("language", "未知"),
                "duration": cached.get("duration", "未知"),
                "description": cached["description"],
                "poster": cached.get("poster", ""),
                "local_poster": "",
                "release_date": cached.get("release_date", "未知"),
                "tags": cached.get("tags", []),
                "wish_count": cached.get("wish_count", "0"),
                "collect_count": cached.get("collect_count", "0"),
                "votes": cached.get("votes", "0"),
                "source": "本地缓存"
            }
            return result

        best_match, score = fuzzy_match_movie(movie_name)
        if best_match and score > 0.6:
            cached = LOCAL_MOVIE_CACHE[best_match]
            result["success"] = True
            result["data"] = {
                "title": cached["title"],
                "year": cached["year"],
                "rating": cached["rating"],
                "director": cached["director"],
                "actors": cached["actors"],
                "genre": cached["genre"],
                "genres": cached.get("genres", [cached["genre"]]),
                "country": cached.get("country", "未知"),
                "language": cached.get("language", "未知"),
                "duration": cached.get("duration", "未知"),
                "description": cached["description"],
                "poster": cached.get("poster", ""),
                "local_poster": "",
                "release_date": cached.get("release_date", "未知"),
                "tags": cached.get("tags", []),
                "wish_count": cached.get("wish_count", "0"),
                "collect_count": cached.get("collect_count", "0"),
                "votes": cached.get("votes", "0"),
                "source": f"模糊匹配({score:.0%})"
            }
            return result

        suggestions = [name for name in LOCAL_MOVIE_CACHE.keys() if movie_name.lower() in name.lower() or name.lower() in movie_name.lower()]
        if suggestions:
            result["suggestions"] = suggestions[:5]
        result["message"] = f"未找到《{movie_name}》"
        return result

    def get_all_data_sources(self):
        return {"system": [{"name": m["title"], "count": len(self.get_system_reviews(m["id"]))} for m in self.movies],
                "crawled": [{"name": name, "count": len(reviews)} for name, reviews in self.crawled_reviews.items()],
                "uploaded": [{"name": name, "count": len(reviews)} for name, reviews in self.uploaded_reviews.items()]}

    def upload_user_data(self, file):
        """上传并解析用户数据文件 - 增强版支持更多列"""
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        try:
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext == '.csv':
                encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'latin-1']
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except:
                        continue
                if df is None:
                    raise Exception("无法解析CSV文件")
            elif file_ext in ['.xlsx', '.xls']:
                try:
                    if file_ext == '.xlsx':
                        df = pd.read_excel(file_path, engine='openpyxl')
                    else:
                        try:
                            df = pd.read_excel(file_path, engine='xlrd')
                        except:
                            df = pd.read_excel(file_path)
                except Exception as e:
                    raise Exception(f"Excel文件解析失败: {str(e)}")
            else:
                raise Exception(f"不支持的文件格式: {file_ext}")

            return self._auto_detect_and_parse(df, filename, file_ext)
        except Exception as e:
            return {'success': False, 'error': str(e)}
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    def _auto_detect_and_parse(self, df, filename, file_ext='.xlsx'):
        columns = df.columns.tolist()
        columns_lower = [c.lower() for c in columns]

        if '电影名称' in columns or 'movie_name' in columns_lower:
            return self._parse_long_format(df, filename, file_ext)
        if any('电影' in c and ('名称' in c or '名' in c) for c in columns):
            return self._parse_long_format(df, filename, file_ext)
        if '观影内容' in columns or 'watch_content' in columns_lower:
            return self._parse_wide_format(df, filename, file_ext)
        if '用户昵称' in columns and ('电影' in columns or any('电影' in c for c in columns)):
            return self._parse_simple_format(df, filename, file_ext)
        return self._parse_long_format(df, filename, file_ext)

    def _parse_long_format(self, df, filename, file_ext='.xlsx'):
        col_map = {
            'nickname': None,
            'gender': None,
            'age': None,
            'movie_name': None,
            'watch_date': None,
            'movie_type': None,
            'movie_rating': None,
            'director': None,
            'actors': None,
            'watch_channel': None,
            'remark': None
        }

        col_mappings = {
            'nickname': ['用户昵称', '昵称', 'nickname', 'user_name', '用户名', '姓名'],
            'gender': ['性别', 'gender', 'sex', '性别男/女'],
            'age': ['年龄', 'age', 'years', '岁数'],
            'movie_name': ['电影名称', '电影名', 'movie_name', 'movie_title', 'title', '片名', '影片名称'],
            'watch_date': ['观影日期', '日期', 'watch_date', 'date', '观影时间', '时间', '上映日期'],
            'movie_type': ['电影类型', '类型', 'movie_type', 'genre', 'type', '影片类型', '类别'],
            'movie_rating': ['电影评分', '评分', 'movie_rating', 'rating', 'score', '分数', '豆瓣评分'],
            'director': ['导演', 'director', '导演讲', '导演名'],
            'actors': ['主演', 'actors', '演员', '主演阵容', '演员表'],
            'watch_channel': ['观影渠道', '渠道', 'watch_channel', 'channel', '观看方式', '来源', '观看平台'],
            'remark': ['备注', 'remark', 'note', '评论', '观影感受', '感想']
        }

        for col in df.columns:
            col_lower = col.lower()
            for key, patterns in col_mappings.items():
                if col in patterns or col_lower in [p.lower() for p in patterns]:
                    col_map[key] = col
                    break

        if not col_map['nickname']:
            raise Exception("缺少用户昵称列")
        if not col_map['movie_name']:
            raise Exception("缺少电影名称列")

        movie_records = []
        user_info_cache = {}

        for idx, row in df.iterrows():
            nickname = str(row.get(col_map['nickname'], '')).strip()
            if not nickname or nickname == 'nan' or nickname == 'None':
                continue
            movie_name = str(row.get(col_map['movie_name'], '')).strip()
            if not movie_name or movie_name == 'nan' or movie_name == 'None':
                continue

            gender = '未知'
            if col_map['gender'] and pd.notna(row.get(col_map['gender'])):
                gender_val = str(row.get(col_map['gender'])).strip()
                if gender_val and gender_val != 'nan':
                    gender = gender_val

            age = 0
            if col_map['age'] and pd.notna(row.get(col_map['age'])):
                try:
                    age_val = row.get(col_map['age'])
                    age = int(float(age_val)) if age_val else 0
                except:
                    age = 0

            watch_date = ''
            if col_map['watch_date'] and pd.notna(row.get(col_map['watch_date'])):
                watch_date_val = row.get(col_map['watch_date'])
                if hasattr(watch_date_val, 'strftime'):
                    watch_date = watch_date_val.strftime('%Y-%m-%d')
                else:
                    watch_date = str(watch_date_val).strip()
                    if watch_date == 'nan':
                        watch_date = ''
                    elif ' ' in watch_date:
                        watch_date = watch_date.split(' ')[0]
                    elif 'T' in watch_date:
                        watch_date = watch_date.split('T')[0]

            movie_type = '未知'
            if col_map['movie_type'] and pd.notna(row.get(col_map['movie_type'])):
                movie_type_val = str(row.get(col_map['movie_type'])).strip()
                if movie_type_val and movie_type_val != 'nan':
                    movie_type = movie_type_val

            movie_rating = None
            if col_map['movie_rating'] and pd.notna(row.get(col_map['movie_rating'])):
                try:
                    rating_val = row.get(col_map['movie_rating'])
                    movie_rating = float(rating_val) if rating_val else None
                    if movie_rating:
                        movie_rating = round(movie_rating, 1)
                except:
                    movie_rating = None

            director = '未知'
            if col_map['director'] and pd.notna(row.get(col_map['director'])):
                director_val = str(row.get(col_map['director'])).strip()
                if director_val and director_val != 'nan':
                    director = director_val

            actors = '未知'
            if col_map['actors'] and pd.notna(row.get(col_map['actors'])):
                actors_val = str(row.get(col_map['actors'])).strip()
                if actors_val and actors_val != 'nan':
                    actors = actors_val

            watch_channel = '未知'
            if col_map['watch_channel'] and pd.notna(row.get(col_map['watch_channel'])):
                channel_val = str(row.get(col_map['watch_channel'])).strip()
                if channel_val and channel_val != 'nan':
                    watch_channel = channel_val

            remark = ''
            if col_map['remark'] and pd.notna(row.get(col_map['remark'])):
                remark_val = str(row.get(col_map['remark'])).strip()
                if remark_val and remark_val != 'nan':
                    remark = remark_val

            movie_record = {
                'nickname': nickname,
                'gender': gender,
                'age': age,
                'movie_name': movie_name,
                'watch_date': watch_date,
                'movie_type': movie_type,
                'movie_rating': movie_rating,
                'director': director,
                'actors': actors,
                'watch_channel': watch_channel,
                'remark': remark
            }
            movie_records.append(movie_record)

            if nickname not in user_info_cache:
                user_info_cache[nickname] = {
                    'nickname': nickname,
                    'gender': gender,
                    'age': age,
                    'movies': [],
                    'genres': [],
                    'ratings': []
                }
            user_info_cache[nickname]['movies'].append(movie_name)
            if movie_type and movie_type != '未知':
                user_info_cache[nickname]['genres'].append(movie_type)
            if movie_rating:
                user_info_cache[nickname]['ratings'].append(movie_rating)

        if not movie_records:
            raise Exception("没有找到有效的用户数据")

        analysis_records = []
        for nickname, summary in user_info_cache.items():
            genre_counter = Counter(summary['genres'])
            top_genres = dict(genre_counter.most_common(3))

            analysis_records.append({
                'nickname': nickname,
                'gender': summary['gender'],
                'age': summary['age'],
                'watch_count': len(summary['movies']),
                'watch_content': ','.join(summary['movies']),
                'top_genres': top_genres,
                'avg_rating': round(sum(summary['ratings']) / len(summary['ratings']), 1) if summary['ratings'] else 0
            })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_id = f"user_data_{timestamp}"
        self.uploaded_user_data[data_id] = {
            'filename': filename,
            'file_type': file_ext,
            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'records': analysis_records,
            'raw_records': movie_records,
            'total_users': len(analysis_records),
            'total_movies': len(movie_records),
            'format': 'long'
        }
        return {
            'success': True,
            'data_id': data_id,
            'total_users': len(analysis_records),
            'total_movies': len(movie_records),
            'preview': movie_records[:10]
        }

    def _parse_wide_format(self, df, filename, file_ext='.xlsx'):
        col_map = {'nickname': None, 'watch_count': None, 'watch_content': None}
        for col in df.columns:
            col_lower = col.lower()
            if '用户昵称' in col or '昵称' in col or col_lower == 'nickname':
                col_map['nickname'] = col
            elif '观影数量' in col or '数量' in col or col_lower == 'watch_count':
                col_map['watch_count'] = col
            elif '观影内容' in col or '内容' in col or col_lower == 'watch_content':
                col_map['watch_content'] = col

        if not col_map['nickname']:
            raise Exception("缺少用户昵称列")

        user_records = []
        for idx, row in df.iterrows():
            nickname = str(row.get(col_map['nickname'], '')).strip()
            if not nickname or nickname == 'nan' or nickname == 'None':
                continue
            watch_count = 0
            if col_map['watch_count'] and pd.notna(row.get(col_map['watch_count'])):
                try:
                    watch_count = int(float(row.get(col_map['watch_count'])))
                except:
                    watch_count = 0
            record = {'nickname': nickname, 'gender': '未知', 'age': 0, 'watch_count': watch_count,
                      'watch_content': str(row.get(col_map['watch_content'], '')) if col_map['watch_content'] and pd.notna(row.get(col_map['watch_content'])) else ''}
            user_records.append(record)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_id = f"user_data_{timestamp}"
        self.uploaded_user_data[data_id] = {
            'filename': filename, 'file_type': file_ext, 'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'records': user_records, 'total_users': len(user_records), 'format': 'wide'
        }
        return {'success': True, 'data_id': data_id, 'total_users': len(user_records), 'preview': user_records[:10]}

    def _parse_simple_format(self, df, filename, file_ext='.xlsx'):
        movie_cols = [c for c in df.columns if '电影' in c or 'movie' in c.lower()]
        col_nickname = None
        for col in df.columns:
            if '用户昵称' in col or '昵称' in col or col.lower() == 'nickname':
                col_nickname = col
                break
        if not col_nickname:
            raise Exception("缺少用户昵称列")

        user_records = []
        for idx, row in df.iterrows():
            nickname = str(row.get(col_nickname, '')).strip()
            if not nickname or nickname == 'nan' or nickname == 'None':
                continue
            movies = []
            for col in movie_cols:
                val = row.get(col, '')
                if pd.notna(val) and str(val).strip() and str(val).strip() != 'nan':
                    movies.append(str(val).strip())
            if not movies:
                continue
            record = {'nickname': nickname, 'gender': '未知', 'age': 0, 'watch_count': len(movies), 'watch_content': ','.join(movies)}
            user_records.append(record)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_id = f"user_data_{timestamp}"
        self.uploaded_user_data[data_id] = {
            'filename': filename, 'file_type': file_ext, 'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'records': user_records, 'total_users': len(user_records), 'format': 'simple'
        }
        return {'success': True, 'data_id': data_id, 'total_users': len(user_records), 'preview': user_records[:10]}

    def analyze_user_data(self, data_id):
        """分析上传的用户数据 - 从 raw_records 直接读取类型"""
        if data_id not in self.uploaded_user_data:
            return None
        data = self.uploaded_user_data[data_id]
        records = data['records']
        raw_records = data.get('raw_records', [])
        total_users = len(records)

        gender_count = Counter([r.get('gender', '未知') for r in records])

        ages = [r.get('age', 0) for r in records if r.get('age', 0) > 0]
        avg_age = sum(ages) / len(ages) if ages else 0

        watch_counts = [r.get('watch_count', 0) for r in records]
        avg_watch = sum(watch_counts) / len(watch_counts) if watch_counts else 0
        max_watch = max(watch_counts) if watch_counts else 0

        age_groups = {'18岁以下': 0, '18-25岁': 0, '26-35岁': 0, '36-50岁': 0, '50岁以上': 0}
        for r in records:
            age = r.get('age', 0)
            if age <= 0: continue
            elif age < 18: age_groups['18岁以下'] += 1
            elif age <= 25: age_groups['18-25岁'] += 1
            elif age <= 35: age_groups['26-35岁'] += 1
            elif age <= 50: age_groups['36-50岁'] += 1
            else: age_groups['50岁以上'] += 1

        # 统计电影和类型 - 从 raw_records 直接读取
        movie_titles = []
        genre_counter = Counter()

        for r in raw_records:
            movie_name = r.get('movie_name', '')
            if movie_name and movie_name != 'nan':
                movie_titles.append(movie_name)

            movie_type = r.get('movie_type', '')
            if movie_type and movie_type != '未知':
                for gt in re.split('[/、,，]', movie_type):
                    gt = gt.strip()
                    if gt:
                        genre_counter[gt] += 1

        # 如果没有 raw_records 或没有类型数据，从 records 中统计
        if len(genre_counter) == 0:
            for r in records:
                content = r.get('watch_content', '')
                if content:
                    movies = re.split('[,，、/;；]', str(content))
                    for movie in movies:
                        movie = movie.strip()
                        if movie:
                            movie_titles.append(movie)

                top_genres = r.get('top_genres', {})
                if isinstance(top_genres, dict):
                    for genre, weight in top_genres.items():
                        genre_counter[genre] += weight

        popular_movies = Counter(movie_titles).most_common(10)

        # 生成年龄分布图
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.bar(list(age_groups.keys()), list(age_groups.values()),
                color=['#667eea', '#48bb78', '#4299e1', '#ecc94b', '#f56565'])
        ax1.set_title('用户年龄分布', fontsize=14, fontweight='bold')
        buffer1 = BytesIO()
        fig1.savefig(buffer1, format='png', dpi=100, bbox_inches='tight')
        buffer1.seek(0)
        age_chart = base64.b64encode(buffer1.getvalue()).decode('utf-8')
        plt.close(fig1)

        # 生成性别分布图
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        colors = ['#48bb78', '#4299e1', '#ecc94b', '#f56565']
        ax2.pie(list(gender_count.values()), labels=list(gender_count.keys()),
                autopct='%1.1f%%', colors=colors[:len(gender_count)])
        ax2.set_title('用户性别分布', fontsize=14, fontweight='bold')
        buffer2 = BytesIO()
        fig2.savefig(buffer2, format='png', dpi=100, bbox_inches='tight')
        buffer2.seek(0)
        gender_chart = base64.b64encode(buffer2.getvalue()).decode('utf-8')
        plt.close(fig2)

        # 生成类型偏好图
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        if genre_counter:
            top_genres_list = genre_counter.most_common(8)
            genres_labels = [g[0] for g in top_genres_list]
            genres_values = [g[1] for g in top_genres_list]
            bars = ax3.barh(genres_labels, genres_values, color='#667eea', edgecolor='white')
            for bar, val in zip(bars, genres_values):
                ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                         str(val), ha='left', va='center', fontsize=11)
            ax3.set_xlabel('观影次数', fontsize=12, fontweight='bold')
            ax3.set_title('用户类型偏好', fontsize=14, fontweight='bold')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
        else:
            ax3.text(0.5, 0.5, '暂无类型数据\n请确保上传的文件包含"电影类型"列',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('用户类型偏好', fontsize=14, fontweight='bold')
        plt.tight_layout()
        buffer3 = BytesIO()
        fig3.savefig(buffer3, format='png', dpi=100, bbox_inches='tight')
        buffer3.seek(0)
        genre_chart = base64.b64encode(buffer3.getvalue()).decode('utf-8')
        plt.close(fig3)

        # 生成观影数量分布图
        fig4, ax4 = plt.subplots(figsize=(12, 7))
        if watch_counts:
            count_dist = Counter(watch_counts)
            sorted_counts = sorted(count_dist.items())
            x_labels = [str(c[0]) for c in sorted_counts]
            y_values = [c[1] for c in sorted_counts]
            bars = ax4.bar(x_labels, y_values, color='#48bb78', edgecolor='white', linewidth=2, alpha=0.8)
            for bar, val in zip(bars, y_values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                         str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax4.set_xlabel('观影数量 (部)', fontsize=14, fontweight='bold')
            ax4.set_ylabel('用户人数 (人)', fontsize=14, fontweight='bold')
            ax4.set_title(f'用户观影数量分布 (共{len(watch_counts)}人，人均{avg_watch:.1f}部)',
                          fontsize=16, fontweight='bold', pad=20)
            ax4.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax4.grid(axis='y', alpha=0.3, linestyle='--')
            ax4.set_facecolor('#f8f9fa')
        else:
            ax4.text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('观影数量分布', fontsize=14, fontweight='bold')

        plt.tight_layout()
        buffer4 = BytesIO()
        fig4.savefig(buffer4, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        buffer4.seek(0)
        watch_chart = base64.b64encode(buffer4.getvalue()).decode('utf-8')
        plt.close(fig4)

        return {
            'success': True,
            'data_id': data_id,
            'filename': data['filename'],
            'file_type': data.get('file_type', '.xlsx'),
            'upload_time': data['upload_time'],
            'total_users': total_users,
            'total_movies': data.get('total_movies', sum(watch_counts)),
            'gender_dist': dict(gender_count),
            'avg_age': round(avg_age, 1),
            'avg_watch': round(avg_watch, 1),
            'max_watch': max_watch,
            'popular_movies': popular_movies,
            'genre_prefs': dict(genre_counter.most_common(10)),
            'age_chart': age_chart,
            'gender_chart': gender_chart,
            'genre_chart': genre_chart,
            'watch_chart': watch_chart,
            'sample_records': records[:5],
            'raw_records': raw_records,
            'format': data.get('format', 'wide')
        }

    def get_user_genre_preferences(self, data_id):
        """获取用户类型偏好（用于盲盒功能）- 从 raw_records 直接读取类型"""
        if data_id not in self.uploaded_user_data:
            return None

        data = self.uploaded_user_data[data_id]
        raw_records = data.get('raw_records', [])

        # 如果没有 raw_records，尝试从 records 中获取
        if not raw_records:
            records = data.get('records', [])
            for record in records:
                watch_content = record.get('watch_content', '')
                if watch_content:
                    movies = re.split('[,，、/;；]', watch_content)
                    for movie in movies:
                        movie = movie.strip()
                        if movie:
                            if movie in LOCAL_MOVIE_CACHE:
                                movie_type = LOCAL_MOVIE_CACHE[movie].get('genre', '未知')
                                raw_records.append({'movie_name': movie, 'movie_type': movie_type})
                            else:
                                raw_records.append({'movie_name': movie, 'movie_type': '未知'})

        # 统计类型偏好 - 从 raw_records 中直接读取 movie_type
        genre_counter = Counter()

        for record in raw_records:
            movie_type = record.get('movie_type', '')
            if movie_type and movie_type != '未知':
                for gt in re.split('[/、,，]', movie_type):
                    gt = gt.strip()
                    if gt:
                        genre_counter[gt] += 1

        # 获取用户偏好
        top_genres = genre_counter.most_common(5)

        # 生成类型偏好图表
        fig, ax = plt.subplots(figsize=(10, 6))
        if genre_counter:
            top_genres_list = genre_counter.most_common(8)
            genres_labels = [g[0] for g in top_genres_list]
            genres_values = [g[1] for g in top_genres_list]

            bars = ax.barh(genres_labels, genres_values, color='#667eea', edgecolor='white')

            for bar, value in zip(bars, genres_values):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        str(value), ha='left', va='center', fontsize=11)

            ax.set_xlabel('观影次数', fontsize=12, fontweight='bold')
            ax.set_title('用户类型偏好分析', fontsize=14, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            ax.text(0.5, 0.5, '暂无类型数据\n请确保上传的文件包含"电影类型"列',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('用户类型偏好', fontsize=14, fontweight='bold')

        plt.tight_layout()
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        genre_chart = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)

        # 构建类型对应的电影列表
        genre_movies = {}
        for record in raw_records:
            movie_name = record.get('movie_name', '')
            movie_type = record.get('movie_type', '')
            if movie_type and movie_type != '未知' and movie_name:
                for gt in re.split('[/、,，]', movie_type):
                    gt = gt.strip()
                    if gt:
                        if gt not in genre_movies:
                            genre_movies[gt] = []
                        if movie_name not in genre_movies[gt]:
                            genre_movies[gt].append(movie_name)

        return {
            'success': True,
            'genre_preferences': dict(genre_counter),
            'top_genres': top_genres,
            'total_users': len(data.get('records', [])),
            'genre_movies': genre_movies if genre_movies else {
                "剧情": ["肖申克的救赎", "霸王别姬", "阿甘正传", "我不是药神", "绿皮书"],
                "科幻": ["星际穿越", "盗梦空间", "流浪地球", "黑客帝国", "阿凡达"],
                "动作": ["让子弹飞", "速度与激情", "碟中谍", "叶问", "战狼2"],
                "爱情": ["泰坦尼克号", "怦然心动", "你的名字", "情书", "爱乐之城"],
                "喜剧": ["三傻大闹宝莱坞", "夏洛特烦恼", "唐人街探案", "疯狂的石头", "西虹市首富"],
                "动画": ["千与千寻", "龙猫", "疯狂动物城", "寻梦环游记", "哪吒之魔童降世"],
                "悬疑": ["盗梦空间", "看不见的客人", "消失的爱人", "调音师", "心迷宫"]
            },
            'has_data': len(genre_counter) > 0,
            'genre_chart': genre_chart
        }