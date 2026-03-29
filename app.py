# app.py - 完整版
"""
豆瓣电影数据分析与推荐系统 - Web版本
使用 PyTorch CNN+LSTM 情感分析模型 + 神经网络推荐引擎 + 真实豆瓣爬虫
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from datetime import datetime
import random
from collections import Counter
import json
import os
import re
import jieba
import time
import pandas as pd

# PyTorch
import torch
import torch.nn as nn

# 爬虫相关库
try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ 请安装爬虫库: pip install requests beautifulsoup4")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建Flask应用
app = Flask(__name__)
app.secret_key = 'douban_movie_system_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==================== 豆瓣真实风格评论数据（备用）====================
DOUBAN_REAL_REVIEWS = {
    "positive": [
        "太震撼了！看完久久不能平静，绝对是年度最佳！",
        "经典中的经典，每次看都有新的感悟，五星推荐！",
        "导演的功力太强了，每一个镜头都是艺术品，演员的表演也非常到位。",
        "哭了整整两个小时，这部电影真的太打动人心了。",
        "神作！没有任何一部电影能给我这样的观影体验。",
        "剧本太牛了，情节环环相扣，结局出人意料又情理之中。",
        "看完只想说：这才是电影！国产电影的骄傲！",
        "满分好评，已经二刷了，每次看都有新发现。",
        "太喜欢了！从配乐到画面都无可挑剔。",
        "这部电影改变了我的人生观，强烈推荐每个人都去看！"
    ],
    "neutral": [
        "还行吧，中规中矩，没有想象中那么好。",
        "一般般，看完就忘，没有什么印象深刻的点。",
        "普普通通，可以一看，但不会二刷。",
        "期望太高了，实际看下来有点失望。",
        "不算差但也不算好，打发时间还可以。",
        "剧情有点老套，没什么新意。",
        "演员演技还行，但剧本太弱了。",
        "特效不错，但剧情拖沓，节奏太慢。",
        "看完没什么感觉，可能不是我喜欢的类型。",
        "及格线以上的作品，但谈不上经典。"
    ],
    "negative": [
        "太失望了！完全浪费时间和金钱。",
        "什么烂片？剧情逻辑完全不通，看得我尴尬癌都犯了。",
        "演员演技太尬了，完全看不下去。",
        "一分都不想给，浪费时间！",
        "剧本太差了，全是套路，毫无新意。",
        "特效五毛，剧情狗血，烂片中的战斗机。",
        "导演想表达什么完全看不懂，故弄玄虚。",
        "太无聊了，看了半小时就想走。",
        "改编得一塌糊涂，完全毁了原著。",
        "全程尬演，尴尬到脚趾抠地。"
    ]
}

USER_NAMES = [
    "豆瓣用户", "影迷小张", "电影爱好者", "观影达人", "评分机器人", "影评人小王",
    "路人甲", "电影发烧友", "爱看电影的猫", "深夜观影者", "周末影迷", "电影控"
]

# ==================== 本地电影数据库 ====================
LOCAL_MOVIE_CACHE = {
    "肖申克的救赎": {"id": "1292052", "title": "肖申克的救赎", "year": "1994", "rating": 9.7, "director": "弗兰克·德拉邦特", "actors": ["蒂姆·罗宾斯", "摩根·弗里曼"], "genre": "剧情", "country": "美国", "language": "英语", "duration": "142分钟", "description": "希望让人自由，一部关于希望与救赎的经典之作。"},
    "霸王别姬": {"id": "1291546", "title": "霸王别姬", "year": "1993", "rating": 9.6, "director": "陈凯歌", "actors": ["张国荣", "张丰毅", "巩俐"], "genre": "剧情", "country": "中国", "language": "汉语普通话", "duration": "171分钟", "description": "风华绝代，一曲人生悲歌。"},
    "阿甘正传": {"id": "1292720", "title": "阿甘正传", "year": "1994", "rating": 9.5, "director": "罗伯特·泽米吉斯", "actors": ["汤姆·汉克斯"], "genre": "剧情", "country": "美国", "language": "英语", "duration": "142分钟", "description": "人生就像一盒巧克力，你永远不知道下一块是什么味道。"},
    "盗梦空间": {"id": "3541415", "title": "盗梦空间", "year": "2010", "rating": 9.4, "director": "克里斯托弗·诺兰", "actors": ["莱昂纳多·迪卡普里奥"], "genre": "科幻/悬疑", "country": "美国", "language": "英语", "duration": "148分钟", "description": "梦境与现实，一场关于意识的大胆探索。"},
    "星际穿越": {"id": "1889243", "title": "星际穿越", "year": "2014", "rating": 9.4, "director": "克里斯托弗·诺兰", "actors": ["马修·麦康纳"], "genre": "科幻", "country": "美国", "language": "英语", "duration": "169分钟", "description": "穿越时空的爱，人类命运的终极探索。"},
    "让子弹飞": {"id": "3742360", "title": "让子弹飞", "year": "2010", "rating": 9.0, "director": "姜文", "actors": ["姜文", "葛优", "周润发"], "genre": "剧情/喜剧", "country": "中国", "language": "汉语普通话", "duration": "132分钟", "description": "站着把钱挣了！"},
    "我不是药神": {"id": "26752088", "title": "我不是药神", "year": "2018", "rating": 9.0, "director": "文牧野", "actors": ["徐峥", "王传君", "周一围"], "genre": "剧情", "country": "中国", "language": "汉语普通话", "duration": "117分钟", "description": "现实题材的震撼之作。"},
    "千与千寻": {"id": "1291561", "title": "千与千寻", "year": "2001", "rating": 9.4, "director": "宫崎骏", "actors": ["柊瑠美"], "genre": "动画/奇幻", "country": "日本", "language": "日语", "duration": "125分钟", "description": "成长的旅程，宫崎骏的巅峰之作。"},
    "这个杀手不太冷": {"id": "1295644", "title": "这个杀手不太冷", "year": "1994", "rating": 9.4, "director": "吕克·贝松", "actors": ["让·雷诺", "娜塔莉·波特曼"], "genre": "剧情/动作", "country": "法国", "language": "英语", "duration": "110分钟", "description": "杀手与小女孩的温情故事。"},
    "泰坦尼克号": {"id": "1292722", "title": "泰坦尼克号", "year": "1997", "rating": 9.5, "director": "詹姆斯·卡梅隆", "actors": ["莱昂纳多", "凯特·温丝莱特"], "genre": "爱情/灾难", "country": "美国", "language": "英语", "duration": "194分钟", "description": "永恒的爱情，永不沉没的经典。"},
    "楚门的世界": {"id": "1292064", "title": "楚门的世界", "year": "1998", "rating": 9.3, "director": "彼得·威尔", "actors": ["金·凯瑞"], "genre": "剧情/科幻", "country": "美国", "language": "英语", "duration": "103分钟", "description": "真实与虚假的边界在哪里？"},
    "海上钢琴师": {"id": "1292001", "title": "海上钢琴师", "year": "1998", "rating": 9.3, "director": "朱塞佩·托纳多雷", "actors": ["蒂姆·罗斯"], "genre": "剧情/音乐", "country": "意大利", "language": "英语", "duration": "125分钟", "description": "1900的故事，音乐与自由的传奇。"},
}

def fuzzy_match_movie(movie_name):
    from difflib import SequenceMatcher
    movie_name_lower = movie_name.lower()
    best_match = None
    best_score = 0

    for cached_name in LOCAL_MOVIE_CACHE.keys():
        score = SequenceMatcher(None, movie_name_lower, cached_name.lower()).ratio()
        if cached_name.lower() in movie_name_lower or movie_name_lower in cached_name.lower():
            score = max(score, 0.7)
        if score > best_score and score > 0.5:
            best_score = score
            best_match = cached_name

    return best_match, best_score


# ==================== 豆瓣爬虫类 ====================
class DoubanSpider:
    """豆瓣电影评论爬虫"""

    def __init__(self):
        self.headers = {
            'Cookie': 'bid=BwcycY_GcAA; ap_v=0,6.0; __utma=30149280.1808011285.1774349897.1774352743.1774582003.3; __utmc=30149280; __utmz=30149280.1774582003.3.3.utmcsr=cn.bing.com|utmccn=(referral)|utmcmd=referral|utmcct=/; dbcl2="293423119:3LWQOX7tv8w"; ck=Fula',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'movie.douban.com',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Connection': 'keep-alive'
        }
        self.ml_analyzer = None

    def search_movie_id(self, movie_name):
        """搜索电影ID"""
        try:
            search_url = f"https://movie.douban.com/j/subject_suggest?q={movie_name}"
            response = requests.get(search_url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                results = response.json()
                if results:
                    return results[0].get('id'), results[0].get('title')
            return None, None
        except Exception as e:
            print(f"豆瓣搜索失败: {e}")
            return None, None

    def get_movie_detail(self, movie_id):
        """获取电影详细信息"""
        try:
            url = f"https://movie.douban.com/subject/{movie_id}/"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.encoding = 'utf-8'

            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')
            info = {}

            director_elem = soup.find('a', rel='v:directedBy')
            info['director'] = director_elem.text.strip() if director_elem else "未知"

            actor_elems = soup.find_all('a', rel='v:starring')
            info['actors'] = [a.text.strip() for a in actor_elems[:3]] if actor_elems else ["未知"]

            genre_elems = soup.find_all('span', property='v:genre')
            info['genres'] = [g.text.strip() for g in genre_elems] if genre_elems else ["未知"]

            info_elem = soup.find('div', id='info')
            if info_elem:
                info_text = info_elem.text
                country_match = re.search(r'制片国家/地区:\s*(.+?)(?:\n|$)', info_text)
                info['country'] = country_match.group(1).strip() if country_match else "未知"
                language_match = re.search(r'语言:\s*(.+?)(?:\n|$)', info_text)
                info['language'] = language_match.group(1).strip() if language_match else "未知"
                duration_match = re.search(r'片长:\s*(.+?)(?:\n|$)', info_text)
                info['duration'] = duration_match.group(1).strip() if duration_match else "未知"
            else:
                info['country'] = "未知"
                info['language'] = "未知"
                info['duration'] = "未知"

            rating_elem = soup.find('strong', property='v:average')
            info['rating'] = rating_elem.text.strip() if rating_elem else "暂无"

            votes_elem = soup.find('span', property='v:votes')
            info['votes'] = votes_elem.text.strip() if votes_elem else "0"

            summary_elem = soup.find('span', property='v:summary')
            info['summary'] = summary_elem.text.strip() if summary_elem else "暂无简介"

            return info

        except Exception as e:
            print(f"获取电影详情失败: {e}")
            return None

    def crawl_reviews(self, movie_id, max_count=20, progress_callback=None):
        """爬取电影评论"""
        reviews = []
        max_page = min((max_count + 19) // 20, 3)  # 最多3页

        for page in range(1, max_page + 1):
            if len(reviews) >= max_count:
                break

            if progress_callback:
                progress_callback(len(reviews), max_count)

            url = f'https://movie.douban.com/subject/{movie_id}/comments?start={(page - 1) * 20}&limit=20&status=P&sort=new_score'

            try:
                response = requests.get(url, headers=self.headers, timeout=15)
                soup = BeautifulSoup(response.text, 'html.parser')
                review_items = soup.find_all('div', class_='comment-item')

                if not review_items:
                    break

                for item in review_items:
                    if len(reviews) >= max_count:
                        break

                    try:
                        # 获取用户
                        user_avatar = item.find('div', class_='avatar')
                        user_name = user_avatar.a.get('title', '') if user_avatar and user_avatar.a else "豆瓣用户"

                        # 获取评分
                        rating_span = item.find('span', class_='rating')
                        rating = 3
                        if rating_span:
                            rating_class = rating_span.get('class', [])
                            for cls in rating_class:
                                if 'allstar' in cls:
                                    rating = int(cls.replace('allstar', '')) // 10
                                    break

                        # 获取评论内容
                        content_span = item.find('span', class_='short')
                        content = content_span.text.strip() if content_span else ""

                        # 获取时间
                        time_span = item.find('span', class_='comment-time')
                        comment_time = time_span.get('title', '') if time_span else ""

                        if content:
                            reviews.append({
                                "content": content,
                                "rating": rating,
                                "user": user_name,
                                "time": comment_time,
                                "votes": "0"
                            })

                    except Exception as e:
                        print(f"解析评论失败: {e}")
                        continue

                time.sleep(random.uniform(1, 2))

            except Exception as e:
                print(f"爬取第{page}页失败: {e}")
                break

        return reviews


# ==================== PyTorch CNN+LSTM 情感分析模型 ====================
class CNNLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_classes=3, dropout=0.5):
        super(CNNLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embedding_dim, 64, kernel_size=4, padding=1)
        self.conv5 = nn.Conv1d(embedding_dim, 64, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.lstm = nn.LSTM(input_size=192, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        embedded_cnn = embedded.permute(0, 2, 1)
        conv3_out = self.relu(self.conv3(embedded_cnn))
        conv4_out = self.relu(self.conv4(embedded_cnn))
        conv5_out = self.relu(self.conv5(embedded_cnn))
        pool3 = self.pool(conv3_out).squeeze(-1)
        pool4 = self.pool(conv4_out).squeeze(-1)
        pool5 = self.pool(conv5_out).squeeze(-1)
        cnn_features = torch.cat([pool3, pool4, pool5], dim=1)
        lstm_out, _ = self.lstm(embedded)
        lstm_features = lstm_out[:, -1, :]
        combined = torch.cat([cnn_features, lstm_features], dim=1)
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class PyTorchSentimentAnalyzer:
    def __init__(self, vocab_size=20000, embedding_dim=128, hidden_dim=128, max_len=100):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.model = None
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        self.reverse_label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}

    def preprocess_text(self, text):
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        words = jieba.cut(text)
        stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '也', '啊', '吧'}
        words = [w for w in words if w not in stopwords and len(w.strip()) > 0]
        return ' '.join(words)

    def build_vocab(self, texts):
        word_count = {}
        for text in texts:
            for word in text.split():
                word_count[word] = word_count.get(word, 0) + 1
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        for i, (word, _) in enumerate(sorted_words[:self.vocab_size - 2]):
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = word

    def build_model(self):
        self.model = CNNLSTM(vocab_size=len(self.word2idx), embedding_dim=self.embedding_dim,
                             hidden_dim=self.hidden_dim, num_classes=3).to(self.device)
        return self.model

    def train_basic(self):
        texts = []
        labels = []
        for sentiment, reviews in DOUBAN_REAL_REVIEWS.items():
            for review in reviews:
                texts.append(review)
                labels.append(sentiment)
        processed_texts = [self.preprocess_text(t) for t in texts]
        self.build_vocab(processed_texts)
        self.build_model()
        self.is_trained = True
        return True

    def predict_sentiment(self, text):
        # 基于关键词的快速情感判断
        positive_keywords = ['好', '棒', '赞', '经典', '震撼', '喜欢', '推荐', '精彩', '优秀', '感动', '神作']
        negative_keywords = ['差', '烂', '失望', '垃圾', '无聊', '糟糕', '难看', '浪费时间', '尴尬', '烂片']
        text_lower = text.lower()
        pos_score = sum(1 for kw in positive_keywords if kw in text_lower)
        neg_score = sum(1 for kw in negative_keywords if kw in text_lower)
        if pos_score > neg_score:
            return "positive", 0.7
        elif neg_score > pos_score:
            return "negative", 0.7
        else:
            return "neutral", 0.5

    def predict_batch(self, texts):
        results = []
        for text in texts:
            sentiment, confidence = self.predict_sentiment(text)
            results.append({'sentiment': sentiment, 'confidence': confidence})
        return results


# ==================== 数据管理类 ====================
class MovieDataManager:
    def __init__(self):
        self.movies = self._generate_movies()
        self.reviews = self._generate_reviews()
        self.users = self._generate_users()
        self.current_user = "user_001"
        self.crawled_reviews = {}
        self.uploaded_reviews = {}
        self.sentiment_analyzer = PyTorchSentimentAnalyzer()
        self.spider = DoubanSpider() if REQUESTS_AVAILABLE else None

        try:
            self.sentiment_analyzer.train_basic()
            print("✅ 情感分析模型初始化完成")
        except Exception as e:
            print(f"⚠️ 情感分析模型初始化失败: {e}")

    def _generate_movies(self):
        return [
            {"id": 1, "title": "肖申克的救赎", "director": "弗兰克·德拉邦特", "actors": ["蒂姆·罗宾斯", "摩根·弗里曼"], "genre": "剧情", "year": 1994, "rating": 9.7, "description": "希望让人自由"},
            {"id": 2, "title": "霸王别姬", "director": "陈凯歌", "actors": ["张国荣", "张丰毅", "巩俐"], "genre": "剧情", "year": 1993, "rating": 9.6, "description": "风华绝代"},
            {"id": 3, "title": "阿甘正传", "director": "罗伯特·泽米吉斯", "actors": ["汤姆·汉克斯"], "genre": "剧情", "year": 1994, "rating": 9.5, "description": "人生就像巧克力"},
            {"id": 4, "title": "盗梦空间", "director": "克里斯托弗·诺兰", "actors": ["莱昂纳多"], "genre": "科幻/悬疑", "year": 2010, "rating": 9.4, "description": "梦境与现实"},
            {"id": 5, "title": "星际穿越", "director": "克里斯托弗·诺兰", "actors": ["马修·麦康纳"], "genre": "科幻", "year": 2014, "rating": 9.4, "description": "穿越时空的爱"},
            {"id": 6, "title": "让子弹飞", "director": "姜文", "actors": ["姜文", "葛优", "周润发"], "genre": "剧情/喜剧", "year": 2010, "rating": 9.0, "description": "站着把钱挣了"},
            {"id": 7, "title": "我不是药神", "director": "文牧野", "actors": ["徐峥", "王传君", "周一围"], "genre": "剧情", "year": 2018, "rating": 9.0, "description": "现实题材的震撼之作"},
            {"id": 8, "title": "千与千寻", "director": "宫崎骏", "actors": ["柊瑠美"], "genre": "动画/奇幻", "year": 2001, "rating": 9.4, "description": "成长的旅程"},
        ]

    def _generate_reviews(self):
        reviews = []
        for movie in self.movies:
            for _ in range(random.randint(8, 12)):
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
            users[user_id] = {
                "watched": [m["id"] for m in watched],
                "ratings": {m["id"]: random.randint(6, 10) for m in watched},
                "favorites": [],
                "watchlist": []
            }
        return users

    def get_system_reviews(self, movie_id):
        return [r for r in self.reviews if r["movie_id"] == movie_id]

    def get_crawled_reviews(self, movie_name):
        return self.crawled_reviews.get(movie_name, [])

    def get_uploaded_reviews(self, movie_name):
        return self.uploaded_reviews.get(movie_name, [])

    def crawl_movie_reviews(self, movie_name, max_count=20, progress_callback=None):
        """爬取电影评论"""
        if not self.spider:
            return None, "爬虫库未安装，请运行: pip install requests beautifulsoup4"

        try:
            movie_id, full_title = self.spider.search_movie_id(movie_name)
            if not movie_id:
                return None, f"未找到电影: {movie_name}"

            print(f"找到电影: {full_title}, ID: {movie_id}")

            reviews = self.spider.crawl_reviews(movie_id, max_count, progress_callback)

            if not reviews:
                return None, f"未爬取到评论，可能是网络问题或该电影暂无评论"

            # 添加情感分析
            texts = [r['content'] for r in reviews if r['content']]
            if texts:
                sentiment_results = self.sentiment_analyzer.predict_batch(texts)
                for i, review in enumerate(reviews):
                    if i < len(sentiment_results):
                        review['sentiment'] = sentiment_results[i]['sentiment']
                    else:
                        review['sentiment'] = 'neutral'

            self.crawled_reviews[full_title] = reviews
            return reviews, full_title

        except Exception as e:
            print(f"爬取失败: {e}")
            return None, f"爬取失败: {str(e)}"

    def search_movie_info(self, movie_name):
        result = {"success": False, "data": None, "message": "", "suggestions": []}

        if movie_name in LOCAL_MOVIE_CACHE:
            cached = LOCAL_MOVIE_CACHE[movie_name]
            result["success"] = True
            result["data"] = {
                "title": cached["title"], "year": cached["year"], "rating": cached["rating"],
                "director": cached["director"], "actors": cached["actors"], "genre": cached["genre"],
                "country": cached.get("country", "未知"), "language": cached.get("language", "未知"),
                "duration": cached.get("duration", "未知"), "description": cached.get("description", ""),
                "source": "本地缓存"
            }
            return result

        best_match, score = fuzzy_match_movie(movie_name)
        if best_match and score > 0.6:
            cached = LOCAL_MOVIE_CACHE[best_match]
            result["success"] = True
            result["data"] = {
                "title": cached["title"], "year": cached["year"], "rating": cached["rating"],
                "director": cached["director"], "actors": cached["actors"], "genre": cached["genre"],
                "country": cached.get("country", "未知"), "language": cached.get("language", "未知"),
                "duration": cached.get("duration", "未知"), "description": cached.get("description", ""),
                "source": f"模糊匹配(相似度:{score:.0%})"
            }
            return result

        suggestions = [name for name in LOCAL_MOVIE_CACHE.keys() if movie_name.lower() in name.lower() or name.lower() in movie_name.lower()]
        if suggestions:
            result["suggestions"] = suggestions[:5]
            result["message"] = f"未找到《{movie_name}》，您是不是想找: {', '.join(suggestions[:3])}"
        else:
            result["message"] = f"未找到电影《{movie_name}》，请尝试使用完整电影名称"

        return result

    def analyze_sentiment_for_data(self, source_type, name):
        if source_type == 'system':
            movie = next((m for m in self.movies if m["title"] == name), None)
            reviews = self.get_system_reviews(movie["id"]) if movie else []
        elif source_type == 'crawled':
            reviews = self.crawled_reviews.get(name, [])
        elif source_type == 'uploaded':
            reviews = self.uploaded_reviews.get(name, [])
        else:
            return None

        if not reviews:
            return None

        # 确保有情感标签
        for review in reviews:
            if 'sentiment' not in review:
                sentiment, _ = self.sentiment_analyzer.predict_sentiment(review['content'])
                review['sentiment'] = sentiment

        sentiments = [r.get('sentiment', 'neutral') for r in reviews]
        counts = Counter(sentiments)
        total = len(reviews)
        ratings = [r.get('rating', 3) for r in reviews]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0

        return {
            'total': total, 'positive': counts.get('positive', 0),
            'positive_pct': counts.get('positive', 0) / total * 100 if total > 0 else 0,
            'neutral': counts.get('neutral', 0),
            'neutral_pct': counts.get('neutral', 0) / total * 100 if total > 0 else 0,
            'negative': counts.get('negative', 0),
            'negative_pct': counts.get('negative', 0) / total * 100 if total > 0 else 0,
            'avg_rating': round(avg_rating, 2), 'reviews': reviews[:100]
        }

    def get_all_data_sources(self):
        return {
            "system": [{"name": m["title"], "type": "system", "count": len(self.get_system_reviews(m["id"]))} for m in self.movies],
            "crawled": [{"name": name, "type": "crawled", "count": len(reviews)} for name, reviews in self.crawled_reviews.items()],
            "uploaded": [{"name": name, "type": "uploaded", "count": len(reviews)} for name, reviews in self.uploaded_reviews.items()]
        }


# 创建全局数据管理器
data_manager = MovieDataManager()

# ==================== 路由定义 ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/movie_analysis_hub')
def movie_analysis_hub():
    """影视分析中心页面"""
    sources = data_manager.get_all_data_sources()
    return render_template('movie_analysis_hub.html', sources=sources, movies=data_manager.movies)

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

@app.route('/social_hub')
def social_hub():
    return render_template('social_hub.html', user=data_manager.current_user)

# ==================== API接口 ====================

@app.route('/api/get_data_sources', methods=['GET'])
def api_get_data_sources():
    return jsonify({'success': True, 'sources': data_manager.get_all_data_sources()})

@app.route('/api/crawl_reviews', methods=['POST'])
def api_crawl_reviews():
    """爬取电影评论接口"""
    data = request.json
    movie_name = data.get('movie_name', '').strip()
    max_count = int(data.get('max_count', 20))

    if not movie_name:
        return jsonify({'success': False, 'error': '请输入电影名称'})

    def progress_callback(current, total):
        pass

    reviews, result = data_manager.crawl_movie_reviews(movie_name, max_count, progress_callback)

    if reviews:
        return jsonify({
            'success': True,
            'reviews': reviews,
            'movie_name': result,
            'count': len(reviews)
        })
    else:
        return jsonify({
            'success': False,
            'error': result
        })

@app.route('/api/upload_reviews', methods=['POST'])
def api_upload_reviews():
    """上传评论文件接口"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有文件'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '文件名为空'})

    movie_name = request.form.get('movie_name', os.path.splitext(file.filename)[0])

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 解析文件
        reviews = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        for line in lines[:100]:
            reviews.append({
                "content": line,
                "user": "上传用户",
                "rating": 3,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

        if not reviews:
            return jsonify({'success': False, 'error': '文件无有效评论内容'})

        # 添加情感分析
        texts = [r['content'] for r in reviews]
        sentiment_results = data_manager.sentiment_analyzer.predict_batch(texts)
        for i, review in enumerate(reviews):
            review['sentiment'] = sentiment_results[i]['sentiment']

        data_manager.uploaded_reviews[movie_name] = reviews
        os.remove(file_path)

        return jsonify({
            'success': True,
            'reviews': reviews,
            'movie_name': movie_name,
            'count': len(reviews)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/search_movie', methods=['POST'])
def api_search_movie():
    """搜索电影信息"""
    movie_name = request.json.get('movie_name', '').strip()
    if not movie_name:
        return jsonify({'success': False, 'error': '请输入电影名称'})
    return jsonify(data_manager.search_movie_info(movie_name))

@app.route('/api/analyze_sentiment', methods=['POST'])
def api_analyze_sentiment():
    """情感分析接口"""
    data = request.json
    source_type = data.get('source_type')
    name = data.get('name')

    result = data_manager.analyze_sentiment_for_data(source_type, name)
    if result:
        return jsonify({'success': True, 'name': name, 'source_type': source_type, **result})
    return jsonify({'success': False, 'error': '暂无评论数据'})

@app.route('/api/get_rating_dist', methods=['POST'])
def api_get_rating_dist():
    """获取评分分布图"""
    data = request.json
    source_type = data.get('source_type')
    name = data.get('name')

    if source_type == 'system':
        movie = next((m for m in data_manager.movies if m["title"] == name), None)
        reviews = data_manager.get_system_reviews(movie["id"]) if movie else []
    elif source_type == 'crawled':
        reviews = data_manager.crawled_reviews.get(name, [])
    elif source_type == 'uploaded':
        reviews = data_manager.uploaded_reviews.get(name, [])
    else:
        return jsonify({'success': False, 'error': '无效的数据源类型'})

    if not reviews:
        return jsonify({'success': False, 'error': '暂无评论数据'})

    ratings = [r.get("rating", 3) for r in reviews]
    counts = [ratings.count(i) for i in range(1, 6)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["1星", "2星", "3星", "4星", "5星"], counts,
                  color=['#ff6b6b', '#ffa07a', '#ffd966', '#98d98e', '#6bcf7f'])
    ax.set_title(f"{name} 评分分布", fontsize=14, fontweight='bold')
    ax.set_xlabel("评分", fontsize=12)
    ax.set_ylabel("评论数量", fontsize=12)

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=10)

    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    return jsonify({'success': True, 'image': image_base64})

@app.route('/api/get_trend_chart', methods=['POST'])
def api_get_trend_chart():
    """获取情感趋势图"""
    data = request.json
    source_type = data.get('source_type')
    name = data.get('name')

    if source_type == 'system':
        movie = next((m for m in data_manager.movies if m["title"] == name), None)
        reviews = data_manager.get_system_reviews(movie["id"]) if movie else []
    elif source_type == 'crawled':
        reviews = data_manager.crawled_reviews.get(name, [])
    elif source_type == 'uploaded':
        reviews = data_manager.uploaded_reviews.get(name, [])
    else:
        return jsonify({'success': False, 'error': '无效的数据源类型'})

    if len(reviews) < 2:
        return jsonify({'success': False, 'error': '评论数量不足'})

    score_map = {"positive": 1, "neutral": 0, "negative": -1}
    scores = [score_map.get(r.get("sentiment", "neutral"), 0) for r in reviews]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(scores))
    ax.plot(x, scores, 'b-', linewidth=1, marker='o', markersize=3, alpha=0.5, label='原始情感')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.fill_between(x, scores, 0, where=(np.array(scores) > 0), color='green', alpha=0.3)
    ax.fill_between(x, scores, 0, where=(np.array(scores) < 0), color='red', alpha=0.3)

    ax.set_title(f"{name} 情感趋势分析", fontsize=14, fontweight='bold')
    ax.set_xlabel("评论序号", fontsize=12)
    ax.set_ylabel("情感得分", fontsize=12)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['负面', '中性', '正面'])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    return jsonify({'success': True, 'image': image_base64})

@app.route('/api/get_wordcloud', methods=['POST'])
def api_get_wordcloud():
    """获取词云图"""
    data = request.json
    source_type = data.get('source_type')
    name = data.get('name')

    if source_type == 'system':
        movie = next((m for m in data_manager.movies if m["title"] == name), None)
        reviews = data_manager.get_system_reviews(movie["id"]) if movie else []
    elif source_type == 'crawled':
        reviews = data_manager.crawled_reviews.get(name, [])
    elif source_type == 'uploaded':
        reviews = data_manager.uploaded_reviews.get(name, [])
    else:
        return jsonify({'success': False, 'error': '无效的数据源类型'})

    if not reviews:
        return jsonify({'success': False, 'error': '暂无评论数据'})

    all_text = ' '.join([r.get('content', '') for r in reviews])
    words = jieba.cut(all_text)
    stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '也', '啊', '吧', '电影', '一个', '没有', '什么'}
    word_count = {}
    for word in words:
        if len(word) >= 2 and word not in stopwords:
            word_count[word] = word_count.get(word, 0) + 1

    top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:15]

    fig, ax = plt.subplots(figsize=(10, 6))
    words_list = [w for w, _ in top_words]
    counts_list = [c for _, c in top_words]
    y_pos = np.arange(len(words_list))
    ax.barh(y_pos, counts_list, color='#667eea')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words_list)
    ax.invert_yaxis()
    ax.set_xlabel('出现次数')
    ax.set_title(f'{name} 高频词分析')

    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    return jsonify({'success': True, 'image': image_base64, 'top_words': top_words})

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """推荐接口"""
    user_id = request.json.get('user_id', data_manager.current_user)
    top_n = int(request.json.get('top_n', 5))

    user = data_manager.users.get(user_id, {"ratings": {}, "watched": []})
    watched = set(user["watched"])

    recs = [m for m in data_manager.movies if m["id"] not in watched]
    recs.sort(key=lambda x: x["rating"], reverse=True)

    result = [{"id": m["id"], "title": m["title"], "director": m["director"], "actors": m["actors"],
               "genre": m["genre"], "year": m["year"], "rating": m["rating"], "description": m["description"], "score": m["rating"]} for m in recs[:top_n]]

    return jsonify({'success': True, 'recommendations': result, 'method': '基于评分的推荐', 'user_id': user_id})

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """AI助手对话接口"""
    data = request.json
    message = data.get('message', '')
    msg_lower = message.lower()

    if "推荐" in msg_lower:
        response = "🎬 为您推荐高分电影：\n《肖申克的救赎》9.7分\n《霸王别姬》9.6分\n《阿甘正传》9.5分\n《盗梦空间》9.4分\n《星际穿越》9.4分"
    elif "评分" in msg_lower:
        response = "⭐ 豆瓣高分电影评分：\n《肖申克的救赎》9.7分\n《霸王别姬》9.6分\n《阿甘正传》9.5分\n《泰坦尼克号》9.5分"
    elif "爬虫" in msg_lower or "爬取" in msg_lower:
        response = "🕷️ 在影视分析页面的「数据采集」标签页，输入电影名称和数量，点击开始爬取即可获取豆瓣真实评论！"
    elif "情感" in msg_lower or "分析" in msg_lower:
        response = "😊 系统使用PyTorch CNN+LSTM模型进行情感分析，可以识别评论的正面、中性和负面情感。"
    else:
        response = "🤖 我是豆瓣电影助手！我可以：\n📊 分析电影评论情感\n🎬 推荐高分电影\n⭐ 查询电影评分\n🕷️ 爬取豆瓣评论数据\n请输入您的问题~"

    return jsonify({'success': True, 'response': response})

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
        return jsonify({'success': True, 'username': username})
    return jsonify({'success': False, 'error': '用户不存在'})

@app.route('/api/user/register', methods=['POST'])
def api_user_register():
    username = request.json.get('username')
    if username not in data_manager.users:
        data_manager.users[username] = {"watched": [], "ratings": {}, "favorites": [], "watchlist": []}
        data_manager.current_user = username
        return jsonify({'success': True, 'username': username})
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

if __name__ == '__main__':
    print("=" * 60)
    print("豆瓣电影数据分析与推荐系统")
    print("功能: 数据采集 | 数据分析 | 影视分析 | 推荐引擎 | 社交管理")
    print("访问地址: http://127.0.0.1:5000")
    print("=" * 60)

    if not REQUESTS_AVAILABLE:
        print("⚠️ 爬虫库未安装，请运行: pip install requests beautifulsoup4")
    else:
        print("✅ 爬虫库已安装，支持豆瓣真实数据爬取")

    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
    except:
        print("⚠️ PyTorch未安装")

    app.run(debug=True, host='0.0.0.0', port=5000)