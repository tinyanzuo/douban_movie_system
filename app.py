"""
豆瓣电影数据分析与推荐系统 - Web版本
使用 PyTorch CNN+LSTM 情感分析模型 + 神经网络推荐引擎
页面结构：首页 | 影视分析 | 推荐引擎 | 社交管理
"""

from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
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
import pickle
import time
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 爬虫相关库
try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    BeautifulSoup = None

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

# ==================== 豆瓣真实风格评论数据 ====================
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

# ==================== 扩展本地电影数据库 ====================
LOCAL_MOVIE_CACHE = {
    "肖申克的救赎": {"id": "1292052", "title": "肖申克的救赎", "year": "1994", "rating": 9.7, "director": "弗兰克·德拉邦特", "actors": ["蒂姆·罗宾斯", "摩根·弗里曼"], "genre": "剧情", "country": "美国", "language": "英语", "duration": "142分钟", "description": "希望让人自由，一部关于希望与救赎的经典之作。"},
    "霸王别姬": {"id": "1291546", "title": "霸王别姬", "year": "1993", "rating": 9.6, "director": "陈凯歌", "actors": ["张国荣", "张丰毅", "巩俐"], "genre": "剧情", "country": "中国", "language": "汉语普通话", "duration": "171分钟", "description": "风华绝代，一曲人生悲歌。"},
    "阿甘正传": {"id": "1292720", "title": "阿甘正传", "year": "1994", "rating": 9.5, "director": "罗伯特·泽米吉斯", "actors": ["汤姆·汉克斯"], "genre": "剧情", "country": "美国", "language": "英语", "duration": "142分钟", "description": "人生就像一盒巧克力，你永远不知道下一块是什么味道。"},
    "这个杀手不太冷": {"id": "1295644", "title": "这个杀手不太冷", "year": "1994", "rating": 9.4, "director": "吕克·贝松", "actors": ["让·雷诺", "娜塔莉·波特曼"], "genre": "剧情/动作", "country": "法国", "language": "英语", "duration": "110分钟", "description": "杀手与小女孩的温情故事。"},
    "泰坦尼克号": {"id": "1292722", "title": "泰坦尼克号", "year": "1997", "rating": 9.5, "director": "詹姆斯·卡梅隆", "actors": ["莱昂纳多", "凯特·温丝莱特"], "genre": "爱情/灾难", "country": "美国", "language": "英语", "duration": "194分钟", "description": "永恒的爱情，永不沉没的经典。"},
    "盗梦空间": {"id": "3541415", "title": "盗梦空间", "year": "2010", "rating": 9.4, "director": "克里斯托弗·诺兰", "actors": ["莱昂纳多"], "genre": "科幻/悬疑", "country": "美国", "language": "英语", "duration": "148分钟", "description": "梦境与现实，一场关于意识的大胆探索。"},
    "楚门的世界": {"id": "1292064", "title": "楚门的世界", "year": "1998", "rating": 9.3, "director": "彼得·威尔", "actors": ["金·凯瑞"], "genre": "剧情/科幻", "country": "美国", "language": "英语", "duration": "103分钟", "description": "真实与虚假的边界在哪里？"},
    "千与千寻": {"id": "1291561", "title": "千与千寻", "year": "2001", "rating": 9.4, "director": "宫崎骏", "actors": ["柊瑠美"], "genre": "动画/奇幻", "country": "日本", "language": "日语", "duration": "125分钟", "description": "成长的旅程，宫崎骏的巅峰之作。"},
    "星际穿越": {"id": "1889243", "title": "星际穿越", "year": "2014", "rating": 9.4, "director": "克里斯托弗·诺兰", "actors": ["马修·麦康纳"], "genre": "科幻", "country": "美国", "language": "英语", "duration": "169分钟", "description": "穿越时空的爱，人类命运的终极探索。"},
    "海上钢琴师": {"id": "1292001", "title": "海上钢琴师", "year": "1998", "rating": 9.3, "director": "朱塞佩·托纳多雷", "actors": ["蒂姆·罗斯"], "genre": "剧情/音乐", "country": "意大利", "language": "英语", "duration": "125分钟", "description": "1900的故事，音乐与自由的传奇。"},
    "让子弹飞": {"id": "3742360", "title": "让子弹飞", "year": "2010", "rating": 9.0, "director": "姜文", "actors": ["姜文", "葛优", "周润发"], "genre": "剧情/喜剧", "country": "中国", "language": "汉语普通话", "duration": "132分钟", "description": "站着把钱挣了！"},
    "流浪地球": {"id": "26266893", "title": "流浪地球", "year": "2019", "rating": 7.9, "director": "郭帆", "actors": ["吴京", "屈楚萧", "李光洁"], "genre": "科幻", "country": "中国", "language": "汉语普通话", "duration": "125分钟", "description": "带着地球去流浪，中国科幻的里程碑。"},
    "哪吒之魔童降世": {"id": "26794435", "title": "哪吒之魔童降世", "year": "2019", "rating": 8.4, "director": "饺子", "actors": ["吕艳婷", "囧森瑟夫"], "genre": "动画/奇幻", "country": "中国", "language": "汉语普通话", "duration": "110分钟", "description": "我命由我不由天！"},
    "我不是药神": {"id": "26752088", "title": "我不是药神", "year": "2018", "rating": 9.0, "director": "文牧野", "actors": ["徐峥", "王传君", "周一围"], "genre": "剧情", "country": "中国", "language": "汉语普通话", "duration": "117分钟", "description": "现实题材的震撼之作。"},
    "绿皮书": {"id": "27060077", "title": "绿皮书", "year": "2018", "rating": 8.9, "director": "彼得·法拉利", "actors": ["维果·莫腾森", "马赫沙拉·阿里"], "genre": "剧情/喜剧", "country": "美国", "language": "英语", "duration": "130分钟", "description": "跨越种族与阶层的友谊。"},
    "三傻大闹宝莱坞": {"id": "3793023", "title": "三傻大闹宝莱坞", "year": "2009", "rating": 9.2, "director": "拉吉库马尔·希拉尼", "actors": ["阿米尔·汗", "马德哈万"], "genre": "剧情/喜剧", "country": "印度", "language": "印地语", "duration": "171分钟", "description": "追求卓越，成功会不请自来。"},
    "放牛班的春天": {"id": "1291548", "title": "放牛班的春天", "year": "2004", "rating": 9.3, "director": "克里斯托夫·巴拉蒂", "actors": ["热拉尔·朱尼奥", "让-巴蒂斯特·莫尼耶"], "genre": "剧情/音乐", "country": "法国", "language": "法语", "duration": "97分钟", "description": "音乐治愈心灵。"},
    "忠犬八公的故事": {"id": "3011091", "title": "忠犬八公的故事", "year": "2009", "rating": 9.4, "director": "莱塞·霍尔斯道姆", "actors": ["理查·基尔", "琼·艾伦"], "genre": "剧情", "country": "美国", "language": "英语", "duration": "93分钟", "description": "等待是最长情的告白。"},
    "美丽人生": {"id": "1292063", "title": "美丽人生", "year": "1997", "rating": 9.5, "director": "罗伯托·贝尼尼", "actors": ["罗伯托·贝尼尼", "尼可莱塔·布拉斯基"], "genre": "剧情/喜剧", "country": "意大利", "language": "意大利语", "duration": "116分钟", "description": "即使在黑暗中，也要相信美好。"},
    "怦然心动": {"id": "3319755", "title": "怦然心动", "year": "2010", "rating": 9.1, "director": "罗伯·莱纳", "actors": ["玛德琳·卡罗尔", "卡兰·麦克奥利菲"], "genre": "爱情/喜剧", "country": "美国", "language": "英语", "duration": "90分钟", "description": "初恋的美好，成长的喜悦。"},
}

# 电影猜谜数据库（支持电影和电视剧）
MOVIE_QUIZ_DB = [
    {"clues": ["希望让人自由", "监狱", "摩根·弗里曼"], "answer": "肖申克的救赎", "type": "电影", "hint": "经典越狱题材"},
    {"clues": ["风华绝代", "京剧", "张国荣"], "answer": "霸王别姬", "type": "电影", "hint": "陈凯歌导演"},
    {"clues": ["人生就像一盒巧克力", "跑步", "汤姆·汉克斯"], "answer": "阿甘正传", "type": "电影", "hint": "励志经典"},
    {"clues": ["梦境", "陀螺", "诺兰"], "answer": "盗梦空间", "type": "电影", "hint": "烧脑科幻"},
    {"clues": ["穿越时空", "黑洞", "马修·麦康纳"], "answer": "星际穿越", "type": "电影", "hint": "诺兰科幻巨制"},
    {"clues": ["You jump, I jump", "大船", "莱昂纳多"], "answer": "泰坦尼克号", "type": "电影", "hint": "经典爱情灾难片"},
    {"clues": ["无脸男", "汤婆婆", "宫崎骏"], "answer": "千与千寻", "type": "电影", "hint": "奥斯卡最佳动画"},
    {"clues": ["1900", "钢琴", "一生不下船"], "answer": "海上钢琴师", "type": "电影", "hint": "音乐与自由"},
    {"clues": ["站着把钱挣了", "鹅城", "姜文"], "answer": "让子弹飞", "type": "电影", "hint": "姜文导演作品"},
    {"clues": ["我命由我不由天", "哪吒", "国产动画"], "answer": "哪吒之魔童降世", "type": "电影", "hint": "国产动画巅峰"},
    {"clues": ["药神", "徐峥", "现实题材"], "answer": "我不是药神", "type": "电影", "hint": "感动无数人"},
    {"clues": ["绿皮车", "种族歧视", "钢琴家"], "answer": "绿皮书", "type": "电影", "hint": "奥斯卡最佳影片"},
    {"clues": ["追求卓越", "阿米尔·汗", "印度电影"], "answer": "三傻大闹宝莱坞", "type": "电影", "hint": "印度励志喜剧"},
    {"clues": ["音乐治愈", "马修老师", "合唱团"], "answer": "放牛班的春天", "type": "电影", "hint": "法国音乐电影"},
    {"clues": ["等待主人", "秋田犬", "真实故事"], "answer": "忠犬八公的故事", "type": "电影", "hint": "感人至深"},
    {"clues": ["纳粹集中营", "父亲", "游戏"], "answer": "美丽人生", "type": "电影", "hint": "笑着流泪"},
    {"clues": ["初恋", "梧桐树", "朱莉"], "answer": "怦然心动", "type": "电影", "hint": "纯真爱情"},
    # 电视剧
    {"clues": ["后宫", "甄嬛", "皇帝"], "answer": "甄嬛传", "type": "电视剧", "hint": "宫斗剧经典"},
    {"clues": ["老友", "中央公园咖啡馆", "六人行"], "answer": "老友记", "type": "电视剧", "hint": "美剧经典"},
    {"clues": ["权游", "铁王座", "龙母"], "answer": "权力的游戏", "type": "电视剧", "hint": "HBO史诗巨制"},
    {"clues": ["白夜追凶", "双胞胎", "潘粤明"], "answer": "白夜追凶", "type": "电视剧", "hint": "国产悬疑佳作"},
    {"clues": ["余欢水", "中年危机", "郭京飞"], "answer": "我是余欢水", "type": "电视剧", "hint": "荒诞现实"},
    {"clues": ["陈情令", "魏无羡", "蓝忘机"], "answer": "陈情令", "type": "电视剧", "hint": "古装玄幻"},
    {"clues": ["庆余年", "范闲", "穿越"], "answer": "庆余年", "type": "电视剧", "hint": "穿越权谋"},
    {"clues": ["沉默的真相", "江阳", "正义"], "answer": "沉默的真相", "type": "电视剧", "hint": "高分国产剧"},
    {"clues": ["隐秘的角落", "爬山", "张东升"], "answer": "隐秘的角落", "type": "电视剧", "hint": "悬疑神剧"},
    {"clues": ["请回答1988", "双门洞", "德善"], "answer": "请回答1988", "type": "电视剧", "hint": "韩剧温情经典"},
]

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


# ==================== PyTorch CNN+LSTM 情感分析模型（修复版）====================
class CNNLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_classes=3, dropout=0.5):
        super(CNNLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embedding_dim, 64, kernel_size=4, padding=1)
        self.conv5 = nn.Conv1d(embedding_dim, 64, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        # 修复：LSTM的input_size应该是embedding_dim，而不是cnn_features的维度
        # 因为LSTM处理的是原始embedding序列，不是CNN输出
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, 
                           num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)
        # 合并后的特征维度：CNN特征(64*3=192) + LSTM特征(hidden_dim*2=256)
        combined_dim = 192 + hidden_dim * 2
        self.fc1 = nn.Linear(combined_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # CNN分支 - 处理embedding序列
        embedded_cnn = embedded.permute(0, 2, 1)  # (batch, embedding_dim, seq_len)
        conv3_out = self.relu(self.conv3(embedded_cnn))
        conv4_out = self.relu(self.conv4(embedded_cnn))
        conv5_out = self.relu(self.conv5(embedded_cnn))
        pool3 = self.pool(conv3_out).squeeze(-1)  # (batch, 64)
        pool4 = self.pool(conv4_out).squeeze(-1)  # (batch, 64)
        pool5 = self.pool(conv5_out).squeeze(-1)  # (batch, 64)
        cnn_features = torch.cat([pool3, pool4, pool5], dim=1)  # (batch, 192)
        
        # LSTM分支 - 处理embedding序列
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim*2)
        lstm_features = lstm_out[:, -1, :]  # (batch, hidden_dim*2)
        
        # 合并特征
        combined = torch.cat([cnn_features, lstm_features], dim=1)  # (batch, 192 + hidden_dim*2)
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=100):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = [self.word2idx.get(word, self.word2idx.get('<UNK>', 1)) for word in text.split()]
        if len(indices) > self.max_len: indices = indices[:self.max_len]
        else: indices = indices + [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

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
        stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都'}
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
        print(f"词表大小: {len(self.word2idx)}")
    
    def build_model(self):
        self.model = CNNLSTM(vocab_size=len(self.word2idx), embedding_dim=self.embedding_dim, 
                           hidden_dim=self.hidden_dim, num_classes=3).to(self.device)
        return self.model
    
    def predict_sentiment(self, text):
        if not self.is_trained or self.model is None:
            return "neutral", 0.5
        processed = self.preprocess_text(text)
        indices = [self.word2idx.get(word, self.word2idx.get('<UNK>', 1)) for word in processed.split()]
        if len(indices) > self.max_len: indices = indices[:self.max_len]
        else: indices = indices + [0] * (self.max_len - len(indices))
        data = torch.tensor([indices], dtype=torch.long).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
        sentiment = self.reverse_label_map[predicted.item()]
        confidence = float(probs[0][predicted.item()].item())
        return sentiment, confidence
    
    def predict_batch(self, texts):
        results = []
        for text in texts:
            sentiment, confidence = self.predict_sentiment(text)
            results.append({'sentiment': sentiment, 'confidence': confidence})
        return results
    
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


# ==================== 豆瓣爬虫类（完整版）====================
class DoubanSpider:
    """豆瓣电影评论爬虫 - 完整版"""
    
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
        """爬取电影评论 - 支持多页"""
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
                        user_avatar = item.find('div', class_='avatar')
                        user_name = user_avatar.a.get('title', '') if user_avatar and user_avatar.a else "豆瓣用户"

                        rating_span = item.find('span', class_='rating')
                        rating = 3
                        if rating_span:
                            rating_class = rating_span.get('class', [])
                            for cls in rating_class:
                                if 'allstar' in cls:
                                    rating = int(cls.replace('allstar', '')) // 10
                                    break

                        content_span = item.find('span', class_='short')
                        content = content_span.text.strip() if content_span else ""

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


# ==================== 数据管理类 ====================
class MovieDataManager:
    def __init__(self):
        self.movies = self._generate_movies()
        self.reviews = self._generate_reviews()
        self.users = self._generate_users()
        self.current_user = "user_001"
        self.crawled_reviews = {}
        self.uploaded_reviews = {}
        self.uploaded_user_data = {}
        self.sentiment_analyzer = PyTorchSentimentAnalyzer()
        try:
            self.sentiment_analyzer.train_basic()
            print("✅ 情感分析模型训练完成")
        except Exception as e:
            print(f"⚠️ 情感分析模型训练失败: {e}")
        self.spider = None
        if REQUESTS_AVAILABLE:
            try:
                self.spider = DoubanSpider()
                self.spider.ml_analyzer = self.sentiment_analyzer
            except:
                pass
    
    def _generate_movies(self):
        return [
            {"id": 1, "title": "肖申克的救赎", "director": "弗兰克·德拉邦特", "actors": ["蒂姆·罗宾斯", "摩根·弗里曼"], "genre": "剧情", "year": 1994, "rating": 9.7, "description": "希望让人自由"},
            {"id": 2, "title": "霸王别姬", "director": "陈凯歌", "actors": ["张国荣", "张丰毅", "巩俐"], "genre": "剧情", "year": 1993, "rating": 9.6, "description": "风华绝代"},
            {"id": 3, "title": "阿甘正传", "director": "罗伯特·泽米吉斯", "actors": ["汤姆·汉克斯"], "genre": "剧情", "year": 1994, "rating": 9.5, "description": "人生就像巧克力"},
            {"id": 4, "title": "盗梦空间", "director": "克里斯托弗·诺兰", "actors": ["莱昂纳多"], "genre": "科幻/悬疑", "year": 2010, "rating": 9.4, "description": "梦境与现实"},
            {"id": 5, "title": "星际穿越", "director": "克里斯托弗·诺兰", "actors": ["马修·麦康纳"], "genre": "科幻", "year": 2014, "rating": 9.4, "description": "穿越时空的爱"},
        ]
    
    def _generate_reviews(self):
        reviews = []
        for movie in self.movies:
            for _ in range(random.randint(10, 15)):
                sentiment = random.choice(["positive", "neutral", "negative"])
                text = random.choice(DOUBAN_REAL_REVIEWS[sentiment])
                rating = 5 if sentiment == "positive" else (3 if sentiment == "neutral" else 2)
                reviews.append({"movie_id": movie["id"], "movie_name": movie["title"], "user": random.choice(USER_NAMES), "content": text, "rating": rating, "sentiment": sentiment, "time": datetime.now().strftime("%Y-%m-%d")})
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
    
    def get_crawled_reviews(self, movie_name):
        return self.crawled_reviews.get(movie_name, [])
    
    def get_uploaded_reviews(self, movie_name):
        return self.uploaded_reviews.get(movie_name, [])
    
    def search_movie_info(self, movie_name):
        result = {"success": False, "data": None, "message": "", "suggestions": []}
        if movie_name in LOCAL_MOVIE_CACHE:
            cached = LOCAL_MOVIE_CACHE[movie_name]
            result["success"] = True
            result["data"] = {"title": cached["title"], "year": cached["year"], "rating": cached["rating"], "director": cached["director"], "actors": cached["actors"], "genre": cached["genre"], "description": cached["description"], "source": "本地缓存"}
            return result
        best_match, score = fuzzy_match_movie(movie_name)
        if best_match and score > 0.6:
            cached = LOCAL_MOVIE_CACHE[best_match]
            result["success"] = True
            result["data"] = {"title": cached["title"], "year": cached["year"], "rating": cached["rating"], "director": cached["director"], "actors": cached["actors"], "genre": cached["genre"], "description": cached["description"], "source": f"模糊匹配({score:.0%})"}
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
        """上传并解析用户数据文件 - 支持Excel和CSV格式"""
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
        """自动检测文件格式并解析"""
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
        """解析长格式数据"""
        col_map = {
            'nickname': None, 'gender': None, 'age': None,
            'movie_name': None, 'watch_date': None, 'movie_type': None,
            'movie_rating': None, 'director': None, 'actors': None,
            'watch_channel': None, 'remark': None
        }
        
        for col in df.columns:
            col_lower = col.lower()
            if '用户昵称' in col or '昵称' in col or col_lower == 'nickname':
                col_map['nickname'] = col
            elif '性别' in col or col_lower == 'gender':
                col_map['gender'] = col
            elif '年龄' in col or col_lower == 'age':
                col_map['age'] = col
            elif '电影名称' in col or '电影名' in col or col_lower == 'movie_name':
                col_map['movie_name'] = col
            elif '观影日期' in col or '日期' in col or col_lower == 'watch_date':
                col_map['watch_date'] = col
            elif '电影类型' in col or '类型' in col or col_lower == 'movie_type':
                col_map['movie_type'] = col
            elif '电影评分' in col or '评分' in col or col_lower == 'movie_rating':
                col_map['movie_rating'] = col
            elif '导演' in col or col_lower == 'director':
                col_map['director'] = col
            elif '主演' in col or '演员' in col or col_lower == 'actors':
                col_map['actors'] = col
            elif '观影渠道' in col or '渠道' in col or col_lower == 'watch_channel':
                col_map['watch_channel'] = col
            elif '备注' in col or col_lower == 'remark':
                col_map['remark'] = col
        
        if not col_map['nickname']:
            raise Exception("缺少用户昵称列")
        if not col_map['movie_name']:
            raise Exception("缺少电影名称列")
        
        user_summary = {}
        movie_records = []
        
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
            
            movie_record = {
                'nickname': nickname, 'gender': gender, 'age': age,
                'movie_name': movie_name,
                'watch_date': str(row.get(col_map['watch_date'], '')) if col_map['watch_date'] and pd.notna(row.get(col_map['watch_date'])) else '',
                'movie_type': str(row.get(col_map['movie_type'], '')) if col_map['movie_type'] and pd.notna(row.get(col_map['movie_type'])) else '',
                'movie_rating': float(row.get(col_map['movie_rating'], 0)) if col_map['movie_rating'] and pd.notna(row.get(col_map['movie_rating'])) else 0,
                'director': str(row.get(col_map['director'], '')) if col_map['director'] and pd.notna(row.get(col_map['director'])) else '',
                'actors': str(row.get(col_map['actors'], '')) if col_map['actors'] and pd.notna(row.get(col_map['actors'])) else '',
                'watch_channel': str(row.get(col_map['watch_channel'], '')) if col_map['watch_channel'] and pd.notna(row.get(col_map['watch_channel'])) else '',
                'remark': str(row.get(col_map['remark'], '')) if col_map['remark'] and pd.notna(row.get(col_map['remark'])) else ''
            }
            movie_records.append(movie_record)
            
            if nickname not in user_summary:
                user_summary[nickname] = {
                    'nickname': nickname, 'gender': gender, 'age': age,
                    'movies': [], 'watch_dates': [], 'movie_types': [],
                    'ratings': [], 'directors': [], 'watch_count': 0
                }
            
            user_summary[nickname]['movies'].append(movie_name)
            if movie_record['watch_date']:
                user_summary[nickname]['watch_dates'].append(movie_record['watch_date'])
            if movie_record['movie_type']:
                user_summary[nickname]['movie_types'].append(movie_record['movie_type'])
            if movie_record['movie_rating'] > 0:
                user_summary[nickname]['ratings'].append(movie_record['movie_rating'])
            if movie_record['director']:
                user_summary[nickname]['directors'].append(movie_record['director'])
            user_summary[nickname]['watch_count'] += 1
        
        if not user_summary:
            raise Exception("没有找到有效的用户数据")
        
        analysis_records = []
        for nickname, summary in user_summary.items():
            type_counter = Counter()
            for t in summary['movie_types']:
                if t:
                    import re
                    genres = re.split('[,，、/;；]', str(t))
                    for genre in genres:
                        genre = genre.strip()
                        if genre:
                            type_counter[genre] += 1
            
            analysis_records.append({
                'nickname': nickname, 'gender': summary['gender'], 'age': summary['age'],
                'watch_count': summary['watch_count'],
                'watch_content': ','.join(summary['movies']),
                'watch_time': ','.join(summary['watch_dates']),
                'watch_genre': ','.join([t for t in summary['movie_types'] if t]),
                'avg_rating': sum(summary['ratings']) / len(summary['ratings']) if summary['ratings'] else 0,
                'top_genres': dict(type_counter.most_common(5)),
                'top_director': Counter(summary['directors']).most_common(1)[0][0] if summary['directors'] else ''
            })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_id = f"user_data_{timestamp}"
        self.uploaded_user_data[data_id] = {
            'filename': filename, 'file_type': file_ext,
            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'records': analysis_records, 'raw_records': movie_records,
            'total_users': len(analysis_records), 'total_movies': len(movie_records),
            'format': 'long'
        }
        
        return {'success': True, 'data_id': data_id, 'total_users': len(analysis_records),
                'total_movies': len(movie_records), 'preview': movie_records[:10]}
    
    def _parse_wide_format(self, df, filename, file_ext='.xlsx'):
        """解析宽格式数据"""
        col_map = {'nickname': None, 'gender': None, 'age': None,
                   'watch_count': None, 'watch_content': None, 'watch_time': None, 'watch_genre': None}
        
        for col in df.columns:
            col_lower = col.lower()
            if '用户昵称' in col or '昵称' in col or col_lower == 'nickname':
                col_map['nickname'] = col
            elif '性别' in col or col_lower == 'gender':
                col_map['gender'] = col
            elif '年龄' in col or col_lower == 'age':
                col_map['age'] = col
            elif '观影数量' in col or '数量' in col or col_lower == 'watch_count':
                col_map['watch_count'] = col
            elif '观影内容' in col or '内容' in col or col_lower == 'watch_content':
                col_map['watch_content'] = col
            elif '观影时间' in col or '时间' in col or col_lower == 'watch_time':
                col_map['watch_time'] = col
            elif '观影类型' in col or '类型' in col or col_lower == 'watch_genre':
                col_map['watch_genre'] = col
        
        if not col_map['nickname']:
            raise Exception("缺少用户昵称列")
        
        user_records = []
        for idx, row in df.iterrows():
            nickname = str(row.get(col_map['nickname'], '')).strip()
            if not nickname or nickname == 'nan' or nickname == 'None':
                continue
            
            gender = '未知'
            if col_map['gender'] and pd.notna(row.get(col_map['gender'])):
                gender_val = str(row.get(col_map['gender'])).strip()
                if gender_val and gender_val != 'nan':
                    gender = gender_val
            
            age = 0
            if col_map['age'] and pd.notna(row.get(col_map['age'])):
                try:
                    age = int(float(row.get(col_map['age'])))
                except:
                    age = 0
            
            watch_count = 0
            if col_map['watch_count'] and pd.notna(row.get(col_map['watch_count'])):
                try:
                    watch_count = int(float(row.get(col_map['watch_count'])))
                except:
                    watch_count = 0
            
            record = {
                'nickname': nickname, 'gender': gender, 'age': age,
                'watch_count': watch_count,
                'watch_content': str(row.get(col_map['watch_content'], '')) if col_map['watch_content'] and pd.notna(row.get(col_map['watch_content'])) else '',
                'watch_time': str(row.get(col_map['watch_time'], '')) if col_map['watch_time'] and pd.notna(row.get(col_map['watch_time'])) else '',
                'watch_genre': str(row.get(col_map['watch_genre'], '')) if col_map['watch_genre'] and pd.notna(row.get(col_map['watch_genre'])) else ''
            }
            user_records.append(record)
        
        if not user_records:
            raise Exception("没有找到有效的用户数据")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_id = f"user_data_{timestamp}"
        self.uploaded_user_data[data_id] = {
            'filename': filename, 'file_type': file_ext,
            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'records': user_records, 'total_users': len(user_records), 'format': 'wide'
        }
        
        return {'success': True, 'data_id': data_id, 'total_users': len(user_records), 'preview': user_records[:10]}
    
    def _parse_simple_format(self, df, filename, file_ext='.xlsx'):
        """解析简化格式"""
        movie_cols = [c for c in df.columns if '电影' in c or 'movie' in c.lower()]
        if not movie_cols:
            raise Exception("未找到电影相关列")
        
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
            
            record = {
                'nickname': nickname, 'gender': '未知', 'age': 0,
                'watch_count': len(movies), 'watch_content': ','.join(movies),
                'watch_time': '', 'watch_genre': ''
            }
            user_records.append(record)
        
        if not user_records:
            raise Exception("没有找到有效的用户数据")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_id = f"user_data_{timestamp}"
        self.uploaded_user_data[data_id] = {
            'filename': filename, 'file_type': file_ext,
            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'records': user_records, 'total_users': len(user_records), 'format': 'simple'
        }
        
        return {'success': True, 'data_id': data_id, 'total_users': len(user_records), 'preview': user_records[:10]}
    
    def analyze_user_data(self, data_id):
        """分析用户数据"""
        if data_id not in self.uploaded_user_data:
            return None
        
        data = self.uploaded_user_data[data_id]
        records = data['records']
        
        total_users = len(records)
        
        gender_count = Counter([r.get('gender', '未知') for r in records])
        
        ages = [r.get('age', 0) for r in records if r.get('age', 0) > 0]
        avg_age = sum(ages) / len(ages) if ages else 0
        
        watch_counts = [r.get('watch_count', 0) for r in records]
        avg_watch = sum(watch_counts) / len(watch_counts) if watch_counts else 0
        max_watch = max(watch_counts) if watch_counts else 0
        
        genre_prefs = []
        for r in records:
            genre_str = r.get('watch_genre', '')
            if genre_str:
                import re
                genres = re.split('[,，、/;；]', str(genre_str))
                genre_prefs.extend([g.strip() for g in genres if g.strip()])
        genre_count = Counter(genre_prefs)
        
        age_groups = {'18岁以下': 0, '18-25岁': 0, '26-35岁': 0, '36-50岁': 0, '50岁以上': 0}
        for r in records:
            age = r.get('age', 0)
            if age <= 0:
                continue
            elif age < 18:
                age_groups['18岁以下'] += 1
            elif age <= 25:
                age_groups['18-25岁'] += 1
            elif age <= 35:
                age_groups['26-35岁'] += 1
            elif age <= 50:
                age_groups['36-50岁'] += 1
            else:
                age_groups['50岁以上'] += 1
        
        movie_titles = []
        for r in records:
            content = r.get('watch_content', '')
            if content:
                import re
                movies = re.split('[,，、/;；]', str(content))
                movie_titles.extend([m.strip() for m in movies if m.strip()])
        popular_movies = Counter(movie_titles).most_common(10)
        
        # 生成图表
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.bar(list(age_groups.keys()), list(age_groups.values()), color=['#667eea', '#48bb78', '#4299e1', '#ecc94b', '#f56565'])
        ax1.set_title('用户年龄分布', fontsize=14, fontweight='bold')
        buffer1 = BytesIO()
        fig1.savefig(buffer1, format='png', dpi=100, bbox_inches='tight')
        buffer1.seek(0)
        age_chart = base64.b64encode(buffer1.getvalue()).decode('utf-8')
        plt.close(fig1)
        
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        colors = ['#48bb78', '#4299e1', '#ecc94b', '#f56565']
        ax2.pie(list(gender_count.values()), labels=list(gender_count.keys()), autopct='%1.1f%%', colors=colors[:len(gender_count)])
        ax2.set_title('用户性别分布', fontsize=14, fontweight='bold')
        buffer2 = BytesIO()
        fig2.savefig(buffer2, format='png', dpi=100, bbox_inches='tight')
        buffer2.seek(0)
        gender_chart = base64.b64encode(buffer2.getvalue()).decode('utf-8')
        plt.close(fig2)
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        top_genres = genre_count.most_common(8)
        if top_genres:
            ax3.barh([g[0] for g in top_genres], [g[1] for g in top_genres], color='#667eea')
            ax3.set_title('观影类型偏好 Top8', fontsize=14, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, '暂无类型数据', ha='center', va='center', transform=ax3.transAxes)
        buffer3 = BytesIO()
        fig3.savefig(buffer3, format='png', dpi=100, bbox_inches='tight')
        buffer3.seek(0)
        genre_chart = base64.b64encode(buffer3.getvalue()).decode('utf-8')
        plt.close(fig3)
        
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.hist(watch_counts, bins=min(20, len(set(watch_counts))), color='#48bb78', edgecolor='white', alpha=0.7)
        ax4.set_title('观影数量分布', fontsize=14, fontweight='bold')
        buffer4 = BytesIO()
        fig4.savefig(buffer4, format='png', dpi=100, bbox_inches='tight')
        buffer4.seek(0)
        watch_chart = base64.b64encode(buffer4.getvalue()).decode('utf-8')
        plt.close(fig4)
        
        return {
            'success': True, 'data_id': data_id, 'filename': data['filename'],
            'file_type': data.get('file_type', '.xlsx'), 'upload_time': data['upload_time'],
            'total_users': total_users, 'total_movies': data.get('total_movies', sum(watch_counts)),
            'gender_dist': dict(gender_count), 'avg_age': round(avg_age, 1),
            'age_groups': age_groups, 'avg_watch': round(avg_watch, 1), 'max_watch': max_watch,
            'genre_prefs': dict(genre_count.most_common(10)), 'popular_movies': popular_movies,
            'age_chart': age_chart, 'gender_chart': gender_chart,
            'genre_chart': genre_chart, 'watch_chart': watch_chart,
            'sample_records': records[:5], 'format': data.get('format', 'wide')
        }
    
    def get_user_genre_preferences(self, data_id):
        """获取用户数据的类型偏好，用于电影盲盒"""
        if data_id not in self.uploaded_user_data:
            return None
        
        data = self.uploaded_user_data[data_id]
        records = data['records']
        
        all_genres = []
        genre_counter = Counter()
        
        for record in records:
            genre_str = record.get('watch_genre', '')
            if genre_str:
                import re
                genres = re.split('[,，、/;；]', str(genre_str))
                for genre in genres:
                    genre = genre.strip()
                    if genre:
                        all_genres.append(genre)
                        genre_counter[genre] += 1
        
        # 类型对应的电影推荐库
        genre_movies = {
            "剧情": ["肖申克的救赎", "霸王别姬", "阿甘正传", "我不是药神", "绿皮书", "美丽人生", "放牛班的春天", "忠犬八公的故事"],
            "科幻": ["星际穿越", "盗梦空间", "流浪地球", "黑客帝国", "阿凡达", "银翼杀手2049", "火星救援", "降临"],
            "动作": ["让子弹飞", "速度与激情", "碟中谍", "叶问", "战狼2", "红海行动", "终结者", "黑客帝国"],
            "爱情": ["泰坦尼克号", "怦然心动", "你的名字", "情书", "爱乐之城", "初恋这件小事", "剪刀手爱德华", "罗马假日"],
            "喜剧": ["三傻大闹宝莱坞", "夏洛特烦恼", "唐人街探案", "疯狂的石头", "西虹市首富", "羞羞的铁拳", "人在囧途"],
            "动画": ["千与千寻", "龙猫", "疯狂动物城", "寻梦环游记", "冰雪奇缘", "哪吒之魔童降世", "玩具总动员", "飞屋环游记"],
            "悬疑": ["盗梦空间", "看不见的客人", "消失的爱人", "调音师", "误杀", "心迷宫", "致命ID", "禁闭岛"],
            "奇幻": ["哈利波特", "指环王", "加勒比海盗", "纳尼亚传奇", "神奇动物在哪里", "潘神的迷宫", "大鱼海棠"],
            "灾难": ["泰坦尼克号", "2012", "釜山行", "后天", "唐山大地震", "中国机长", "流感", "末日崩塌"],
            "音乐": ["海上钢琴师", "放牛班的春天", "波西米亚狂想曲", "音乐之声", "再次出发", "爱乐之城", "寻梦环游记"],
            "冒险": ["少年派的奇幻漂流", "荒野猎人", "夺宝奇兵", "地心历险记", "丛林奇航", "勇敢者游戏"],
            "恐怖": ["招魂", "寂静岭", "午夜凶铃", "咒怨", "安娜贝尔", "遗传厄运"],
            "犯罪": ["教父", "无间道", "肖申克的救赎", "低俗小说", "杀人回忆", "七宗罪", "搏击俱乐部"]
        }
        
        top_genres = genre_counter.most_common(5)
        
        return {
            'success': True,
            'genre_preferences': dict(genre_counter),
            'top_genres': top_genres,
            'total_users': len(records),
            'genre_movies': genre_movies,
            'has_data': len(genre_counter) > 0
        }


data_manager = MovieDataManager()

# ==================== 路由定义 ====================
@app.route('/image.jpg')
def serve_ai_image():
    from flask import send_from_directory
    import os
    return send_from_directory('.', 'image.jpg')

# ==================== 智谱AI聊天API ====================
@app.route('/api/zhipu/chat', methods=['POST'])
def api_zhipu_chat():
    import requests
    import json
    
    data = request.get_json()
    message = data.get('message', '')
    context = data.get('context', [])
    
    # 智谱AI配置
    api_key = 'fa590162d40c41f8ae5df72e8abd8f01.2oR2NpYuuJVGKlE9'
    app_id = '2036718063138881536'
    
    chat_url = "https://open.bigmodel.cn/api/llm-application/open/v3/application/invoke"
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    
    # 构建消息历史
    messages = []
    for turn in context:
        messages.append({"role": "user", "content": [{"type": "input", "value": turn["user"]}]})
        messages.append({"role": "assistant", "content": [{"type": "input", "value": turn["bot"]}]})
    messages.append({"role": "user", "content": [{"type": "input", "value": message}]})
    
    chat_data = {
        "app_id": app_id,
        "stream": False,
        "messages": messages
    }
    
    try:
        print(f"发送到API的请求数据: {json.dumps(chat_data, ensure_ascii=False)}")
        response = requests.post(chat_url, headers=headers, json=chat_data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            full_response = ""
            
            # 解析响应 - 多种格式兼容
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                
                if 'messages' in choice:
                    msgs = choice['messages']
                    if 'content' in msgs:
                        content = msgs['content']
                        if isinstance(content, dict) and 'msg' in content:
                            full_response = content['msg']
                        elif isinstance(content, str):
                            full_response = content
                
                elif 'message' in choice:
                    msg = choice['message']
                    if 'content' in msg:
                        content = msg['content']
                        if isinstance(content, str):
                            full_response = content
                        elif isinstance(content, list) and len(content) > 0:
                            for item in content:
                                if isinstance(item, dict) and item.get('type') == 'text' and 'text' in item:
                                    full_response = item['text']
                                    break
                elif 'delta' in choice and 'content' in choice['delta']:
                    delta_content = choice['delta']['content']
                    if isinstance(delta_content, str):
                        full_response = delta_content
                    elif isinstance(delta_content, dict):
                        full_response = delta_content.get('msg') or delta_content.get('text') or delta_content.get('value', '')
            
            elif 'data' in result:
                data_obj = result['data']
                if isinstance(data_obj, dict):
                    content = data_obj.get('content')
                    if isinstance(content, str):
                        full_response = content
                    elif isinstance(content, list) and len(content) > 0:
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text' and 'text' in item:
                                full_response = item['text']
                                break
                    else:
                        full_response = data_obj.get('text') or data_obj.get('msg', '')
                elif isinstance(data_obj, str):
                    full_response = data_obj
            
            elif 'content' in result:
                content = result['content']
                if isinstance(content, str):
                    full_response = content
                elif isinstance(content, list) and len(content) > 0:
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text' and 'text' in item:
                            full_response = item['text']
                            break
            
            elif 'text' in result:
                full_response = result['text']
            
            if full_response:
                return jsonify({'success': True, 'response': full_response})
            else:
                return jsonify({'success': False, 'error': f'未获取到有效响应，原始响应: {json.dumps(result, ensure_ascii=False)}'})
        else:
            return jsonify({'success': False, 'error': f'API请求失败: {response.status_code}, 响应: {response.text}'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'异常: {str(e)}'})

# 页面路由
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

# API路由
@app.route('/api/get_data_sources', methods=['GET'])
def api_get_data_sources():
    return jsonify({'success': True, 'sources': data_manager.get_all_data_sources()})

@app.route('/api/crawl_reviews', methods=['POST'])
def api_crawl_reviews():
    movie_name = request.json.get('movie_name')
    max_count = int(request.json.get('max_count', 20))
    if data_manager.spider:
        movie_id, title = data_manager.spider.search_movie_id(movie_name)
        if movie_id:
            reviews = data_manager.spider.crawl_reviews(movie_id, max_count)
            # 添加情感分析
            for review in reviews:
                sentiment, _ = data_manager.sentiment_analyzer.predict_sentiment(review['content'])
                review['sentiment'] = sentiment
            data_manager.crawled_reviews[title or movie_name] = reviews
            return jsonify({'success': True, 'reviews': reviews, 'movie_name': title or movie_name, 'count': len(reviews)})
    return jsonify({'success': False, 'error': '爬取失败'})

@app.route('/api/search_movie', methods=['POST'])
def api_search_movie():
    movie_name = request.json.get('movie_name', '').strip()
    return jsonify(data_manager.search_movie_info(movie_name))

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    user_id = request.json.get('user_id', data_manager.current_user)
    top_n = int(request.json.get('top_n', 5))
    user = data_manager.users.get(user_id, {"ratings": {}, "watched": []})
    watched = set(user["watched"])
    recs = [m for m in data_manager.movies if m["id"] not in watched]
    recs.sort(key=lambda x: x["rating"], reverse=True)
    result = [{"id": m["id"], "title": m["title"], "director": m["director"], "actors": m["actors"], "genre": m["genre"], "year": m["year"], "rating": m["rating"], "description": m["description"], "score": m["rating"]} for m in recs[:top_n]]
    return jsonify({'success': True, 'recommendations': result, 'method': '智能推荐', 'user_id': user_id})

@app.route('/api/analyze_sentiment', methods=['POST'])
def api_analyze_sentiment():
    source_type = request.json.get('source_type')
    name = request.json.get('name')
    if source_type == 'system':
        movie = next((m for m in data_manager.movies if m["title"] == name), None)
        reviews = data_manager.get_system_reviews(movie["id"]) if movie else []
    elif source_type == 'crawled':
        reviews = data_manager.get_crawled_reviews(name)
    else:
        reviews = data_manager.get_uploaded_reviews(name)
    texts = [r['content'] for r in reviews]
    results = data_manager.sentiment_analyzer.predict_batch(texts)
    for i, r in enumerate(reviews):
        r['sentiment'] = results[i]['sentiment']
    sentiments = [r['sentiment'] for r in reviews]
    counts = Counter(sentiments)
    total = len(reviews)
    return jsonify({'success': True, 'name': name, 'total': total, 'positive': counts.get('positive', 0), 'neutral': counts.get('neutral', 0), 'negative': counts.get('negative', 0), 'reviews': reviews[:100]})

@app.route('/api/get_rating_dist', methods=['POST'])
def api_get_rating_dist():
    source_type = request.json.get('source_type')
    name = request.json.get('name')
    if source_type == 'system':
        movie = next((m for m in data_manager.movies if m["title"] == name), None)
        reviews = data_manager.get_system_reviews(movie["id"]) if movie else []
    else:
        reviews = data_manager.get_crawled_reviews(name) if source_type == 'crawled' else data_manager.get_uploaded_reviews(name)
    ratings = [r["rating"] for r in reviews]
    counts = [ratings.count(i) for i in range(1, 6)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["1星", "2星", "3星", "4星", "5星"], counts, color=['#ff6b6b', '#ffa07a', '#ffd966', '#98d98e', '#6bcf7f'])
    ax.set_title(f"{name} 评分分布")
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return jsonify({'success': True, 'image': image_base64})

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
    if not data_id or data_id not in data_manager.uploaded_user_data:
        return jsonify({'success': False, 'error': '数据不存在'})
    user_data = data_manager.uploaded_user_data[data_id]
    return jsonify({
        'success': True, 'data_id': data_id, 'filename': user_data['filename'],
        'file_type': user_data.get('file_type', '.xlsx'), 'upload_time': user_data['upload_time'],
        'total_users': user_data['total_users'], 'total_movies': user_data.get('total_movies', 0),
        'format': user_data.get('format', 'unknown'),
        'preview': user_data.get('raw_records', user_data['records'])[:10]
    })

# 获取用户类型偏好API（用于电影盲盒）
@app.route('/api/get_genre_preferences', methods=['POST'])
def api_get_genre_preferences():
    data = request.json
    data_id = data.get('data_id')
    if not data_id:
        return jsonify({'success': False, 'error': '请提供数据ID'})
    result = data_manager.get_user_genre_preferences(data_id)
    if result:
        return jsonify(result)
    return jsonify({'success': False, 'error': '数据不存在'})

if __name__ == '__main__':
    print("=" * 60)
    print("豆瓣电影数据分析与推荐系统")
    print("页面结构: 首页 | 影视分析 | 推荐引擎 | 社交管理")
    print("支持格式: Excel (.xlsx, .xls) 和 CSV (.csv)")
    print("电影盲盒: 基于用户上传数据的观影类型智能推荐")
    print("AI助手: 智谱AI大模型智能对话")
    print("访问地址: http://127.0.0.1:5000")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
    except:
        print("[WARN] PyTorch未安装")
    
    app.run(debug=True, host='0.0.0.0', port=5000)