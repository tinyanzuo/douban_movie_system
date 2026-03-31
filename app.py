"""
豆瓣电影数据分析与推荐系统 - Web版本
使用BERT预训练模型进行情感分析
页面结构：首页 | 影视分析 | 推荐引擎 | 社交管理
"""
import torch
from flask import Flask, render_template, request, jsonify, session
from torch import nn, optim
from torch.utils.data import DataLoader
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
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
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# BERT情感分析模块
from bert_sentiment import BERTSentimentAnalyzer

# 爬虫相关库
try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    BeautifulSoup = None

# 词云相关库
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    WordCloud = None

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建Flask应用
app = Flask(__name__)
app.secret_key = 'douban_movie_system_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 海报存储目录
POSTER_DIR = os.path.join(app.static_folder, 'posters')
os.makedirs(POSTER_DIR, exist_ok=True)

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==================== 豆瓣真实风格评论数据（扩充版） ====================
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
        "这部电影改变了我的人生观，强烈推荐每个人都去看！",
        "演技炸裂，剧情紧凑，全程无尿点！",
        "视觉盛宴，特效太震撼了，值得二刷！",
        "笑中带泪，感人至深，今年最好的电影。",
        "立意深刻，发人深省，看完久久不能平静。",
        "配乐太棒了，画面唯美，每一帧都是壁纸。",
        "这部电影完全超出了我的预期，无论是剧情还是制作都无可挑剔。",
        "看完让人热血沸腾，国产电影终于有了自己的超级英雄！",
        "细腻的情感刻画，真实的生活写照，太有共鸣了。",
        "每一帧都可以截图当壁纸，导演的审美太在线了。",
        "演员们的演技集体在线，特别是主角的表演，堪称教科书级别。",
        "这部电影让我重新相信爱情了，太美好了。",
        "笑点密集不尴尬，泪点自然不煽情，难得的佳作。",
        "看完久久不能平静，已经开始二刷了。",
        "这才是我想看到的中国电影，有深度有温度。",
        "特效震撼，剧情烧脑，神作无疑！",
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
        "及格线以上的作品，但谈不上经典。",
        "中规中矩，没有惊喜也没有失望。",
        "还可以吧，适合无聊的时候看看。",
        "有亮点也有不足，整体一般。",
        "不算精彩，但也不至于难看。",
        "普通商业片，没有什么特别出彩的地方。",
        "看完就忘了，没什么记忆点。",
        "还算可以，但没有达到预期。",
        "典型的爆米花电影，看完就忘。",
        "及格分，没有什么特别想说的地方。",
        "不好不坏，属于看过就忘的类型。",
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
        "全程尬演，尴尬到脚趾抠地。",
        "剧情漏洞百出，逻辑混乱。",
        "浪费时间，后悔买票。",
        "特效廉价，演技尴尬，看不下去。",
        "毫无诚意，纯粹圈钱之作。",
        "这是什么垃圾电影，浪费我两个小时的生命！",
        "剧情莫名其妙，完全不知所云。",
        "演员选角失败，完全没有代入感。",
        "特效还不如五年前的电影，太敷衍了。",
        "强行煽情，尴尬到让人想离场。",
        "剧本抄袭痕迹明显，毫无原创性。",
        "导演水平太差，浪费了好题材。",
        "整部电影就像是在凑时长，毫无看点。",
        "看完只觉得被欺骗了，差评！",
        "逻辑硬伤太多，完全经不起推敲。",
    ]
}

USER_NAMES = [
    "豆瓣用户", "影迷小张", "电影爱好者", "观影达人", "评分机器人", "影评人小王",
    "路人甲", "电影发烧友", "爱看电影的猫", "深夜观影者", "周末影迷", "电影控"
]

# ==================== 扩展本地电影数据库（增强版） ====================
LOCAL_MOVIE_CACHE = {
    "肖申克的救赎": {"id": "1292052", "title": "肖申克的救赎", "year": "1994", "rating": 9.7, "director": "弗兰克·德拉邦特", "actors": ["蒂姆·罗宾斯", "摩根·弗里曼"], "genre": "剧情", "genres": ["剧情", "犯罪"], "country": "美国", "language": "英语", "duration": "142分钟", "description": "希望让人自由，一部关于希望与救赎的经典之作。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p480747492.jpg", "release_date": "1994-09-10", "tags": ["经典", "励志", "人性", "自由", "希望"], "wish_count": "120万", "collect_count": "350万", "votes": "280万"},
    "霸王别姬": {"id": "1291546", "title": "霸王别姬", "year": "1993", "rating": 9.6, "director": "陈凯歌", "actors": ["张国荣", "张丰毅", "巩俐"], "genre": "剧情", "genres": ["剧情", "爱情"], "country": "中国", "language": "汉语普通话", "duration": "171分钟", "description": "风华绝代，一曲人生悲歌。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p2561716440.jpg", "release_date": "1993-01-01", "tags": ["经典", "张国荣", "人性", "文革", "爱情"], "wish_count": "85万", "collect_count": "280万", "votes": "220万"},
    "阿甘正传": {"id": "1292720", "title": "阿甘正传", "year": "1994", "rating": 9.5, "director": "罗伯特·泽米吉斯", "actors": ["汤姆·汉克斯"], "genre": "剧情", "genres": ["剧情", "爱情"], "country": "美国", "language": "英语", "duration": "142分钟", "description": "人生就像一盒巧克力，你永远不知道下一块是什么味道。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p511146803.jpg", "release_date": "1994-06-23", "tags": ["励志", "经典", "人生", "美国", "成长"], "wish_count": "100万", "collect_count": "320万", "votes": "250万"},
    "这个杀手不太冷": {"id": "1295644", "title": "这个杀手不太冷", "year": "1994", "rating": 9.4, "director": "吕克·贝松", "actors": ["让·雷诺", "娜塔莉·波特曼"], "genre": "剧情/动作", "genres": ["剧情", "动作", "犯罪"], "country": "法国", "language": "英语", "duration": "110分钟", "description": "杀手与小女孩的温情故事。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p511146978.jpg", "release_date": "1994-09-14", "tags": ["经典", "温情", "杀手", "爱情", "法国"], "wish_count": "95万", "collect_count": "300万", "votes": "240万"},
    "泰坦尼克号": {"id": "1292722", "title": "泰坦尼克号", "year": "1997", "rating": 9.5, "director": "詹姆斯·卡梅隆", "actors": ["莱昂纳多", "凯特·温丝莱特"], "genre": "爱情/灾难", "genres": ["剧情", "爱情", "灾难"], "country": "美国", "language": "英语", "duration": "194分钟", "description": "永恒的爱情，永不沉没的经典。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p457760035.jpg", "release_date": "1997-12-19", "tags": ["爱情", "经典", "灾难", "美国", "浪漫"], "wish_count": "110万", "collect_count": "340万", "votes": "270万"},
    "盗梦空间": {"id": "3541415", "title": "盗梦空间", "year": "2010", "rating": 9.4, "director": "克里斯托弗·诺兰", "actors": ["莱昂纳多"], "genre": "科幻/悬疑", "genres": ["科幻", "悬疑", "惊悚"], "country": "美国", "language": "英语", "duration": "148分钟", "description": "梦境与现实，一场关于意识的大胆探索。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p513344864.jpg", "release_date": "2010-09-01", "tags": ["科幻", "诺兰", "烧脑", "梦境", "悬疑"], "wish_count": "105万", "collect_count": "330万", "votes": "260万"},
    "楚门的世界": {"id": "1292064", "title": "楚门的世界", "year": "1998", "rating": 9.3, "director": "彼得·威尔", "actors": ["金·凯瑞"], "genre": "剧情/科幻", "genres": ["剧情", "科幻"], "country": "美国", "language": "英语", "duration": "103分钟", "description": "真实与虚假的边界在哪里？", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p479682872.jpg", "release_date": "1998-06-05", "tags": ["经典", "人性", "自由", "金·凯瑞", "人生"], "wish_count": "80万", "collect_count": "260万", "votes": "210万"},
    "千与千寻": {"id": "1291561", "title": "千与千寻", "year": "2001", "rating": 9.4, "director": "宫崎骏", "actors": ["柊瑠美"], "genre": "动画/奇幻", "genres": ["动画", "奇幻", "冒险"], "country": "日本", "language": "日语", "duration": "125分钟", "description": "成长的旅程，宫崎骏的巅峰之作。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p2557573348.jpg", "release_date": "2001-07-20", "tags": ["动画", "宫崎骏", "成长", "日本", "奇幻"], "wish_count": "130万", "collect_count": "380万", "votes": "290万"},
    "星际穿越": {"id": "1889243", "title": "星际穿越", "year": "2014", "rating": 9.4, "director": "克里斯托弗·诺兰", "actors": ["马修·麦康纳"], "genre": "科幻", "genres": ["科幻", "冒险", "剧情"], "country": "美国", "language": "英语", "duration": "169分钟", "description": "穿越时空的爱，人类命运的终极探索。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p2206088801.jpg", "release_date": "2014-11-07", "tags": ["科幻", "诺兰", "太空", "亲情", "时空"], "wish_count": "100万", "collect_count": "310万", "votes": "250万"},
    "海上钢琴师": {"id": "1292001", "title": "海上钢琴师", "year": "1998", "rating": 9.3, "director": "朱塞佩·托纳多雷", "actors": ["蒂姆·罗斯"], "genre": "剧情/音乐", "genres": ["剧情", "音乐"], "country": "意大利", "language": "英语", "duration": "125分钟", "description": "1900的故事，音乐与自由的传奇。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p2579176634.jpg", "release_date": "1998-10-28", "tags": ["经典", "音乐", "人生", "意大利", "孤独"], "wish_count": "75万", "collect_count": "240万", "votes": "200万"},
    "让子弹飞": {"id": "3742360", "title": "让子弹飞", "year": "2010", "rating": 9.0, "director": "姜文", "actors": ["姜文", "葛优", "周润发"], "genre": "剧情/喜剧", "genres": ["剧情", "喜剧", "动作"], "country": "中国", "language": "汉语普通话", "duration": "132分钟", "description": "站着把钱挣了！", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p511147276.jpg", "release_date": "2010-12-16", "tags": ["黑色幽默", "姜文", "经典", "中国", "政治"], "wish_count": "90万", "collect_count": "290万", "votes": "230万"},
    "流浪地球": {"id": "26266893", "title": "流浪地球", "year": "2019", "rating": 7.9, "director": "郭帆", "actors": ["吴京", "屈楚萧", "李光洁"], "genre": "科幻", "genres": ["科幻", "灾难", "冒险"], "country": "中国", "language": "汉语普通话", "duration": "125分钟", "description": "带着地球去流浪，中国科幻的里程碑。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p2541234097.jpg", "release_date": "2019-02-05", "tags": ["科幻", "中国科幻", "刘慈欣", "灾难", "吴京"], "wish_count": "125万", "collect_count": "350万", "votes": "290万"},
    "哪吒之魔童降世": {"id": "26794435", "title": "哪吒之魔童降世", "year": "2019", "rating": 8.4, "director": "饺子", "actors": ["吕艳婷", "囧森瑟夫"], "genre": "动画/奇幻", "genres": ["动画", "奇幻", "喜剧"], "country": "中国", "language": "汉语普通话", "duration": "110分钟", "description": "我命由我不由天！", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p2564668633.jpg", "release_date": "2019-07-26", "tags": ["动画", "国漫", "中国", "哪吒", "成长"], "wish_count": "115万", "collect_count": "320万", "votes": "270万"},
    "我不是药神": {"id": "26752088", "title": "我不是药神", "year": "2018", "rating": 9.0, "director": "文牧野", "actors": ["徐峥", "王传君", "周一围"], "genre": "剧情", "genres": ["剧情", "喜剧"], "country": "中国", "language": "汉语普通话", "duration": "117分钟", "description": "现实题材的震撼之作。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p2519064487.jpg", "release_date": "2018-07-05", "tags": ["现实", "徐峥", "中国", "社会", "感动"], "wish_count": "140万", "collect_count": "370万", "votes": "300万"},
    "绿皮书": {"id": "27060077", "title": "绿皮书", "year": "2018", "rating": 8.9, "director": "彼得·法拉利", "actors": ["维果·莫腾森", "马赫沙拉·阿里"], "genre": "剧情/喜剧", "genres": ["剧情", "喜剧", "传记"], "country": "美国", "language": "英语", "duration": "130分钟", "description": "跨越种族与阶层的友谊。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p2544438975.jpg", "release_date": "2018-09-11", "tags": ["温情", "种族", "美国", "友情", "公路"], "wish_count": "95万", "collect_count": "300万", "votes": "240万"},
    "三傻大闹宝莱坞": {"id": "3793023", "title": "三傻大闹宝莱坞", "year": "2009", "rating": 9.2, "director": "拉吉库马尔·希拉尼", "actors": ["阿米尔·汗", "马德哈万"], "genre": "剧情/喜剧", "genres": ["剧情", "喜剧", "爱情"], "country": "印度", "language": "印地语", "duration": "171分钟", "description": "追求卓越，成功会不请自来。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p579729424.jpg", "release_date": "2009-12-23", "tags": ["励志", "印度", "教育", "喜剧", "阿米尔·汗"], "wish_count": "110万", "collect_count": "320万", "votes": "260万"},
    "放牛班的春天": {"id": "1291548", "title": "放牛班的春天", "year": "2004", "rating": 9.3, "director": "克里斯托夫·巴拉蒂", "actors": ["热拉尔·朱尼奥", "让-巴蒂斯特·莫尼耶"], "genre": "剧情/音乐", "genres": ["剧情", "音乐"], "country": "法国", "language": "法语", "duration": "97分钟", "description": "音乐治愈心灵。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p2561717895.jpg", "release_date": "2004-03-17", "tags": ["音乐", "教育", "法国", "温情", "成长"], "wish_count": "80万", "collect_count": "260万", "votes": "210万"},
    "忠犬八公的故事": {"id": "3011091", "title": "忠犬八公的故事", "year": "2009", "rating": 9.4, "director": "莱塞·霍尔斯道姆", "actors": ["理查·基尔", "琼·艾伦"], "genre": "剧情", "genres": ["剧情"], "country": "美国", "language": "英语", "duration": "93分钟", "description": "等待是最长情的告白。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p524964016.jpg", "release_date": "2009-06-13", "tags": ["温情", "狗狗", "感动", "美国", "忠诚"], "wish_count": "105万", "collect_count": "330万", "votes": "260万"},
    "美丽人生": {"id": "1292063", "title": "美丽人生", "year": "1997", "rating": 9.5, "director": "罗伯托·贝尼尼", "actors": ["罗伯托·贝尼尼", "尼可莱塔·布拉斯基"], "genre": "剧情/喜剧", "genres": ["剧情", "喜剧", "战争"], "country": "意大利", "language": "意大利语", "duration": "116分钟", "description": "即使在黑暗中，也要相信美好。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p2578471708.jpg", "release_date": "1997-12-20", "tags": ["经典", "温情", "二战", "父亲", "意大利"], "wish_count": "90万", "collect_count": "290万", "votes": "230万"},
    "怦然心动": {"id": "3319755", "title": "怦然心动", "year": "2010", "rating": 9.1, "director": "罗伯·莱纳", "actors": ["玛德琳·卡罗尔", "卡兰·麦克奥利菲"], "genre": "爱情/喜剧", "genres": ["剧情", "喜剧", "爱情"], "country": "美国", "language": "英语", "duration": "90分钟", "description": "初恋的美好，成长的喜悦。", "poster": "https://img9.doubanio.com/view/photo/s_ratio_poster/public/p567590933.jpg", "release_date": "2010-07-26", "tags": ["青春", "爱情", "成长", "美国", "初恋"], "wish_count": "120万", "collect_count": "360万", "votes": "280万"},
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


# ==================== BERT情感分析模型已在bert_sentiment.py中实现 ====================

# ==================== 豆瓣爬虫类（增强版 - 支持获取电影详情和海报下载） ====================
class DoubanSpider:
    """豆瓣电影评论爬虫 - 增强版"""

    def __init__(self):
        self.headers = {
            'Cookie': 'll="118254"; bid=BwcycY_GcAA; _pk_id.100001.4cf6=3a9e7e0cc60384ca.1774349925.; _vwo_uuid_v2=DE05BC0B267794C027A0B55F121E447FB|67213e974038f3093b609c7cfb2d4998; __yadk_uid=WlOmJpAAxxz8cTGpBqTGzHf1dBFSvk6k; dbcl2="293423119:3LWQOX7tv8w"; push_noty_num=0; push_doumail_num=0; __utmv=30149280.29342; ck=Fula; __utmc=30149280; __utmc=223695111; frodotk_db="fa15ea4797f8f4a9ee41bed6c7eaddfb"; _ga=GA1.2.462099050.1774833762; _gid=GA1.2.600847403.1774833763; _ga_Y4GN1R87RG=GS2.1.s1774833762$o1$g0$t1774833771$j51$l0$h0; __utmz=30149280.1774840531.8.6.utmcsr=search.douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/movie/subject_search; __utmz=223695111.1774840531.8.6.utmcsr=search.douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/movie/subject_search; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1774865168%2C%22https%3A%2F%2Fsearch.douban.com%2Fmovie%2Fsubject_search%3Fsearch_text%3D%E8%AE%A9%E5%AD%90%E5%BC%B9%E9%A3%9E%26cat%3D1002%22%5D; _pk_ses.100001.4cf6=1; ap_v=0,6.0; __utma=30149280.1808011285.1774349897.1774857365.1774865170.11; __utmb=30149280.0.10.1774865170; __utma=223695111.406837352.1774349924.1774857365.1774865170.11; __utmb=223695111.0.10.1774865170',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'movie.douban.com',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Connection': 'keep-alive'
        }

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
        """获取电影详细信息（包括海报、标签等）"""
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

            poster_elem = soup.find('img', {'rel': 'v:image'})
            poster_url = poster_elem['src'] if poster_elem and poster_elem.get('src') else ""
            info['poster'] = poster_url
            info['local_poster'] = self.download_poster(poster_url, info.get('title', movie_id), movie_id)

            info_elem = soup.find('div', id='info')
            if info_elem:
                info_text = info_elem.text
                country_match = re.search(r'制片国家/地区:\s*(.+?)(?:\n|$)', info_text)
                info['country'] = country_match.group(1).strip() if country_match else "未知"
                language_match = re.search(r'语言:\s*(.+?)(?:\n|$)', info_text)
                info['language'] = language_match.group(1).strip() if language_match else "未知"
                duration_match = re.search(r'片长:\s*(.+?)(?:\n|$)', info_text)
                info['duration'] = duration_match.group(1).strip() if duration_match else "未知"
                year_match = re.search(r'(\d{4})', info_text)
                info['year'] = year_match.group(1) if year_match else "未知"
                release_date_match = re.search(r'上映日期:\s*(.+?)(?:\n|$)', info_text)
                info['release_date'] = release_date_match.group(1).strip() if release_date_match else "未知"
            else:
                info['country'] = "未知"
                info['language'] = "未知"
                info['duration'] = "未知"
                info['year'] = "未知"
                info['release_date'] = "未知"

            rating_elem = soup.find('strong', property='v:average')
            info['rating'] = rating_elem.text.strip() if rating_elem else "暂无"

            votes_elem = soup.find('span', property='v:votes')
            info['votes'] = votes_elem.text.strip() if votes_elem else "0"

            summary_elem = soup.find('span', property='v:summary')
            info['summary'] = summary_elem.text.strip() if summary_elem else "暂无简介"

            tags_elem = soup.find('div', class_='tags-body')
            if tags_elem:
                tag_elems = tags_elem.find_all('a')
                info['tags'] = [tag.text.strip() for tag in tag_elems[:8]] if tag_elems else []
            else:
                info['tags'] = []

            wish_elem = soup.find('a', href=lambda x: x and 'wish' in x)
            info['wish_count'] = wish_elem.text.strip() if wish_elem else "0"

            collect_elem = soup.find('a', href=lambda x: x and 'collect' in x)
            info['collect_count'] = collect_elem.text.strip() if collect_elem else "0"

            return info

        except Exception as e:
            print(f"获取电影详情失败: {e}")
            return None

    def crawl_reviews(self, movie_id, max_count=20, progress_callback=None):
        """爬取电影评论"""
        reviews = []
        max_page = min((max_count + 19) // 20, 20)

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

    def download_poster(self, url, movie_title, movie_id):
        if not url:
            return None
        import hashlib
        safe_title = re.sub(r'[\\/*?:"<>|]', '_', movie_title)
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        ext = url.split('.')[-1].split('?')[0]
        if ext.lower() not in ['jpg', 'jpeg', 'png', 'webp']:
            ext = 'jpg'
        filename = f"{safe_title}_{url_hash}.{ext}"
        local_path = os.path.join(POSTER_DIR, filename)

        # 如果本地已存在且不为空，直接返回
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return f"static/posters/{filename}"

        # 构建请求头，注意 Host 要与图片域名一致，Referer 使用电影详情页
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        host = parsed_url.netloc

        headers = {
            'User-Agent': self.headers.get('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
            'Referer': f'https://movie.douban.com/subject/{movie_id}/',
            'Origin': 'https://movie.douban.com',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Connection': 'keep-alive',
            'Host': host,  # 关键：设置为图片服务器的域名
            'Cookie': self.headers.get('Cookie', '')
        }

        # 重试 3 次
        for attempt in range(3):
            try:
                # 允许重定向，但只接受最终是图片的响应
                resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
                content_type = resp.headers.get('Content-Type', '').lower()
                # 如果返回的是图片，保存并返回
                if resp.status_code == 200 and ('image' in content_type or 'application/octet-stream' in content_type):
                    with open(local_path, 'wb') as f:
                        f.write(resp.content)
                    print(f"[DEBUG] 海报下载成功: {local_path}")
                    return f"static/posters/{filename}"
                else:
                    # 打印详细信息用于调试
                    print(f"[DEBUG] 海报下载失败 (尝试 {attempt+1}/3): {url}")
                    print(f"  状态码: {resp.status_code}, Content-Type: {content_type}")
                    if 'html' in content_type and len(resp.content) < 1024:
                        # 如果返回的是短 HTML，可能是反爬页面
                        print(f"  可能原因: 反爬或Cookie失效")
                        # 可选：保存HTML内容用于分析
                        with open(local_path + '.html', 'wb') as f:
                            f.write(resp.content)
            except Exception as e:
                print(f"[DEBUG] 海报下载异常 (尝试 {attempt+1}/3): {e}")

            # 等待后重试
            time.sleep(random.uniform(1, 2))

        return None

# ==================== 深度学习推荐引擎 ====================

class NeuralCollaborativeFiltering(nn.Module):
    """神经协同过滤模型 - 用于排序层"""

    def __init__(self, num_users, num_movies, embedding_dim=64, hidden_dims=[128, 64]):
        super(NeuralCollaborativeFiltering, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # GMF部分
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # MLP部分
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_movie_embedding = nn.Embedding(num_movies, embedding_dim)

        mlp_layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        self.mlp_layers = nn.Sequential(*mlp_layers)
        self.final_layer = nn.Linear(embedding_dim + hidden_dims[-1], 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, user_ids, movie_ids):
        gmf_user_vec = self.gmf_user_embedding(user_ids)
        gmf_movie_vec = self.gmf_movie_embedding(movie_ids)
        gmf_output = gmf_user_vec * gmf_movie_vec

        mlp_user_vec = self.mlp_user_embedding(user_ids)
        mlp_movie_vec = self.mlp_movie_embedding(movie_ids)
        mlp_input = torch.cat([mlp_user_vec, mlp_movie_vec], dim=1)
        mlp_output = self.mlp_layers(mlp_input)

        concat = torch.cat([gmf_output, mlp_output], dim=1)
        prediction = self.final_layer(concat)
        prediction = torch.sigmoid(prediction) * 4 + 1

        return prediction.squeeze()


class DeepRecommendationEngine:
    """深度学习推荐引擎 - 召回层 + 排序层"""

    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.model = None
        self.user2id = {}
        self.movie2id = {}
        self.id2user = {}
        self.id2movie = {}
        self.num_users = 0
        self.num_movies = 0
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_losses = []

        # 召回层：用户/物品Embedding向量库
        self.user_embeddings = {}
        self.movie_embeddings = {}

        # 数据管理器引用
        self.data_manager = None

    def set_data_manager(self, dm):
        """设置数据管理器引用"""
        self.data_manager = dm

    def build_vocab(self, users, movies):
        """构建用户和电影的ID映射"""
        self.user2id = {user: idx for idx, user in enumerate(users)}
        self.movie2id = {movie: idx for idx, movie in enumerate(movies)}
        self.id2user = {idx: user for user, idx in self.user2id.items()}
        self.id2movie = {idx: movie for movie, idx in self.movie2id.items()}
        self.num_users = len(users)
        self.num_movies = len(movies)
        print(f"📊 用户数量: {self.num_users}, 电影数量: {self.num_movies}")

    def build_model(self):
        self.model = NeuralCollaborativeFiltering(
            num_users=self.num_users,
            num_movies=self.num_movies,
            embedding_dim=self.embedding_dim
        ).to(self.device)
        return self.model

    def train(self, ratings_data, epochs=30, batch_size=16, lr=0.001):
        """训练推荐模型"""
        if len(ratings_data) < 10:
            print(f"⚠️ 训练数据不足({len(ratings_data)}条)，跳过训练")
            return False

        if not self.user2id:
            users = list(set([r['user_id'] for r in ratings_data]))
            movies = list(set([r['movie_id'] for r in ratings_data]))
            self.build_vocab(users, movies)

        self.build_model()

        X_users = torch.tensor([self.user2id[r['user_id']] for r in ratings_data], dtype=torch.long)
        X_movies = torch.tensor([self.movie2id[r['movie_id']] for r in ratings_data], dtype=torch.long)
        y_ratings = torch.tensor([r['rating'] for r in ratings_data], dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_users, X_movies, y_ratings)
        actual_batch_size = min(batch_size, len(dataset))

        loader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        print(f"🚀 开始训练深度学习推荐模型...")
        print(f"   训练样本: {len(dataset)}, Batch Size: {actual_batch_size}")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0

            for batch_users, batch_movies, batch_ratings in loader:
                if batch_users.size(0) < 1:
                    continue

                batch_users = batch_users.to(self.device)
                batch_movies = batch_movies.to(self.device)
                batch_ratings = batch_ratings.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_users, batch_movies)
                loss = criterion(predictions, batch_ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1

            if batch_count > 0:
                avg_loss = total_loss / batch_count
                self.train_losses.append(avg_loss)

                if (epoch + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        self._extract_embeddings()
        self.is_trained = True
        print("✅ 深度学习推荐模型训练完成!")
        return True

    def _extract_embeddings(self):
        """提取用户和电影的Embedding向量"""
        if self.model is None:
            return

        self.model.eval()
        self.user_embeddings = {}
        self.movie_embeddings = {}

        with torch.no_grad():
            for user_id, user_idx in self.user2id.items():
                user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                embedding = self.model.user_embedding(user_tensor).cpu().numpy()
                self.user_embeddings[user_id] = embedding[0]

            for movie_id, movie_idx in self.movie2id.items():
                movie_tensor = torch.tensor([movie_idx], dtype=torch.long).to(self.device)
                embedding = self.model.movie_embedding(movie_tensor).cpu().numpy()
                self.movie_embeddings[movie_id] = embedding[0]

    def get_recommendation_for_uploaded_data(self, data_id, all_movies, top_n=10):
        """为上传的数据文件生成推荐"""
        if self.data_manager is None:
            return []

        if data_id not in self.data_manager.uploaded_user_data:
            return []

        user_data = self.data_manager.uploaded_user_data[data_id]
        records = user_data.get('records', [])

        if not records:
            return []

        watched_movies = set()
        genre_weights = Counter()

        for record in records:
            watch_content = record.get('watch_content', '')
            if watch_content:
                movies = re.split('[,，、/;；]', watch_content)
                for movie in movies:
                    movie = movie.strip()
                    if movie:
                        watched_movies.add(movie)

            top_genres = record.get('top_genres', {})
            if isinstance(top_genres, dict):
                for genre, weight in top_genres.items():
                    genre_weights[genre] += weight

        user_pref_vector = {}
        total_weight = sum(genre_weights.values())
        if total_weight > 0:
            for genre, weight in genre_weights.items():
                user_pref_vector[genre] = weight / total_weight

        recommendations = []
        for movie in all_movies:
            movie_title = movie['title']

            if movie_title in watched_movies:
                continue

            movie_genres = movie.get('genre', '').split('/')
            genre_score = 0
            for genre in movie_genres:
                genre = genre.strip()
                if genre in user_pref_vector:
                    genre_score += user_pref_vector[genre]

            base_score = movie.get('rating', 8.0) / 10
            final_score = base_score * 0.5 + genre_score * 0.5

            recommendations.append({
                **movie,
                'predicted_score': round(final_score * 5, 2),
                'recommendation_score': round(final_score * 5, 2),
                'match_reason': f"类型匹配度: {round(genre_score * 100)}%"
            })

        recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
        return recommendations[:top_n]

    def recall_by_embedding(self, user_id, all_movies, top_k=200):
        """召回层：基于用户Embedding向量进行向量召回"""
        if not self.is_trained or user_id not in self.user_embeddings:
            return all_movies[:top_k] if len(all_movies) > top_k else all_movies

        user_emb = self.user_embeddings[user_id]
        scores = []

        for movie in all_movies:
            movie_id = movie['id']
            if movie_id in self.movie_embeddings:
                movie_emb = self.movie_embeddings[movie_id]
                similarity = np.dot(user_emb, movie_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(movie_emb) + 1e-8)
                scores.append((movie, similarity))
            else:
                scores.append((movie, 0.5))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [movie for movie, _ in scores[:top_k]]

    def rank_by_model(self, user_id, candidate_movies):
        """排序层：用深度模型对候选集进行精排"""
        if not self.is_trained or user_id not in self.user2id:
            return [(movie, movie.get('rating', 8.0) / 2.0) for movie in candidate_movies]

        user_idx = self.user2id[user_id]
        user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)

        scores = []
        self.model.eval()
        with torch.no_grad():
            for movie in candidate_movies:
                movie_id = movie['id']
                if movie_id in self.movie2id:
                    movie_idx = self.movie2id[movie_id]
                    movie_tensor = torch.tensor([movie_idx], dtype=torch.long).to(self.device)
                    rating = self.model(user_tensor, movie_tensor).item()
                else:
                    rating = movie.get('rating', 8.0) / 2.0
                scores.append((movie, rating))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def predict_rating(self, user_id, movie_id):
        """预测用户对电影的评分"""
        if not self.is_trained:
            return 3.0

        if user_id not in self.user2id or movie_id not in self.movie2id:
            return 3.0

        user_idx = torch.tensor([self.user2id[user_id]], dtype=torch.long).to(self.device)
        movie_idx = torch.tensor([self.movie2id[movie_id]], dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            rating = self.model(user_idx, movie_idx).item()

        return round(rating, 2)

    def find_similar_movies(self, movie_id, top_n=10):
        """基于Embedding找到相似电影"""
        if not self.is_trained or movie_id not in self.movie_embeddings:
            return []

        target_emb = self.movie_embeddings[movie_id]
        similarities = []

        for other_id, other_emb in self.movie_embeddings.items():
            if other_id == movie_id:
                continue
            similarity = np.dot(target_emb, other_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(other_emb) + 1e-8)
            similarities.append((other_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

    def find_similar_users(self, user_id, top_n=10):
        """基于Embedding找到相似用户"""
        if not self.is_trained or user_id not in self.user_embeddings:
            return []

        target_emb = self.user_embeddings[user_id]
        similarities = []

        for other_id, other_emb in self.user_embeddings.items():
            if other_id == user_id:
                continue
            similarity = np.dot(target_emb, other_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(other_emb) + 1e-8)
            similarities.append((other_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

    def save_model(self, filepath='deep_recommend_model.pth'):
        """保存模型"""
        if self.model is None:
            return
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'user2id': self.user2id,
            'movie2id': self.movie2id,
            'id2user': self.id2user,
            'id2movie': self.id2movie,
            'num_users': self.num_users,
            'num_movies': self.num_movies,
            'embedding_dim': self.embedding_dim,
            'train_losses': self.train_losses,
            'user_embeddings': self.user_embeddings,
            'movie_embeddings': self.movie_embeddings
        }
        torch.save(checkpoint, filepath)
        print(f"💾 模型已保存到: {filepath}")

    def load_model(self, filepath='deep_recommend_model.pth'):
        """加载模型"""
        if not os.path.exists(filepath):
            return False

        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

        self.user2id = checkpoint['user2id']
        self.movie2id = checkpoint['movie2id']
        self.id2user = checkpoint.get('id2user', {})
        self.id2movie = checkpoint.get('id2movie', {})
        self.num_users = checkpoint['num_users']
        self.num_movies = checkpoint['num_movies']
        self.embedding_dim = checkpoint['embedding_dim']
        self.train_losses = checkpoint.get('train_losses', [])
        self.user_embeddings = checkpoint.get('user_embeddings', {})
        self.movie_embeddings = checkpoint.get('movie_embeddings', {})

        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        print(f"✅ 模型已加载: {filepath}")
        return True

    def get_recommendation(self, user_id, all_movies, top_n=10, exclude_watched=True, watched_list=None):
        """完整推荐流程：召回层 + 排序层"""
        watched_set = set(watched_list) if watched_list else set()

        candidate_movies = [m for m in all_movies if not exclude_watched or m['id'] not in watched_set]

        if not candidate_movies:
            return []

        recall_count = min(200, len(candidate_movies))
        recalled_movies = self.recall_by_embedding(user_id, candidate_movies, recall_count)
        ranked_results = self.rank_by_model(user_id, recalled_movies)

        recommendations = []
        for movie, score in ranked_results[:top_n]:
            recommendations.append({
                **movie,
                'predicted_score': round(score, 2),
                'recommendation_score': round(score, 2)
            })

        return recommendations


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

        # 使用BERT情感分析器
        print("\n" + "="*50)
        print("正在初始化BERT情感分析器...")
        print("="*50)
        self.sentiment_analyzer = BERTSentimentAnalyzer(model_name='bert-base-chinese')

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
            print(f"📊 准备训练数据: {len(ratings_data)} 条评分记录")
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
        """上传并解析用户数据文件"""
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
        col_map = {'nickname': None, 'gender': None, 'age': None, 'movie_name': None}

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
                'nickname': nickname, 'gender': gender, 'age': age, 'movie_name': movie_name
            }
            movie_records.append(movie_record)

            if nickname not in user_summary:
                user_summary[nickname] = {'nickname': nickname, 'gender': gender, 'age': age, 'movies': [], 'watch_count': 0}

            user_summary[nickname]['movies'].append(movie_name)
            user_summary[nickname]['watch_count'] += 1

        if not user_summary:
            raise Exception("没有找到有效的用户数据")

        analysis_records = []
        for nickname, summary in user_summary.items():
            analysis_records.append({
                'nickname': nickname, 'gender': summary['gender'], 'age': summary['age'],
                'watch_count': summary['watch_count'], 'watch_content': ','.join(summary['movies'])
            })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_id = f"user_data_{timestamp}"
        self.uploaded_user_data[data_id] = {
            'filename': filename, 'file_type': file_ext, 'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'records': analysis_records, 'raw_records': movie_records,
            'total_users': len(analysis_records), 'total_movies': len(movie_records), 'format': 'long'
        }
        return {'success': True, 'data_id': data_id, 'total_users': len(analysis_records), 'total_movies': len(movie_records), 'preview': movie_records[:10]}

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

        age_groups = {'18岁以下': 0, '18-25岁': 0, '26-35岁': 0, '36-50岁': 0, '50岁以上': 0}
        for r in records:
            age = r.get('age', 0)
            if age <= 0: continue
            elif age < 18: age_groups['18岁以下'] += 1
            elif age <= 25: age_groups['18-25岁'] += 1
            elif age <= 35: age_groups['26-35岁'] += 1
            elif age <= 50: age_groups['36-50岁'] += 1
            else: age_groups['50岁以上'] += 1

        movie_titles = []
        for r in records:
            content = r.get('watch_content', '')
            if content:
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

        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.hist(watch_counts, bins=min(20, len(set(watch_counts))), color='#48bb78', edgecolor='white', alpha=0.7)
        ax4.set_title('观影数量分布', fontsize=14, fontweight='bold')
        buffer4 = BytesIO()
        fig4.savefig(buffer4, format='png', dpi=100, bbox_inches='tight')
        buffer4.seek(0)
        watch_chart = base64.b64encode(buffer4.getvalue()).decode('utf-8')
        plt.close(fig4)

        return {
            'success': True, 'data_id': data_id, 'filename': data['filename'], 'file_type': data.get('file_type', '.xlsx'),
            'upload_time': data['upload_time'], 'total_users': total_users, 'total_movies': data.get('total_movies', sum(watch_counts)),
            'gender_dist': dict(gender_count), 'avg_age': round(avg_age, 1), 'avg_watch': round(avg_watch, 1), 'max_watch': max_watch,
            'popular_movies': popular_movies, 'age_chart': age_chart, 'gender_chart': gender_chart,
            'watch_chart': watch_chart, 'sample_records': records[:5], 'format': data.get('format', 'wide')
        }

    def get_user_genre_preferences(self, data_id):
        if data_id not in self.uploaded_user_data:
            return None
        data = self.uploaded_user_data[data_id]
        records = data['records']
        genre_counter = Counter()

        genre_movies = {
            "剧情": ["肖申克的救赎", "霸王别姬", "阿甘正传", "我不是药神", "绿皮书", "美丽人生", "放牛班的春天"],
            "科幻": ["星际穿越", "盗梦空间", "流浪地球", "黑客帝国", "阿凡达", "银翼杀手2049"],
            "动作": ["让子弹飞", "速度与激情", "碟中谍", "叶问", "战狼2", "红海行动"],
            "爱情": ["泰坦尼克号", "怦然心动", "你的名字", "情书", "爱乐之城", "初恋这件小事"],
            "喜剧": ["三傻大闹宝莱坞", "夏洛特烦恼", "唐人街探案", "疯狂的石头", "西虹市首富"],
            "动画": ["千与千寻", "龙猫", "疯狂动物城", "寻梦环游记", "冰雪奇缘", "哪吒之魔童降世"],
            "悬疑": ["盗梦空间", "看不见的客人", "消失的爱人", "调音师", "误杀", "心迷宫"]
        }

        for record in records:
            watch_content = record.get('watch_content', '')
            if watch_content:
                for genre, movies in genre_movies.items():
                    for movie in movies:
                        if movie.lower() in watch_content.lower():
                            genre_counter[genre] += 1
                            break

        return {'success': True, 'genre_preferences': dict(genre_counter), 'top_genres': genre_counter.most_common(5),
                'total_users': len(records), 'genre_movies': genre_movies, 'has_data': len(genre_counter) > 0}


data_manager = MovieDataManager()

# ==================== 路由定义 ====================
@app.route('/image.jpg')
def serve_ai_image():
    from flask import send_from_directory
    return send_from_directory('.', 'image.jpg')

# ==================== 智谱AI聊天API ====================
@app.route('/api/zhipu/chat', methods=['POST'])
def api_zhipu_chat():
    import requests
    import json

    data = request.get_json()
    message = data.get('message', '')
    context = data.get('context', [])

    api_key = 'fa590162d40c41f8ae5df72e8abd8f01.2oR2NpYuuJVGKlE9'
    app_id = '2036718063138881536'

    chat_url = "https://open.bigmodel.cn/api/llm-application/open/v3/application/invoke"
    headers = {"Authorization": api_key, "Content-Type": "application/json"}

    messages = []
    for turn in context:
        messages.append({"role": "user", "content": [{"type": "input", "value": turn["user"]}]})
        messages.append({"role": "assistant", "content": [{"type": "input", "value": turn["bot"]}]})
    messages.append({"role": "user", "content": [{"type": "input", "value": message}]})

    chat_data = {"app_id": app_id, "stream": False, "messages": messages}

    try:
        response = requests.post(chat_url, headers=headers, json=chat_data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            full_response = ""
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    content = choice['message']['content']
                    if isinstance(content, str):
                        full_response = content
                    elif isinstance(content, list) and len(content) > 0:
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                full_response = item.get('text', '')
                                break
            if full_response:
                return jsonify({'success': True, 'response': full_response})
            else:
                return jsonify({'success': False, 'error': '未获取到有效响应'})
        else:
            return jsonify({'success': False, 'error': f'API请求失败: {response.status_code}'})
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

# ==================== API路由 ====================
@app.route('/api/get_data_sources', methods=['GET'])
def api_get_data_sources():
    return jsonify({'success': True, 'sources': data_manager.get_all_data_sources()})

@app.route('/api/crawl_reviews', methods=['POST'])
def api_crawl_reviews():
    movie_name = request.json.get('movie_name')
    max_count = int(request.json.get('max_count', 20))
    max_count = min(max_count, 1000)

    if data_manager.spider:
        movie_id, title = data_manager.spider.search_movie_id(movie_name)
        if movie_id:
            def progress_callback(current, total):
                print(f"爬取进度: {current}/{total}")

            reviews = data_manager.spider.crawl_reviews(movie_id, max_count, progress_callback)
            # 使用BERT进行情感分析（用于统计分析，但不显示在预览栏）
            texts = [r['content'] for r in reviews]
            results = data_manager.sentiment_analyzer.predict_batch(texts)
            for i, review in enumerate(reviews):
                review['sentiment'] = results[i]['sentiment']
                review['confidence'] = results[i]['confidence']
            data_manager.crawled_reviews[title or movie_name] = reviews

            # 返回时过滤掉情感相关字段，只保留用户、评分、评论内容
            preview_reviews = []
            for review in reviews:
                preview_reviews.append({
                    'user': review.get('user', '豆瓣用户'),
                    'rating': review.get('rating', 3),
                    'content': review.get('content', ''),
                    'time': review.get('time', '')
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
    """上传评论文件"""
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
        if file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
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
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row and len(row) > 0:
                        reviews.append({
                            'content': row[0] if len(row) > 0 else '',
                            'rating': int(row[1]) if len(row) > 1 and row[1].isdigit() else 3,
                            'user': row[2] if len(row) > 2 else '上传用户',
                            'time': row[3] if len(row) > 3 else datetime.now().strftime('%Y-%m-%d')
                        })
        elif file_ext == '.json':
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
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

        # 使用BERT进行情感分析（用于统计分析）
        if reviews:
            texts = [r['content'] for r in reviews]
            results = data_manager.sentiment_analyzer.predict_batch(texts)
            for i, review in enumerate(reviews):
                review['sentiment'] = results[i]['sentiment']
                review['confidence'] = results[i]['confidence']

        if movie_name not in data_manager.uploaded_reviews:
            data_manager.uploaded_reviews[movie_name] = []
        data_manager.uploaded_reviews[movie_name].extend(reviews)

        os.remove(file_path)

        # 返回时过滤掉情感相关字段，只保留用户、评分、评论内容
        preview_reviews = []
        for review in reviews[:20]:
            preview_reviews.append({
                'user': review.get('user', '上传用户'),
                'rating': review.get('rating', 3),
                'content': review.get('content', ''),
                'time': review.get('time', '')
            })

        return jsonify({
            'success': True,
            'movie_name': movie_name,
            'count': len(reviews),
            'reviews': preview_reviews
        })

    except Exception as e:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/search_movie', methods=['POST'])
def api_search_movie():
    movie_name = request.json.get('movie_name', '').strip()
    return jsonify(data_manager.search_movie_info(movie_name))

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """智能推荐API - 基于当前登录用户"""
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
        method = "深度学习推荐（召回层+排序层）"
    else:
        watched_set = set(watched_list)
        candidate = [m for m in all_movies if m['id'] not in watched_set]
        candidate.sort(key=lambda x: x.get('rating', 0), reverse=True)
        recommendations = [{
            **m,
            'predicted_score': m.get('rating', 8.0) / 2,
            'recommendation_score': m.get('rating', 8.0) / 2
        } for m in candidate[:top_n]]
        method = "基础评分推荐（模型训练中）"

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
        'method': '基于上传数据的智能推荐',
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
            'message': f'模型训练完成！共 {len(ratings_data)} 条训练数据，{epochs} 轮训练',
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

    # 使用BERT进行情感分析
    texts = [r['content'] for r in reviews]
    results = data_manager.sentiment_analyzer.predict_batch(texts)

    for i, review in enumerate(reviews):
        review['sentiment'] = results[i]['sentiment']
        review['confidence'] = results[i]['confidence']

    # 统计情感分布
    sentiments = [r['sentiment'] for r in reviews]
    counts = Counter(sentiments)
    total = len(reviews)

    positive_pct = counts.get('positive', 0) / total * 100 if total > 0 else 0
    neutral_pct = counts.get('neutral', 0) / total * 100 if total > 0 else 0
    negative_pct = counts.get('negative', 0) / total * 100 if total > 0 else 0

    # 计算平均评分
    ratings = [r.get('rating', 0) for r in reviews if r.get('rating', 0) > 0]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0

    # 计算平均置信度
    confidences = [r.get('confidence', 0) for r in reviews]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # 生成情感分布饼图
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
        'reviews': reviews[:100]
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
    bars = ax.bar(["⭐ 1星", "⭐⭐ 2星", "⭐⭐⭐ 3星", "⭐⭐⭐⭐ 4星", "⭐⭐⭐⭐⭐ 5星"],
                  star_counts, color=['#ff6b6b', '#ffa07a', '#ffd966', '#98d98e', '#6bcf7f'],
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
    """生成评论词云图"""
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

    # 使用jieba分词
    words = jieba.cut(all_text)

    # 停用词表
    stopwords = set(['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '个', '也', '这',
                     '那', '到', '说', '为', '以', '吧', '吗', '呢', '啊', '哦', '呀', '啦', '电影', '这部', '一部',
                     '真的', '觉得', '感觉', '非常', '特别', '一个', '没有', '不是', '就是', '但是', '还是', '可以',
                     '很', '太', '看', '让', '给', '会', '能', '都', '也', '有', '在', '还', '就', '与', '和',
                     '对', '于', '其', '这', '那', '么', '怎么', '什么', '为什么', '如何', '哪', '哪些', '哪里',
                     '怎样', '等等', '哈哈', '真的', '特别', '比较', '一般', '有点', '有些', '感觉', '觉得', '好像',
                     '似乎', '可能', '大概', '还是', '就是', '但是', '然而', '所以', '因为', '因此', '于是', '然后'])

    # 过滤词汇
    filtered_words = []
    for word in words:
        word = word.strip()
        if len(word) >= 2 and word not in stopwords and not word.isdigit():
            filtered_words.append(word)

    if not filtered_words:
        return jsonify({'success': False, 'error': '没有足够的词汇生成词云'})

    # 统计词频
    word_freq = Counter(filtered_words)

    # 限制词汇数量
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:100])

    # 尝试生成词云图
    if WORDCLOUD_AVAILABLE:
        try:
            # 查找中文字体路径
            font_path = None
            possible_fonts = [
                'C:/Windows/Fonts/simhei.ttf',  # Windows黑体
                'C:/Windows/Fonts/msyh.ttc',    # Windows微软雅黑
                '/System/Library/Fonts/PingFang.ttc',  # macOS
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # Linux文泉驿
                'simhei.ttf',
                'msyh.ttc'
            ]
            for font in possible_fonts:
                if os.path.exists(font):
                    font_path = font
                    break

            # 创建词云对象
            wc = WordCloud(
                width=800,
                height=600,
                background_color='white',
                font_path=font_path,
                max_words=100,
                relative_scaling=0.5,
                colormap='viridis',
                random_state=42
            )

            # 生成词云
            wc.generate_from_frequencies(top_words)

            # 保存到内存
            buffer = BytesIO()
            wc.to_image().save(buffer, format='PNG')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return jsonify({'success': True, 'image': image_base64})

        except Exception as e:
            print(f"生成词云失败: {e}")
            return _generate_bar_chart_fallback(top_words, name)
    else:
        # 如果没有wordcloud库，使用柱状图
        print("wordcloud库未安装，使用柱状图替代")
        return _generate_bar_chart_fallback(top_words, name)

def _generate_bar_chart_fallback(word_freq, name):
    """生成柱状图作为词云的备选方案"""
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]

    fig, ax = plt.subplots(figsize=(12, 8))
    words_top = [w[0] for w in top_words]
    counts_top = [w[1] for w in top_words]
    bars = ax.barh(words_top, counts_top, color='#667eea', edgecolor='white', linewidth=1)
    ax.set_xlabel('词频', fontsize=12, fontweight='bold')
    ax.set_title(f'《{name}》高频词汇 Top 20', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for bar, count in zip(bars, counts_top):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, str(count),
                ha='left', va='center', fontsize=10)
    plt.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
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
        # 使用爬虫搜索电影
        if data_manager.spider:
            movie_id, title = data_manager.spider.search_movie_id(movie_name)
            if movie_id:
                # 获取电影详情（包含海报URL）
                detail = data_manager.spider.get_movie_detail(movie_id)
                if detail and detail.get('poster'):
                    # 下载海报到本地
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

        # 爬虫失败时返回None
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

if __name__ == '__main__':
    print("=" * 60)
    print("豆瓣电影数据分析与推荐系统 - BERT情感分析版")
    print("=" * 60)
    print("页面结构: 首页 | 影视分析 | 推荐引擎 | 社交管理")
    print("支持格式: Excel (.xlsx, .xls) 和 CSV (.csv)")
    print("情感分析: BERT预训练模型 (bert-base-chinese)")
    print("模型特点: 12层Transformer、110M参数、上下文语义理解")
    print("电影盲盒: 基于用户上传数据的观影类型智能推荐")
    print("AI助手: 智谱AI大模型智能对话")
    print("=" * 60)
    print("访问地址: http://127.0.0.1:5000")
    print("=" * 60)
    print("首次运行会下载BERT模型（约400MB），请耐心等待...")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)