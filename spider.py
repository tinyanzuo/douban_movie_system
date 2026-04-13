"""
豆瓣爬虫模块
"""
import re
import time
import random
import os
import hashlib
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

# 导入配置中的常量
from config import LOCAL_MOVIE_CACHE, DOUBAN_SEARCH_CACHE, CACHE_DURATION, LAST_REQUEST_TIME, REQUEST_INTERVAL

# 爬虫相关库可用性检查
REQUESTS_AVAILABLE = True
BeautifulSoup = BeautifulSoup

# 海报存储目录
POSTER_DIR = None

def set_poster_dir(path):
    """设置海报存储目录"""
    global POSTER_DIR
    POSTER_DIR = path
    os.makedirs(POSTER_DIR, exist_ok=True)


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

        # 构建请求头
        parsed_url = urlparse(url)
        host = parsed_url.netloc

        headers = {
            'User-Agent': self.headers.get('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
            'Referer': f'https://movie.douban.com/subject/{movie_id}/',
            'Origin': 'https://movie.douban.com',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Connection': 'keep-alive',
            'Host': host,
            'Cookie': self.headers.get('Cookie', '')
        }

        # 重试 3 次
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
                content_type = resp.headers.get('Content-Type', '').lower()
                if resp.status_code == 200 and ('image' in content_type or 'application/octet-stream' in content_type):
                    with open(local_path, 'wb') as f:
                        f.write(resp.content)
                    print(f"[DEBUG] 海报下载成功: {local_path}")
                    return f"static/posters/{filename}"
                else:
                    print(f"[DEBUG] 海报下载失败 (尝试 {attempt+1}/3): {url}")
                    print(f"  状态码: {resp.status_code}, Content-Type: {content_type}")
            except Exception as e:
                print(f"[DEBUG] 海报下载异常 (尝试 {attempt+1}/3): {e}")

            time.sleep(random.uniform(1, 2))

        return None


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