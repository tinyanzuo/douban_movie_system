"""
推荐引擎模块 - MCP模型上下文协议 + 召回层(向量检索) + 排序层(NCF)
"""
import torch
import numpy as np
from datetime import datetime
from collections import Counter
from torch import nn, optim
from torch.utils.data import DataLoader
import os

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


class MCPContextManager:
    """MCP上下文管理器 - 管理用户上下文、电影上下文和推荐上下文"""

    def __init__(self):
        self.user_context = {}
        self.movie_context = {}
        self.session_context = {}
        self.knowledge_base = {
            'genre_mappings': {
                '剧情': ['肖申克的救赎', '霸王别姬', '阿甘正传', '美丽人生', '放牛班的春天'],
                '科幻': ['星际穿越', '盗梦空间', '流浪地球', '黑客帝国', '阿凡达'],
                '动作': ['让子弹飞', '速度与激情', '碟中谍', '叶问', '战狼2'],
                '爱情': ['泰坦尼克号', '怦然心动', '你的名字', '情书', '爱乐之城'],
                '喜剧': ['三傻大闹宝莱坞', '夏洛特烦恼', '唐人街探案', '疯狂的石头', '西虹市首富'],
                '动画': ['千与千寻', '龙猫', '疯狂动物城', '寻梦环游记', '冰雪奇缘'],
                '悬疑': ['盗梦空间', '看不见的客人', '消失的爱人', '调音师', '误杀']
            },
            'director_famous': {
                '克里斯托弗·诺兰': ['盗梦空间', '星际穿越', '记忆碎片'],
                '宫崎骏': ['千与千寻', '龙猫', '天空之城'],
                '陈凯歌': ['霸王别姬', '黄土地'],
                '弗兰克·德拉邦特': ['肖申克的救赎']
            }
        }

    def update_user_context(self, user_id, watched_movies, ratings=None):
        self.user_context[user_id] = {
            'watched_movies': watched_movies,
            'ratings': ratings or {},
            'last_update': datetime.now().isoformat(),
            'preference_vector': self._compute_preference_vector(watched_movies)
        }
        return self.user_context[user_id]

    def update_movie_context(self, movie_id, movie_info):
        self.movie_context[movie_id] = {
            'info': movie_info,
            'last_update': datetime.now().isoformat()
        }
        return self.movie_context[movie_id]

    def _compute_preference_vector(self, watched_movies):
        genre_counter = Counter()
        for movie in watched_movies:
            genre = movie.get('genre', '')
            if genre:
                for g in genre.split('/'):
                    genre_counter[g.strip()] += 1
        total = sum(genre_counter.values())
        if total > 0:
            return {g: c/total for g, c in genre_counter.items()}
        return {}

    def get_user_context(self, user_id):
        return self.user_context.get(user_id, {})

    def get_movie_context(self, movie_id):
        return self.movie_context.get(movie_id, {})

    def query_knowledge_base(self, query_type, key):
        if query_type == 'genre':
            return self.knowledge_base.get('genre_mappings', {}).get(key, [])
        elif query_type == 'director':
            return self.knowledge_base.get('director_famous', {}).get(key, [])
        return []


class DeepRecommendationEngine:
    """MCP模型上下文协议推荐引擎 - MCP上下文管理 + 召回层(向量检索) + 排序层(NCF)"""

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

        self.user_embeddings = {}
        self.movie_embeddings = {}

        self.mcp_context = MCPContextManager()

        self.data_manager = None

    def set_data_manager(self, dm):
        self.data_manager = dm

    def build_vocab(self, users, movies):
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

        print(f"🚀 开始训练MCP推荐模型...")
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
        print("✅ MCP推荐模型训练完成!")
        return True

    def _extract_embeddings(self):
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

    def recall_by_embedding(self, user_id, all_movies, top_k=200):
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

    def get_recommendation_for_uploaded_data(self, data_id, all_movies, top_n=10):
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
                import re
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
                'match_reason': f"MCP协议分析 | 类型匹配度: {round(genre_score * 100)}%"
            })

        recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
        return recommendations[:top_n]

    def get_recommendation(self, user_id, all_movies, top_n=10, exclude_watched=True, watched_list=None):
        """完整推荐流程：召回层 + 排序层 (增强版)"""
        watched_set = set(watched_list) if watched_list else set()

        print(f"\n{'='*60}")
        print(f"🎯 MCP推荐引擎启动")
        print(f"👤 用户: {user_id}")
        print(f"📊 总电影数: {len(all_movies)}, 已看: {len(watched_set)}")
        print(f"{'='*60}")

        # MCP上下文更新
        watched_movies_info = [m for m in all_movies if m['id'] in watched_set]
        user_context = self.mcp_context.update_user_context(user_id, watched_movies_info)
        print(f"📝 MCP用户上下文已更新 | 偏好: {list(user_context.get('preference_vector', {}).keys())}")

        candidate_movies = [m for m in all_movies if not exclude_watched or m['id'] not in watched_set]

        if not candidate_movies:
            return []

        recall_count = min(200, len(candidate_movies))
        recalled_movies = self.recall_by_embedding(user_id, candidate_movies, recall_count)
        print(f"🔍 召回层(向量检索): {len(recalled_movies)} 个候选")

        if not recalled_movies:
            print("⚠️ 召回无结果，使用热度召回")
            recalled_movies = [m for m in candidate_movies][:50]

        print(f"🎯 排序层(NCF深度排序): 精排中...")
        ranked_results = self.rank_by_model(user_id, recalled_movies)

        recommendations = []
        for movie, score in ranked_results[:top_n]:
            genre = movie.get('genre', '')
            if genre:
                match_reason = f"MCP协议分析 | 根据您的观影偏好推荐 | 类型: {genre}"
            else:
                match_reason = "MCP协议分析 | 综合评分和热度推荐"

            recommendations.append({
                **movie,
                'predicted_score': round(score, 2),
                'recommendation_score': round(score, 2),
                'match_reason': match_reason
            })

        print(f"✅ MCP推荐完成，共 {len(recommendations)} 部电影")
        print(f"{'='*60}\n")

        return recommendations

    def predict_rating(self, user_id, movie_id):
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
        print(f"💾 MCP模型已保存到: {filepath}")

    def load_model(self, filepath='deep_recommend_model.pth'):
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
        print(f"✅ MCP模型已加载: {filepath}")
        return True