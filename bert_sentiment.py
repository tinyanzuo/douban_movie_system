"""
BERT中文情感分析模型 - 微调版
使用微调分类头进行三分类情感分析
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW  # 使用 PyTorch 的 AdamW，兼容性最好
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import re
import json
import hashlib
from collections import Counter
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==================== 豆瓣真实评论数据 ====================
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
        "太精彩了！全程高能，看得我热血沸腾！",
        "非常感动，看完哭得稀里哗啦。",
        "很棒的电影，剧情紧凑，演员演技在线。",
        "值得一看的好电影，强烈安利！",
        "太棒了，完全超出了我的预期！"
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
        "还行，但不会看第二遍。",
        "普普通通，没什么特别的亮点。",
        "可以看，但没必要特意去电影院。",
        "一般水平，没有特别惊艳的地方。",
        "中规中矩的及格作品。"
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
        "烂片，千万别看！",
        "太差了，浪费钱！"
    ]
}


class BERTSentimentClassifier(nn.Module):
    """BERT情感分类器 - 带微调分类头"""

    def __init__(self, model_name='bert-base-chinese', num_classes=3, dropout_rate=0.3):
        super(BERTSentimentClassifier, self).__init__()

        # 加载预训练BERT
        print(f"📥 加载BERT模型: {model_name}")
        self.bert = AutoModel.from_pretrained(model_name)
        self.num_classes = num_classes
        self.hidden_size = self.bert.config.hidden_size

        # 微调分类头（3层全连接网络）
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化分类头权重"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """前向传播"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 使用[CLS] token的表示
        cls_output = outputs.pooler_output

        # 分类头
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits


class SentimentDataset(Dataset):
    """情感分析数据集"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class FineTunedBERTSentimentAnalyzer:
    """微调版BERT情感分析器"""

    def __init__(self, model_path='bert_finetuned_sentiment.pth', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.tokenizer = None
        self.model = None

        # 标签映射: 0=负面, 1=中性, 2=正面
        self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.sentiment_to_idx = {'negative': 0, 'neutral': 1, 'positive': 2}

        print(f"\n{'='*60}")
        print(f"🤖 初始化微调版BERT情感分析器")
        print(f"💻 设备: {self.device}")
        print(f"💾 模型路径: {model_path}")
        print(f"{'='*60}")

        self._initialize()

    def _initialize(self):
        """初始化模型和tokenizer"""
        # 加载tokenizer
        print("📥 加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

        # 加载或创建模型
        if os.path.exists(self.model_path):
            print(f"✅ 加载微调模型: {self.model_path}")
            self.model = BERTSentimentClassifier()
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("✅ 模型加载成功（已微调）")
        else:
            print(f"⚠️ 未找到微调模型，将使用预训练模型")
            print("💡 首次运行会自动进行微调训练...")
            self.model = BERTSentimentClassifier()
            self.model.to(self.device)
            self.model.eval()

    def prepare_training_data(self):
        """准备训练数据"""
        texts = []
        labels = []

        # 正面评论 (label=2)
        for text in DOUBAN_REAL_REVIEWS['positive']:
            texts.append(text)
            labels.append(2)

        # 中性评论 (label=1)
        for text in DOUBAN_REAL_REVIEWS['neutral']:
            texts.append(text)
            labels.append(1)

        # 负面评论 (label=0)
        for text in DOUBAN_REAL_REVIEWS['negative']:
            texts.append(text)
            labels.append(0)

        print(f"📊 原始数据: 正面{len(DOUBAN_REAL_REVIEWS['positive'])}条, "
              f"中性{len(DOUBAN_REAL_REVIEWS['neutral'])}条, "
              f"负面{len(DOUBAN_REAL_REVIEWS['negative'])}条")

        # 数据增强
        augmented_texts, augmented_labels = self._augment_data(texts, labels)
        texts.extend(augmented_texts)
        labels.extend(augmented_labels)

        print(f"📊 增强后: 共{len(texts)}条数据")

        return train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

    def _augment_data(self, texts, labels):
        """数据增强"""
        augmented = []
        aug_labels = []

        # 同义词替换
        synonym_map = {
            '好': ['棒', '赞', '优秀', '出色'],
            '棒': ['好', '赞', '优秀', '出色'],
            '赞': ['好', '棒', '优秀', '出色'],
            '差': ['烂', '糟糕', '劣质', '差劲'],
            '烂': ['差', '糟糕', '劣质', '差劲'],
            '喜欢': ['喜爱', '钟爱', '欣赏', '热爱'],
            '讨厌': ['厌恶', '反感', '嫌弃'],
            '失望': ['失落', '遗憾', '沮丧', '灰心'],
            '精彩': ['出色', '绝妙', '精湛', '精妙'],
            '无聊': ['乏味', '枯燥', '无趣', '沉闷'],
            '感动': ['动容', '感慨', '触动'],
            '震撼': ['震惊', '惊愕', '冲击']
        }

        for text, label in zip(texts, labels):
            for original, synonyms in synonym_map.items():
                if original in text:
                    for syn in synonyms[:2]:  # 每个词生成2个变体
                        new_text = text.replace(original, syn, 1)
                        if new_text != text and new_text not in augmented:
                            augmented.append(new_text)
                            aug_labels.append(label)

        return augmented, aug_labels

    def train(self, epochs=10, batch_size=8, learning_rate=2e-5):
        """微调训练模型"""
        print("\n" + "="*60)
        print("🎯 开始微调BERT情感分类器")
        print("="*60)

        # 准备数据
        train_texts, val_texts, train_labels, val_labels = self.prepare_training_data()

        print(f"\n📊 训练集: {len(train_texts)} 条")
        print(f"📊 验证集: {len(val_texts)} 条")

        # 统计类别分布
        train_dist = Counter(train_labels)
        val_dist = Counter(val_labels)
        print(f"📊 训练集分布: 正面{train_dist[2]}, 中性{train_dist[1]}, 负面{train_dist[0]}")
        print(f"📊 验证集分布: 正面{val_dist[2]}, 中性{val_dist[1]}, 负面{val_dist[0]}")

        # 创建数据集
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 优化器和损失函数 - 使用 PyTorch 的 AdamW
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # 训练循环
        best_val_acc = 0
        best_val_loss = float('inf')

        print(f"\n🚀 开始训练 (共{epochs}轮)...\n")

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            total_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # 验证阶段
            val_loss, val_acc = self._evaluate(val_loader, criterion)

            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {avg_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"  ✅ 保存最佳模型 (准确率: {val_acc:.4f})")

        print(f"\n" + "="*60)
        print(f"✅ 微调完成！")
        print(f"📊 最佳验证准确率: {best_val_acc:.4f}")
        print(f"📊 最佳验证损失: {best_val_loss:.4f}")
        print(f"💾 模型保存至: {self.model_path}")
        print("="*60)

        # 重新加载最佳模型
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        return best_val_acc

    def _evaluate(self, val_loader, criterion):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy

    @torch.no_grad()
    def predict_sentiment(self, text):
        """
        预测单条评论的情感

        Args:
            text: 评论文本

        Returns:
            (sentiment, confidence): 情感标签和置信度
        """
        if not text or not text.strip():
            return 'neutral', 0.5

        text = text.strip()

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 预测
        self.model.eval()
        logits = self.model(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=1)

        pred_class = torch.argmax(logits, dim=1).item()
        confidence = torch.max(probabilities).item()

        sentiment = self.label_map[pred_class]

        return sentiment, confidence

    def predict_batch(self, texts, batch_size=32):
        """
        批量预测评论情感

        Args:
            texts: 评论文本列表
            batch_size: 批次大小

        Returns:
            results: 包含sentiment和confidence的字典列表
        """
        if not texts:
            return []

        results = []

        for text in texts:
            if not text or not text.strip():
                results.append({'sentiment': 'neutral', 'confidence': 0.5})
            else:
                sentiment, confidence = self.predict_sentiment(text)
                results.append({'sentiment': sentiment, 'confidence': round(confidence, 4)})

        return results

    def analyze_with_details(self, text):
        """
        详细分析，返回更多信息

        Returns:
            dict: 包含详细分析结果
        """
        sentiment, confidence = self.predict_sentiment(text)

        # 获取各类别概率
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'positive': float(probabilities[2]),
                'neutral': float(probabilities[1]),
                'negative': float(probabilities[0])
            },
            'text_length': len(text),
            'model_type': 'fine_tuned_bert'
        }

    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_name': 'bert-base-chinese',
            'model_type': 'fine_tuned_classifier',
            'device': str(self.device),
            'is_fine_tuned': os.path.exists(self.model_path),
            'model_path': self.model_path,
            'num_classes': 3,
            'classes': ['positive', 'neutral', 'negative'],
            'classifier_layers': '3-layer MLP (768→256→128→3)',
            'description': '微调版BERT情感分析器，使用3层分类头'
        }


# ==================== 规则匹配备选方案（兼容旧版） ====================
class RuleBasedSentimentAnalyzer:
    """规则匹配情感分析器（作为备选）"""

    POSITIVE_KEYWORDS = [
        '好', '棒', '赞', '喜欢', '爱', '精彩', '经典', '震撼', '感动', '推荐',
        '值得', '好看', '优秀', '神作', '完美', '杰出', '出色', '惊喜', '炸裂',
        '太棒', '超棒', '绝了', '牛', '厉害', '给力', '过瘾', '爽', '牛逼'
    ]

    NEGATIVE_KEYWORDS = [
        '差', '烂', '失望', '垃圾', '无聊', '糟糕', '失败', '尴尬', '浪费时间',
        '后悔', '烂片', '狗血', '无语', '崩溃', '差劲', '垃圾片', '骗钱', '敷衍'
    ]

    def predict_sentiment(self, text):
        text_lower = text.lower()

        pos_count = sum(1 for w in self.POSITIVE_KEYWORDS if w in text_lower)
        neg_count = sum(1 for w in self.NEGATIVE_KEYWORDS if w in text_lower)

        if pos_count > neg_count:
            return 'positive', min(0.8, 0.5 + pos_count * 0.1)
        elif neg_count > pos_count:
            return 'negative', min(0.8, 0.5 + neg_count * 0.1)
        else:
            return 'neutral', 0.6

    def predict_batch(self, texts):
        return [{'sentiment': self.predict_sentiment(t)[0], 'confidence': self.predict_sentiment(t)[1]}
                for t in texts]

    def get_model_info(self):
        return {'model_type': 'rule_based', 'description': '基于关键词匹配的情感分析'}


# ==================== 全局单例 ====================
_global_analyzer = None
_fallback_analyzer = None


def get_sentiment_analyzer():
    """
    获取全局情感分析器单例（微调版优先，失败则使用规则匹配）
    """
    global _global_analyzer, _fallback_analyzer

    if _global_analyzer is None:
        try:
            print("\n" + "="*60)
            print("初始化情感分析器...")
            print("="*60)

            analyzer = FineTunedBERTSentimentAnalyzer()

            # 检查是否需要训练
            if not os.path.exists('bert_finetuned_sentiment.pth'):
                print("\n📚 首次运行，开始自动微调训练...")
                print("⏳ 请稍等，大约需要2-3分钟...\n")
                analyzer.train(epochs=8, batch_size=8, learning_rate=2e-5)
            else:
                print("✅ 使用已保存的微调模型")

            _global_analyzer = analyzer
            print("\n✅ 微调版BERT情感分析器已就绪\n")

        except Exception as e:
            print(f"\n⚠️ 微调版初始化失败: {e}")

            # 创建规则匹配备选
            if _fallback_analyzer is None:
                _fallback_analyzer = RuleBasedSentimentAnalyzer()

            _global_analyzer = _fallback_analyzer
            print("✅ 使用规则匹配分析器\n")

    return _global_analyzer


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("测试微调版BERT情感分析器...")

    analyzer = get_sentiment_analyzer()

    # 显示模型信息
    print("\n模型信息:")
    print(json.dumps(analyzer.get_model_info(), ensure_ascii=False, indent=2))

    # 测试用例
    test_texts = [
        "这部电影太好看了，强烈推荐！",
        "一般般吧，没什么特别的感觉。",
        "太烂了，浪费时间！",
        "神作！绝对是我看过最好的电影！",
        "剧情精彩，演技在线，值得一看。",
        "中规中矩，没有惊喜也没有失望。",
        "什么垃圾电影，完全看不下去！",
        "看完只想说：浪费时间！",
        "太震撼了！特效无敌！",
        "还行吧，可以看但不会二刷。"
    ]

    print("\n" + "="*60)
    print("测试预测结果:")
    print("="*60)

    for text in test_texts:
        sentiment, confidence = analyzer.predict_sentiment(text)
        print(f"\n文本: {text}")
        print(f"情感: {sentiment}, 置信度: {confidence:.4f}")

        # 如果是微调版，显示详细概率
        if hasattr(analyzer, 'analyze_with_details'):
            details = analyzer.analyze_with_details(text)
            print(f"概率: 正面={details['probabilities']['positive']:.3f}, "
                  f"中性={details['probabilities']['neutral']:.3f}, "
                  f"负面={details['probabilities']['negative']:.3f}")
        print("-" * 40)