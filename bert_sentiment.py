"""
BERT中文情感分析模型
使用预训练的BERT模型进行三分类情感分析（正面/中性/负面）
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import json
import os
import pickle
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 豆瓣真实评论数据（扩充版）
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
        "太精彩了！从头到尾都没有冷场，强烈安利！",
        "看完热血沸腾，这才是我们需要的国产大片！",
        "剧本扎实，演技在线，制作精良，无可挑剔。",
        "感动到流泪，这部电影让我重新思考了人生。",
        "二刷依然感动，每次看都有新的收获。",
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
        "没有太多亮点，但也没有明显缺点。",
        "平平无奇，就是一部普通的电影。",
        "可看可不看，不是特别推荐。",
        "整体还行，但不会让人印象深刻。",
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
        "看得我想睡觉，太无聊了。",
        "这就是传说中的烂片，名不虚传。",
        "完全不能理解好评哪里来的，太假了。",
        "浪费时间，还不如在家睡觉。",
    ]
}


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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTSentimentClassifier(nn.Module):
    """BERT情感分类模型"""
    def __init__(self, model_name='bert-base-chinese', num_classes=3, dropout=0.3):
        super(BERTSentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BERTSentimentAnalyzer:
    """BERT情感分析器"""

    def __init__(self, model_name='bert-base-chinese', device=None, use_pretrained=True):
        """
        初始化BERT情感分析器

        Args:
            model_name: BERT模型名称，可选 'bert-base-chinese', 'hfl/rbt3' 等
            device: 运行设备，None表示自动选择
            use_pretrained: 是否使用预训练模型（如果为False，将使用规则匹配）
        """
        self.model_name = model_name
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.is_trained = False
        self.label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
        self.idx_to_label = {0: 'positive', 1: 'neutral', 2: 'negative'}
        self.use_pretrained = use_pretrained

        print(f"\n{'='*50}")
        print(f"正在初始化BERT情感分析器...")
        print(f"设备: {self.device}")
        print(f"模型: {model_name}")
        print(f"{'='*50}")

        if use_pretrained:
            try:
                # 初始化tokenizer
                print("正在加载tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                print("✅ Tokenizer加载成功")

                # 尝试加载已训练的模型
                if os.path.exists('best_bert_sentiment.pth'):
                    print("发现已训练的模型，正在加载...")
                    self._load_model('best_bert_sentiment.pth')
                    self.is_trained = True
                    print("✅ 已训练模型加载成功")
                else:
                    # 训练新模型
                    print("未找到已训练模型，开始训练...")
                    self._train_model()
            except Exception as e:
                print(f"⚠️ BERT模型初始化失败: {e}")
                print("将使用规则匹配作为备选方案")
                self.use_pretrained = False
                self.is_trained = False
        else:
            print("使用规则匹配模式（备选方案）")

    def _prepare_data(self):
        """准备训练数据"""
        print("正在准备训练数据...")

        texts = []
        labels = []

        # 加载正面评论
        for review in DOUBAN_REAL_REVIEWS["positive"]:
            texts.append(review)
            labels.append(0)  # positive

        # 加载中性评论
        for review in DOUBAN_REAL_REVIEWS["neutral"]:
            texts.append(review)
            labels.append(1)  # neutral

        # 加载负面评论
        for review in DOUBAN_REAL_REVIEWS["negative"]:
            texts.append(review)
            labels.append(2)  # negative

        # 数据增强：添加轻微变化的评论
        augmented_texts = []
        augmented_labels = []

        for text, label in zip(texts, labels):
            # 同义词替换增强
            if label == 0:  # positive
                new_text = text.replace('好', '棒').replace('经典', '杰作')
                augmented_texts.append(new_text)
                augmented_labels.append(label)
                new_text2 = text.replace('震撼', '感动').replace('推荐', '值得看')
                augmented_texts.append(new_text2)
                augmented_labels.append(label)
            elif label == 2:  # negative
                new_text = text.replace('差', '烂').replace('失望', '后悔')
                augmented_texts.append(new_text)
                augmented_labels.append(label)
                new_text2 = text.replace('无聊', '乏味').replace('浪费时间', '浪费生命')
                augmented_texts.append(new_text2)
                augmented_labels.append(label)

        texts.extend(augmented_texts)
        labels.extend(augmented_labels)

        print(f"总训练数据量: {len(texts)} 条")
        print(f"正面: {labels.count(0)}, 中性: {labels.count(1)}, 负面: {labels.count(2)}")

        # 划分训练集和验证集
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.15, random_state=42, stratify=labels
        )

        return train_texts, val_texts, train_labels, val_labels

    def _train_model(self, epochs=5, batch_size=8, learning_rate=2e-5):
        """训练BERT模型（减少epochs和batch_size以适应CPU）"""
        # 准备数据
        train_texts, val_texts, train_labels, val_labels = self._prepare_data()

        # 创建数据集
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 初始化模型
        self.model = BERTSentimentClassifier(self.model_name).to(self.device)

        # 优化器 - 使用torch.optim.AdamW替代
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # 学习率调度器
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # 损失函数
        criterion = nn.CrossEntropyLoss()

        print(f"\n🚀 开始训练BERT模型...")
        print(f"   训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
        print(f"   Epochs: {epochs}, Batch Size: {batch_size}")
        print(f"   学习率: {learning_rate}")
        print("-" * 50)

        best_val_accuracy = 0

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # 验证阶段
            val_accuracy, _ = self._evaluate(val_loader)

            print(f"Epoch {epoch+1}/{epochs} - 训练损失: {avg_train_loss:.4f} - 验证准确率: {val_accuracy:.4f}")

            # 保存最佳模型
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self._save_model('best_bert_sentiment.pth')
                print(f"  ✅ 保存最佳模型 (准确率: {val_accuracy:.4f})")

        # 加载最佳模型
        self._load_model('best_bert_sentiment.pth')
        self.is_trained = True

        print(f"\n{'='*50}")
        print(f"✅ BERT模型训练完成！")
        print(f"📊 最佳验证准确率: {best_val_accuracy:.4f}")
        print(f"{'='*50}")

        # 打印最终分类报告
        _, final_report = self._evaluate(val_loader, print_report=True)

    def _evaluate(self, data_loader, print_report=False):
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)

        if print_report:
            print("\n📊 分类报告:")
            print(classification_report(all_labels, all_preds,
                                        target_names=['正面', '中性', '负面']))

        return accuracy, classification_report(all_labels, all_preds)

    def predict_sentiment(self, text):
        """
        预测单条评论的情感

        Args:
            text: 评论文本

        Returns:
            (sentiment, confidence): 情感标签和置信度
        """
        if not self.use_pretrained or not self.is_trained or self.model is None:
            return self._rule_based_fallback(text), 0.5

        try:
            self.model.eval()

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

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(logits, dim=1).item()
                confidence = probabilities[0][pred_class].item()

            sentiment = self.label_map[pred_class]
            return sentiment, confidence

        except Exception as e:
            print(f"BERT预测失败: {e}, 使用规则备选")
            return self._rule_based_fallback(text), 0.5

    def predict_batch(self, texts, batch_size=32):
        """
        批量预测评论情感

        Args:
            texts: 评论文本列表
            batch_size: 批次大小

        Returns:
            results: 包含sentiment和confidence的字典列表
        """
        if not self.use_pretrained or not self.is_trained or self.model is None:
            results = []
            for text in texts:
                sentiment = self._rule_based_fallback(text)
                results.append({'sentiment': sentiment, 'confidence': 0.6})
            return results

        try:
            self.model.eval()
            results = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]

                # Tokenize批量文本
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )

                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)

                with torch.no_grad():
                    logits = self.model(input_ids, attention_mask)
                    probabilities = torch.softmax(logits, dim=1)
                    pred_classes = torch.argmax(logits, dim=1)
                    confidences = probabilities[range(len(batch_texts)), pred_classes]

                for j, text in enumerate(batch_texts):
                    sentiment = self.label_map[pred_classes[j].item()]
                    confidence = confidences[j].item()
                    results.append({'sentiment': sentiment, 'confidence': confidence})

            return results

        except Exception as e:
            print(f"BERT批量预测失败: {e}")
            results = []
            for text in texts:
                sentiment = self._rule_based_fallback(text)
                results.append({'sentiment': sentiment, 'confidence': 0.5})
            return results

    def _rule_based_fallback(self, text):
        """
        基于规则的备用情感分析（当BERT不可用时使用）

        Args:
            text: 评论文本

        Returns:
            sentiment: 情感标签
        """
        positive_words = ['好', '棒', '赞', '喜欢', '爱', '精彩', '经典', '震撼', '感动', '推荐',
                          '值得', '好看', '优秀', '神作', '完美', '杰出', '出色', '惊喜', '炸裂',
                          '太棒', '超棒', '绝了', '牛', '厉害', '给力', '过瘾', '爽']
        negative_words = ['差', '烂', '失望', '垃圾', '无聊', '糟糕', '失败', '尴尬', '浪费时间',
                          '后悔', '烂片', '狗血', '无语', '崩溃', '差劲', '垃圾片', '骗钱', '敷衍',
                          '尴尬', '莫名其妙', '不知所云', '逻辑混乱']

        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

    def _save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'label_map': self.label_map
        }, path)
        print(f"💾 模型已保存: {path}")

    def _load_model(self, path):
        """加载模型"""
        if not os.path.exists(path):
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model = BERTSentimentClassifier(self.model_name).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model_name = checkpoint.get('model_name', self.model_name)
            self.label_map = checkpoint.get('label_map', self.label_map)
            print(f"✅ 模型已加载: {path}")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'use_pretrained': self.use_pretrained,
            'num_classes': 3,
            'classes': ['positive', 'neutral', 'negative']
        }


# 测试代码
if __name__ == '__main__':
    print("测试BERT情感分析器...")

    # 初始化分析器
    analyzer = BERTSentimentAnalyzer(model_name='bert-base-chinese', use_pretrained=True)

    # 测试用例
    test_texts = [
        "这部电影太好看了，强烈推荐！",
        "一般般吧，没什么特别的感觉。",
        "太烂了，浪费时间！",
        "剧情精彩，演技在线，值得一看。",
        "中规中矩，没有惊喜也没有失望。",
        "什么垃圾电影，完全看不下去！"
    ]

    print("\n" + "="*50)
    print("测试预测结果:")
    print("="*50)

    for text in test_texts:
        sentiment, confidence = analyzer.predict_sentiment(text)
        print(f"文本: {text}")
        print(f"情感: {sentiment}, 置信度: {confidence:.4f}")
        print("-" * 30)

    # 批量测试
    print("\n批量测试:")
    results = analyzer.predict_batch(test_texts)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['sentiment']} (置信度: {result['confidence']:.4f})")