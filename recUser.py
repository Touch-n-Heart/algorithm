import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# 创建示例用户数据
def create_example_data():
    data = {
        'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'mbti': ['INTJ', 'ESFP', 'ENTP', 'INFJ', 'ESTJ', 'ISFP', 'ENTJ', 'INFP', 'ISTJ', 'ENFJ'],
        'age': [30, 25, 35, 28, 40, 22, 33, 27, 29, 31],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'education': ['Master', 'Bachelor', 'PhD', 'Master', 'Bachelor', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
        'location': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Beijing', 'Chengdu', 'Hangzhou', 'Wuhan', 'Beijing', 'Nanjing'],
        'interests': [
            'technology history', 'fashion travel', 'science music', 'psychology art',
            'business sports', 'food movies', 'technology business', 'writing travel',
            'history science', 'art psychology'
        ],
        'likes': [
            'deep_analysis_1 deep_analysis_2', 'travel_blog_1 fashion_blog_1', 'science_article_1 music_video_1',
            'psychology_paper_1 art_exhibit_1', 'business_news_1 sports_highlight_1', 'food_blog_1 movie_review_1',
            'tech_blog_1 business_article_1', 'travel_blog_2 writing_piece_1', 'science_article_2 history_documentary_1',
            'art_exhibit_2 psychology_paper_2'
        ],
        # 'relationship_status': ['single', 'married', 'single', 'married', 'single', 'single', 'married', 'single', 'married', 'single'],
        'work_industry': ['IT', 'Fashion', 'Science', 'Art', 'Business', 'Food', 'IT', 'Writing', 'Science', 'Art']
    }

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 保存为CSV文件
    df.to_csv('users.csv', index=False)

    return df

# 生成并保存数据
create_example_data()

# 特征选择和预处理
def prepare_pipeline():
    features = ['mbti', 'age', 'gender', 'education', 'location', 'interests', 'likes', 'work_industry']
    
    # 定义预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['age']),
            ('cat', OneHotEncoder(), ['mbti', 'gender', 'education', 'location', 'work_industry']),
            ('text_interests', CountVectorizer(), 'interests'),
            ('text_likes', CountVectorizer(), 'likes')
        ])
    
    return features, preprocessor

# 训练最近邻模型
def train_model(df, features, preprocessor):
    X = preprocessor.fit_transform(df[features])
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X)
    return nbrs

# 基于最近邻进行好友推荐
def recommend_friends(user_id, df, nbrs, preprocessor, features, top_n=5):
    user_data = df[df['user_id'] == user_id]
    if user_data.empty:
        return []

    user_vector = preprocessor.transform(user_data[features])
    distances, indices = nbrs.kneighbors(user_vector, n_neighbors=top_n+1)
    
    # 计算相似度
    similarities = 1 / (1 + distances.flatten()[1:])
    
    # 获取推荐的用户ID和相似度
    recommendations = df.iloc[indices[0][1:]].copy()  # 创建副本以避免警告
    recommendations['similarity'] = similarities
    
    return recommendations[['user_id', 'similarity']]

# 测试函数
def test_recommendation_system(datapath):
    # 准备管道
    features, preprocessor = prepare_pipeline()
    df = pd.read_csv(datapath)
    # 训练模型
    nbrs = train_model(df, features, preprocessor)

    # 对用户1进行好友推荐
    recommended_friends = recommend_friends(1, df, nbrs, preprocessor, features)
    print("Recommended friends for user 1:")
    print(recommended_friends)

# 运行测试
test_recommendation_system("./users.csv")
