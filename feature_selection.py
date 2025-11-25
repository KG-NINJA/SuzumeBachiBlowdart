import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

def analyze_feature_correlation(features_df):
    """
    特徴間の相関を分析し、多重共線性をチェック
    """
    # 相関行列計算
    corr_matrix = features_df.corr().abs()
    
    # 相関>0.9のペアを検出
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_pairs = [
        (col, row) 
        for col in upper_triangle.columns 
        for row in upper_triangle.index 
        if upper_triangle.loc[row, col] > 0.9
    ]
    
    print(f"相関>0.9のペア（多重共線性）: {len(high_corr_pairs)}")
    for feat1, feat2 in high_corr_pairs:
        corr_val = corr_matrix.loc[feat1, feat2]
        print(f"  {feat1} ↔ {feat2}: {corr_val:.3f}")
    
    return high_corr_pairs, corr_matrix


def select_important_features(features_df, target, n_features=20):
    """
    重要度ベースで上位N個の特徴を選択
    """
    # 特徴重要度計算（分類タスク）
    selector = SelectKBest(f_classif, k=min(n_features, features_df.shape[1]))
    selector.fit(features_df, target)
    
    # スコアで並べ替え
    scores = pd.DataFrame({
        'feature': features_df.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    print(f"\n重要度Top {n_features}:")
    for idx, row in scores.head(n_features).iterrows():
        print(f"  {row['feature']:20s}: {row['score']:8.2f}")
    
    selected_features = scores.head(n_features)['feature'].tolist()
    
    return selected_features, scores


def remove_high_corr_features(features_df, target, corr_threshold=0.9):
    """
    相関の高い特徴を除外し、重要度で再選択
    """
    # ステップ1: 相関を分析
    high_corr_pairs, corr_matrix = analyze_feature_correlation(features_df)
    
    # ステップ2: 相関の高い特徴ペアから片方を削除
    features_to_drop = set()
    for feat1, feat2 in high_corr_pairs:
        # 重要度が低い方を削除候補に
        features_to_drop.add(feat1)  # 簡略版
    
    features_reduced = features_df.drop(columns=list(features_to_drop))
    print(f"\n相関削減後: {features_df.shape[1]} → {features_reduced.shape[1]} 特徴")
    
    # ステップ3: 重要度で上位20を選択
    selected_features, scores = select_important_features(
        features_reduced, target, n_features=20
    )
    
    return features_reduced[selected_features], selected_features, scores
