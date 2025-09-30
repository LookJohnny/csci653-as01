"""
Enhanced feature engineering for Amazon reviews
Extracts sentiment, engagement metrics, and product features
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

# Simple sentiment analysis (can upgrade to TextBlob or VADER)
POSITIVE_WORDS = set(['great', 'excellent', 'amazing', 'love', 'best', 'perfect',
                      'awesome', 'good', 'wonderful', 'fantastic', 'recommend'])
NEGATIVE_WORDS = set(['bad', 'terrible', 'awful', 'worst', 'poor', 'hate',
                      'disappointing', 'broken', 'useless', 'waste'])

def simple_sentiment(text):
    """Calculate simple sentiment score from text"""
    if pd.isna(text):
        return 0.0

    text = str(text).lower()
    words = re.findall(r'\b\w+\b', text)

    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    return (pos_count - neg_count) / total

def extract_review_features(df):
    """Extract features from individual reviews"""
    print("Extracting review-level features...")

    # 1. Text length features
    df['title_length'] = df['title'].fillna('').astype(str).str.len()
    df['text_length'] = df['text'].fillna('').astype(str).str.len()
    df['has_long_review'] = (df['text_length'] > 200).astype(int)

    # 2. Image presence
    df['has_image'] = df['images'].apply(
        lambda x: 1 if pd.notna(x) and len(x) > 0 else 0
    )

    # 3. Sentiment from text
    print("Computing sentiment scores...")
    df['sentiment'] = df['text'].progress_apply(simple_sentiment)

    # 4. Engagement metrics
    df['helpful_vote'] = pd.to_numeric(df['helpful_vote'], errors='coerce').fillna(0)
    df['has_helpful_votes'] = (df['helpful_vote'] > 0).astype(int)

    # 5. Rating features
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(3.0)
    df['is_extreme_rating'] = ((df['rating'] == 5) | (df['rating'] == 1)).astype(int)

    # 6. Verified purchase
    df['verified_purchase'] = df['verified_purchase'].fillna(False).astype(int)

    return df

def aggregate_product_features(df):
    """Aggregate features at product level"""
    print("Aggregating product-level features...")

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    df['week_start'] = df['timestamp'].dt.to_period('W-MON').dt.start_time

    # Weekly aggregation per product
    agg_dict = {
        # Basic counts
        'reviews': ('parent_asin', 'size'),

        # Rating features
        'rating_mean': ('rating', 'mean'),
        'rating_std': ('rating', 'std'),
        'pct_5star': ('rating', lambda x: (x == 5).mean()),
        'pct_1star': ('rating', lambda x: (x == 1).mean()),

        # Engagement features
        'helpful_sum': ('helpful_vote', 'sum'),
        'helpful_mean': ('helpful_vote', 'mean'),
        'helpful_max': ('helpful_vote', 'max'),
        'pct_with_helpful': ('has_helpful_votes', 'mean'),

        # Content features
        'avg_text_length': ('text_length', 'mean'),
        'avg_title_length': ('title_length', 'mean'),
        'pct_long_reviews': ('has_long_review', 'mean'),

        # Image features
        'pct_with_images': ('has_image', 'mean'),
        'image_count': ('has_image', 'sum'),

        # Sentiment features
        'sentiment_mean': ('sentiment', 'mean'),
        'sentiment_std': ('sentiment', 'std'),

        # Verification
        'verified_count': ('verified_purchase', 'sum'),
        'verified_ratio': ('verified_purchase', 'mean'),

        # Extreme ratings
        'pct_extreme': ('is_extreme_rating', 'mean')
    }

    weekly = df.groupby(['parent_asin', 'week_start']).agg(**agg_dict).reset_index()

    # Fill NaN std with 0
    weekly['rating_std'] = weekly['rating_std'].fillna(0)
    weekly['sentiment_std'] = weekly['sentiment_std'].fillna(0)

    # Calculate engagement score
    weekly['engagement_score'] = (
        weekly['helpful_mean'] * 0.3 +
        weekly['pct_with_helpful'] * 100 * 0.2 +
        weekly['pct_long_reviews'] * 100 * 0.2 +
        weekly['pct_with_images'] * 100 * 0.3
    )

    # Quality score
    weekly['quality_score'] = (
        weekly['rating_mean'] * 20 +  # 0-100
        weekly['verified_ratio'] * 30 +  # 0-30
        (1 - weekly['pct_extreme']) * 20 +  # Less polarization = higher quality
        weekly['sentiment_mean'] * 30  # -30 to +30
    )

    return weekly

def add_growth_features(df, top_quantile=0.95, min_reviews=1):
    """Add growth-based features and labels"""
    print("Computing growth features...")

    df = df.sort_values(['parent_asin', 'week_start']).reset_index(drop=True)

    def compute_growth(group):
        reviews = group['reviews'].astype(float)

        # Historical features (looking back)
        group['rev_prev4'] = reviews.rolling(window=4, min_periods=1).sum()
        group['rev_prev4_mean'] = reviews.rolling(window=4, min_periods=1).mean()
        group['rev_prev4_std'] = reviews.rolling(window=4, min_periods=1).std().fillna(0)

        # Momentum
        group['review_momentum'] = reviews / (group['rev_prev4_mean'] + 1)

        # Future target (looking forward)
        reversed_sum = reviews.iloc[::-1].rolling(window=12, min_periods=1).sum().iloc[::-1]
        group['rev_next12'] = (reversed_sum - reviews).clip(lower=0)

        # Growth score
        group['growth_score'] = group['rev_next12'] / (group['rev_prev4'] + 1.0)

        # Engagement growth
        helpful = group['helpful_sum'].astype(float)
        group['helpful_growth'] = helpful / (helpful.rolling(window=4, min_periods=1).mean() + 1)

        # Sentiment trend
        sentiment = group['sentiment_mean']
        group['sentiment_trend'] = sentiment - sentiment.rolling(window=4, min_periods=1).mean().fillna(sentiment)

        return group

    df = df.groupby('parent_asin', group_keys=False).progress_apply(compute_growth)

    # Filter by minimum reviews
    df = df[df['rev_prev4'] >= min_reviews].copy()

    # Create label based on growth quantile
    thresh = df.groupby('week_start')['growth_score'].quantile(top_quantile)
    df = df.merge(thresh.rename('growth_threshold'), left_on='week_start', right_index=True, how='left')
    df['label_top5'] = (df['growth_score'] >= df['growth_threshold']).astype(int)

    print(f"Label distribution: {df['label_top5'].value_counts().to_dict()}")

    return df

def main():
    parser = argparse.ArgumentParser(description='Build enhanced feature dataset')
    parser.add_argument('--input', required=True, help='Input parquet file with reviews')
    parser.add_argument('--out', required=True, help='Output CSV file')
    parser.add_argument('--top_quantile', type=float, default=0.95, help='Top quantile for labels')
    parser.add_argument('--min_reviews', type=int, default=1, help='Minimum reviews threshold')
    args = parser.parse_args()

    # Enable progress bars for pandas
    tqdm.pandas()

    print("Loading data...")
    if args.input.endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    print(f"Loaded {len(df):,} reviews")
    print(f"Columns: {df.columns.tolist()}")

    # Extract review features
    df = extract_review_features(df)

    # Aggregate to product-week level
    weekly = aggregate_product_features(df)

    print(f"Created {len(weekly):,} product-week observations")

    # Add growth features and labels
    panel = add_growth_features(weekly,
                                top_quantile=args.top_quantile,
                                min_reviews=args.min_reviews)

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {out_path}...")
    panel.to_csv(out_path, index=False)

    print(f"\nFinal dataset: {len(panel):,} rows, {len(panel.columns)} columns")
    print("\nFeature columns:")
    for col in sorted(panel.columns):
        print(f"  - {col}")

    print("\nFeature statistics:")
    numeric_cols = panel.select_dtypes(include=[np.number]).columns
    print(panel[numeric_cols].describe())

if __name__ == '__main__':
    main()