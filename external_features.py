"""
External Data Features

Adds temporal and economic indicators:
1. Seasonality (holidays, shopping events)
2. Day of week / Month patterns
3. Economic indicators (optional, requires data)
4. Category trends
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# HOLIDAY AND SHOPPING EVENT CALENDAR
# ============================================================================

HOLIDAYS_2020_2024 = {
    # Major US Shopping Holidays
    'Black Friday': [
        '2020-11-27', '2021-11-26', '2022-11-25', '2023-11-24', '2024-11-29'
    ],
    'Cyber Monday': [
        '2020-11-30', '2021-11-29', '2022-11-28', '2023-11-27', '2024-12-02'
    ],
    'Prime Day': [
        '2020-10-13', '2021-06-21', '2022-07-12', '2023-07-11', '2024-07-16'
    ],
    'Christmas': [
        '2020-12-25', '2021-12-25', '2022-12-25', '2023-12-25', '2024-12-25'
    ],
    'New Year': [
        '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01'
    ],
    'Valentines Day': [
        '2020-02-14', '2021-02-14', '2022-02-14', '2023-02-14', '2024-02-14'
    ],
    'Mothers Day': [  # Second Sunday of May
        '2020-05-10', '2021-05-09', '2022-05-08', '2023-05-14', '2024-05-12'
    ],
    'Fathers Day': [  # Third Sunday of June
        '2020-06-21', '2021-06-20', '2022-06-19', '2023-06-18', '2024-06-16'
    ],
    'Back to School': [  # Late August
        '2020-08-24', '2021-08-23', '2022-08-22', '2023-08-21', '2024-08-26'
    ],
    'Halloween': [
        '2020-10-31', '2021-10-31', '2022-10-31', '2023-10-31', '2024-10-31'
    ],
    'Thanksgiving': [  # Fourth Thursday of November
        '2020-11-26', '2021-11-25', '2022-11-24', '2023-11-23', '2024-11-28'
    ]
}

# Shopping seasons
SHOPPING_SEASONS = {
    'Holiday Season': [(11, 15), (12, 31)],  # Mid-Nov to End-Dec
    'Back to School': [(8, 1), (9, 15)],     # Aug-early Sep
    'Summer': [(6, 1), (8, 31)],             # Summer shopping
    'Spring': [(3, 1), (5, 31)],             # Spring shopping
    'Winter Sales': [(1, 1), (2, 15)]        # Post-holiday sales
}

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def add_temporal_features(df, date_col='week_start'):
    """
    Add temporal features from date column

    Features:
    - Year, month, week_of_year
    - Quarter
    - Day of week (for daily data)
    - Is weekend
    - Days to/from major holidays
    - Shopping season indicators
    """
    df = df.copy()

    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Basic temporal
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['quarter'] = df[date_col].dt.quarter
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Month sin/cos encoding (cyclical)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Week sin/cos encoding
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    return df

def add_holiday_features(df, date_col='week_start', window_weeks=2):
    """
    Add holiday proximity features

    For each date, calculates:
    - Is holiday week (binary for each holiday)
    - Weeks until next holiday
    - Weeks since last holiday
    - Holiday intensity score
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Create holiday dataframe
    holiday_dates = []
    for holiday, dates in HOLIDAYS_2020_2024.items():
        for date_str in dates:
            holiday_dates.append({
                'date': pd.to_datetime(date_str),
                'holiday': holiday
            })

    holidays_df = pd.DataFrame(holiday_dates)

    # For each row, check proximity to holidays
    for holiday in HOLIDAYS_2020_2024.keys():
        holiday_key = holiday.lower().replace(' ', '_')

        # Get dates for this holiday
        h_dates = holidays_df[holidays_df['holiday'] == holiday]['date'].values

        # Check if within window
        df[f'near_{holiday_key}'] = df[date_col].apply(
            lambda x: any(abs((x - h).days) <= window_weeks * 7 for h in h_dates)
        ).astype(int)

    # Days to next major shopping event
    major_events = ['Black Friday', 'Cyber Monday', 'Prime Day', 'Christmas']

    def days_to_next_event(date, events):
        future_events = [h for h in holidays_df[holidays_df['holiday'].isin(events)]['date'].values if h >= date]
        if len(future_events) == 0:
            return 365  # No upcoming event
        return (min(future_events) - date).days

    df['days_to_next_event'] = df[date_col].apply(lambda x: days_to_next_event(x, major_events))

    # Weeks to next event (normalized)
    df['weeks_to_next_event'] = (df['days_to_next_event'] / 7).clip(0, 52)

    return df

def add_shopping_season_features(df, date_col='week_start'):
    """Add shopping season indicators"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Extract month and day
    df['_month'] = df[date_col].dt.month
    df['_day'] = df[date_col].dt.day

    # Check each season
    for season, (start, end) in SHOPPING_SEASONS.items():
        season_key = season.lower().replace(' ', '_')

        start_month, start_day = start
        end_month, end_day = end

        if start_month <= end_month:
            # Same year range
            mask = (
                ((df['_month'] == start_month) & (df['_day'] >= start_day)) |
                ((df['_month'] > start_month) & (df['_month'] < end_month)) |
                ((df['_month'] == end_month) & (df['_day'] <= end_day))
            )
        else:
            # Wraps around year end
            mask = (
                ((df['_month'] == start_month) & (df['_day'] >= start_day)) |
                (df['_month'] > start_month) |
                (df['_month'] < end_month) |
                ((df['_month'] == end_month) & (df['_day'] <= end_day))
            )

        df[f'season_{season_key}'] = mask.astype(int)

    # Drop temp columns
    df = df.drop(['_month', '_day'], axis=1)

    return df

def add_category_trends(df, date_col='week_start', product_col='parent_asin', target_col='reviews'):
    """
    Add category-level trend features

    For each time period:
    - Category average performance
    - Product rank within category
    - Percentile within category
    """
    df = df.copy()

    # Assume categories are implicit in product clusters
    # For explicit categories, join with product metadata

    # Global trends
    global_avg = df.groupby(date_col)[target_col].transform('mean')
    global_std = df.groupby(date_col)[target_col].transform('std')

    df['reviews_vs_global_mean'] = (df[target_col] - global_avg) / (global_std + 1e-6)

    # Percentile rank per week
    df['percentile_rank'] = df.groupby(date_col)[target_col].rank(pct=True)

    # Momentum vs global
    df_sorted = df.sort_values([product_col, date_col])
    df['global_momentum'] = global_avg / (global_avg.shift(1) + 1)

    return df

def add_economic_indicators(df, date_col='week_start'):
    """
    Add economic indicator proxies (if external data available)

    This is a placeholder. In production, you would:
    1. Fetch data from FRED (Federal Reserve Economic Data)
    2. Join consumer confidence index
    3. Join unemployment rate
    4. Join retail sales index

    For now, we add simple proxies.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Proxy: Linear growth trend (stand-in for economic growth)
    df['time_index'] = (df[date_col] - df[date_col].min()).dt.days

    # Proxy: Seasonal consumer confidence (higher in holidays)
    df['consumer_confidence_proxy'] = (
        50 +  # Base
        10 * np.sin(2 * np.pi * df[date_col].dt.dayofyear / 365) +  # Seasonal
        5 * (df[date_col].dt.month == 12).astype(int)  # Holiday boost
    )

    return df

def build_complete_feature_set(df, date_col='week_start'):
    """
    Build complete feature set with all external data

    Returns dataframe with all features
    """
    print("Building complete external feature set...")

    print("  - Adding temporal features...")
    df = add_temporal_features(df, date_col)

    print("  - Adding holiday features...")
    df = add_holiday_features(df, date_col)

    print("  - Adding shopping season features...")
    df = add_shopping_season_features(df, date_col)

    print("  - Adding category trends...")
    df = add_category_trends(df, date_col)

    print("  - Adding economic indicators...")
    df = add_economic_indicators(df, date_col)

    print(f"✓ Complete feature set: {df.shape[1]} columns")

    # List new features
    external_features = [
        col for col in df.columns
        if any(keyword in col for keyword in [
            'year', 'month', 'week', 'quarter', 'day', 'season',
            'holiday', 'near_', 'event', 'sin', 'cos', 'trend',
            'percentile', 'momentum', 'economic', 'confidence', 'weekend'
        ])
    ]

    print(f"  External features added: {len(external_features)}")

    return df, external_features

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input CSV with panel data')
    parser.add_argument('--output', required=True, help='Output CSV with external features')
    parser.add_argument('--date_col', default='week_start', help='Date column name')
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input, parse_dates=[args.date_col])

    print(f"Original shape: {df.shape}")

    # Add all features
    df_enhanced, feature_list = build_complete_feature_set(df, args.date_col)

    print(f"Enhanced shape: {df_enhanced.shape}")

    # Save
    print(f"Saving to {args.output}...")
    df_enhanced.to_csv(args.output, index=False)

    # Save feature list
    feature_list_path = args.output.replace('.csv', '_features.txt')
    with open(feature_list_path, 'w') as f:
        f.write('\n'.join(feature_list))

    print(f"✓ Saved feature list to {feature_list_path}")
    print("Done!")

if __name__ == '__main__':
    main()