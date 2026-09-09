#import argparse
import pandas as pd
import os
import numpy as np

def merge_csv_files(root_dir, output_dir, output_filename, seeds):

    # Get column order from a sample CSV file in a sub-sub-directory (root/seed/results) 
    sample_csv_path = os.path.join(root_dir, str(seeds[0]), 'results', 'env_data_marl.csv')
    column_order = pd.read_csv(sample_csv_path).columns.tolist()

    # Initial empty DataFrame
    merged_df = pd.DataFrame(columns=column_order)

    # Loop through each sub-directory
    for subdir in seeds:  
        csv_path = os.path.join(root_dir, str(subdir), 'results', 'env_data_marl.csv')
        # Load and merge the marl CSV files
        df = pd.read_csv(csv_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)              

    # Reorder columns based on the sample CSV file
    merged_df = merged_df[column_order]
    
    # Rename columns as in the paper
    merged_df = merged_df.rename(columns={
        'Unnamed: 0':       'year',
        'Comp. Int.':       'Competitive.Intensity',
        'country_spread':   'Market.size.variation',
        'Home_size-firm_0': 'Home.market.size.firm.A',
        'Home_size-firm_1': 'Home.market.size.firm.B',
        'LOF OR redux':     'LoF.discount',
    })

    # Process / add additional columns for analyses
    # Collocation Index (Alcácer et al. 2013)
    merged_df['colloc.0'] = merged_df['Common_markets-firm_0'] / merged_df['Markets-firm_0']
    merged_df['colloc.1'] = merged_df['Common_markets-firm_1'] / merged_df['Markets-firm_1']

    # Environmental variables
    half = round(merged_df['Home_country-firm_0'].max() / 2) # get middle point of countries
    merged_df['Remoteness.0']  = (half - merged_df['Home_country-firm_0']).abs()
    merged_df['Remoteness.1']  = (half - merged_df['Home_country-firm_1']).abs()
    merged_df['Firm.distance'] = (merged_df['Home_country-firm_0'] - merged_df['Home_country-firm_1']).abs()

    # Home market size ratio
    merged_df['Home.market.size.ratio'] = (
        merged_df['Home.market.size.firm.A'] / merged_df['Home.market.size.firm.B']
    )

    # Adjust cumulative profit share to (sensible) positive values for visualization (only used for rewards in RL)
    pmin = np.minimum(merged_df['Cumu_profit-firm_0'], merged_df['Cumu_profit-firm_1'])
    condition = (merged_df['Cumu_profit-firm_0'] < 0) | (merged_df['Cumu_profit-firm_1'] < 0)
    # Add |smallest negative return| to both returns and +1 for same home country
    merged_df['Cumu_profit-firm_0_corr'] = np.where(
        condition,
        merged_df['Cumu_profit-firm_0'] + (1 + pmin.abs()),
        merged_df['Cumu_profit-firm_0']
    )
    merged_df['Cumu_profit-firm_1_corr'] = np.where(
        condition,
        merged_df['Cumu_profit-firm_1'] + (1 + pmin.abs()),
        merged_df['Cumu_profit-firm_1']
    )
    # Calculate cumulative return share for each firm (based on corr values)
    total_corr = merged_df['Cumu_profit-firm_0_corr'] + merged_df['Cumu_profit-firm_1_corr']
    merged_df['Cumu_profit_share-firm_0'] = merged_df['Cumu_profit-firm_0_corr'] / total_corr
    merged_df['Cumu_profit_share-firm_1'] = merged_df['Cumu_profit-firm_1_corr'] / total_corr

    # Save the final DataFrame to a CSV file
    output_path = os.path.join(output_dir, output_filename)
    merged_df.to_csv(output_path, index=False)