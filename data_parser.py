import pandas as pd
import json
import numpy as np
from pathlib import Path

class PropertyMarketDataParser:
    def __init__(self, data_dir=None):
        if data_dir is None:
            current_dir = Path.cwd()
            if (current_dir / 'kpi.json').exists():
                self.data_dir = current_dir
            elif (current_dir / 'data' / 'kpi.json').exists():
                self.data_dir = current_dir / 'data'
            else:
                self.data_dir = current_dir
        else:
            self.data_dir = Path(data_dir)
        
        print(f"Looking for data files in: {self.data_dir}")

    def parse_cpi_data(self):
        """Parse consumer price index (monthly -> quarterly)"""
        file_path = self.data_dir / 'kpi.json'
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cpi_values = []
        periods = list(data['dimension']['Tid']['category']['label'].keys())
        
        for i, period in enumerate(periods):
            if 'M' in period:
                year = int(period[:4])
                month = int(period[5:7])
                value = data['value'][i*2]
                cpi_values.append((year, month, value))
        
        quarterly_cpi = {}
        for year in range(2015, 2025):
            for q in range(1, 5):
                months = [(q-1)*3 + 1, (q-1)*3 + 2, (q-1)*3 + 3]
                q_values = [v for y, m, v in cpi_values 
                           if y == year and m in months]
                if q_values:
                    quarterly_cpi[f"{year}K{q}"] = np.mean(q_values)
        
        return quarterly_cpi
    
    def parse_policy_rate_data(self):
        """Parse policy rate (monthly -> quarterly)"""
        file_path = self.data_dir / 'policy_rate.json'
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rate_values = []
        periods = list(data['dimension']['Tid']['category']['label'].keys())
        
        for i, period in enumerate(periods):
            if 'M' in period:
                year = int(period[:4])
                month = int(period[5:7])
                value = data['value'][i]
                rate_values.append((year, month, value))
        
        quarterly_rate = {}
        for year in range(2015, 2025):
            for q in range(1, 5):
                months = [(q-1)*3 + 1, (q-1)*3 + 2, (q-1)*3 + 3]
                q_values = [v for y, m, v in rate_values 
                           if y == year and m in months]
                if q_values:
                    quarterly_rate[f"{year}K{q}"] = np.mean(q_values)
        
        return quarterly_rate
    
    def parse_population_data(self):
        """Parse population change data"""
        file_path = self.data_dir / 'population_change.json'
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        population_data = {}
        counties = list(data['dimension']['Region']['category']['label'].values())
        quarters = list(data['dimension']['Tid']['category']['label'].values())
        
        for i, county in enumerate(counties):
            county_clean = county.split(' - ')[0]
            population_data[county_clean] = {}
            for j, quarter in enumerate(quarters):
                idx = i * len(quarters) + j
                if idx < len(data['value']):
                    population_data[county_clean][quarter] = data['value'][idx]
        
        return population_data
    
    def parse_price_index_data(self):
        """Parse property price index data"""
        file_path = self.data_dir / 'property_price_index.json'
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        price_data = {}
        regions = list(data['dimension']['Region']['category']['label'].values())
        quarters = list(data['dimension']['Tid']['category']['label'].values())
        
        for i, region in enumerate(regions):
            price_data[region] = {}
            for j, quarter in enumerate(quarters):
                idx = i * len(quarters) + j
                if idx < len(data['value']):
                    price_data[region][quarter] = data['value'][idx]
        
        return price_data
    
    def parse_revenue_data(self):
        """Parse property sales volume data"""
        try:
            with open(self.data_dir / 'revenue_properties_2020_2023.json', 'r', encoding='utf-8') as f:
                data_old = json.load(f)
        except:
            data_old = None
            
        try:
            with open(self.data_dir / 'revenue_properties_2024.json', 'r', encoding='utf-8') as f:
                data_new = json.load(f)
        except:
            data_new = None
        
        revenue_data = {}
        
        if data_old:
            old_regions = list(data_old['dimension']['Region']['category']['label'].values())
            old_quarters = list(data_old['dimension']['Tid']['category']['label'].values())
            
            for i, region in enumerate(old_regions):
                region_clean = region.split(' (')[0].split(' - ')[0]
                if region_clean not in revenue_data:
                    revenue_data[region_clean] = {}
                
                for j, quarter in enumerate(old_quarters):
                    idx = j * len(old_regions) * 2 + i * 2
                    if idx < len(data_old['value']):
                        revenue_data[region_clean][quarter] = data_old['value'][idx]
        
        if data_new:
            new_regions = list(data_new['dimension']['Region']['category']['label'].values())
            new_quarters = list(data_new['dimension']['Tid']['category']['label'].values())
            
            for i, region in enumerate(new_regions):
                region_clean = region.split(' - ')[0]
                if region_clean not in revenue_data:
                    revenue_data[region_clean] = {}
                
                for j, quarter in enumerate(new_quarters):
                    idx = i * 2 + j * len(new_regions) * 2
                    if idx < len(data_new['value']):
                        revenue_data[region_clean][quarter] = data_new['value'][idx]
        
        return revenue_data
    
    def create_unified_dataset(self):
        """Create unified dataset"""
        print("=== Parsing all data sources ===")
        
        cpi_data = self.parse_cpi_data()
        policy_rate = self.parse_policy_rate_data()
        population = self.parse_population_data()
        price_index = self.parse_price_index_data()
        revenue = self.parse_revenue_data()
        
        print("=== Creating unified dataset ===")
        
        quarters = [f"{y}K{q}" for y in range(2015, 2025) for q in range(1, 5)]
        main_regions = ['Oslo', 'Rogaland', 'Vestland', 'Trøndelag', 'Innlandet', 
                       'Agder', 'Nordland', 'Møre og Romsdal']
        
        data_rows = []
        
        for region in main_regions:
            for quarter in quarters:
                if int(quarter[:4]) > 2024:
                    continue
                    
                row = {
                    'region': region,
                    'quarter': quarter,
                    'year': int(quarter[:4]),
                    'quarter_num': int(quarter[-1])
                }
                
                if quarter in cpi_data:
                    row['cpi'] = cpi_data[quarter]
                
                if quarter in policy_rate:
                    row['policy_rate'] = policy_rate[quarter]
                
                if region in population and quarter in population[region]:
                    row['population_change'] = population[region][quarter]
                
                price_region = self.map_region_to_price(region)
                if price_region and price_region in price_index and quarter in price_index[price_region]:
                    row['price_index'] = price_index[price_region][quarter]
                
                if region in revenue and quarter in revenue[region]:
                    row['sales_volume'] = revenue[region][quarter]
                elif 'Trøndelag' in revenue and quarter in revenue['Trøndelag']:
                    row['sales_volume'] = revenue['Trøndelag'][quarter]
                
                data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        df = df.dropna()  # Remove rows with missing critical data
        
        return df
    
    def map_region_to_price(self, region):
        """Map regions to price index regions"""
        mapping = {
            'Oslo': 'Oslo med Bærum',
            'Rogaland': 'Stavanger',
            'Vestland': 'Bergen',
            'Trøndelag': 'Trondheim',
            'Innlandet': 'Innlandet',
            'Agder': 'Agder og Rogaland uten Stavanger',
            'Nordland': 'Nord-Norge',
            'Møre og Romsdal': 'Møre og Romsdal og Vestland uten Bergen'
        }
        return mapping.get(region)
    
    def save_processed_data(self, df, output_file='processed_data.csv'):
        """Save processed data"""
        output_dir = Path('processed')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / output_file
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved processed data to: {output_path.absolute()}")
        return output_path

if __name__ == "__main__":
    parser = PropertyMarketDataParser()
    df = parser.create_unified_dataset()
    parser.save_processed_data(df)
    
    print(f"\n=== Data Summary ===")
    print(f"Total rows: {len(df)}")
    print(f"Regions: {df['region'].unique()}")
    print(f"Years: {df['year'].min()}-{df['year'].max()}")
    print(f"Columns: {df.columns.tolist()}")