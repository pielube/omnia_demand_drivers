# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:05:54 2024

@author: ucbvplu
"""

import pandas as pd
import numpy as np

def add_iso3(df):
    
    data = pd.read_csv('inputs/mapping_regions_countries/iiasa_country_mapping.csv')
    iiasacountry_to_iso3 = data.set_index('Country')['ISO3'].to_dict()
        
    df['ISO3'] = df['Region'].map(iiasacountry_to_iso3)
    df.insert(df.columns.get_loc('Region') + 1, 'ISO3', df.pop('ISO3'))
    
    return df

def add_omnia_region(df):
    
    data = pd.read_csv('inputs/mapping_regions_countries/OMNIA_region_mapping_241120.csv')
    iso3_to_region = data.set_index('ISO3')['region'].to_dict()
        
    df['OMNIA'] = df['ISO3'].map(iso3_to_region)
    df.insert(df.columns.get_loc('Region') + 2, 'OMNIA',  df.pop('OMNIA'))
    
    return df
   
def get_driver_proj(ssp_data, driver, ssp_scen):
    
    # Select the relevant driver
    filtered_data = ssp_data[ssp_data['Variable'] == driver]
    
    # Select historical data and relevant SSP scenario projection
    filtered_data = filtered_data[filtered_data['Scenario'].str.contains('Historical|'+ssp_scen, case=False, na=False)]
    
    # Reshape the dataframe to merge annual data for the same "Region"
    filtered_data = filtered_data.melt(
        id_vars=['Model', 'Scenario', 'Region', 'ISO3','OMNIA','Variable', 'Unit'], 
        var_name='Year', 
        value_name='Value')

    # Pivot the data to combine annual data by "Region" in rows with years as columns
    filtered_data = filtered_data.pivot_table(
        index=['Region', 'Model', 'ISO3', 'OMNIA', 'Variable', 'Unit'], 
        columns='Year', 
        values='Value').reset_index()
    
    # Add ISO3 country code column and OMNIA regions column
    filtered_data = add_iso3(filtered_data)
    filtered_data = add_omnia_region(filtered_data)

    # Start from the aligned pivoted data
    expanded_years_data = filtered_data.copy()
    
    year_col_index = next((i for i, col in enumerate(filtered_data.columns) if col.isnumeric()),None)

    # Get the range of years from the current columns
    current_years = sorted(int(year) for year in expanded_years_data.columns[year_col_index:])
    full_years = list(range(current_years[0], current_years[-1] + 1))

    # Reindex the dataframe to include all years, filling missing values with interpolation
    expanded_years_data = expanded_years_data.set_index(['Region', 'ISO3','OMNIA','Model','Variable','Unit'])
    expanded_years_data = expanded_years_data.reindex(columns=map(str, full_years)).interpolate(axis=1).reset_index()
       
    return expanded_years_data

def get_gdppercapita_proj(pop_data, gdp_data):
    
    # Align dataframes on both rows and columns
    aligned_df1, aligned_df2 = gdp_data.align(pop_data, join='inner', axis=0)

    # Separate numeric and non-numeric columns
    numeric_cols = aligned_df1.select_dtypes(include=[np.number]).columns
    non_numeric_cols = aligned_df1.select_dtypes(exclude=[np.number]).columns

    # Perform element-wise division on numeric columns
    numeric_result = aligned_df1[numeric_cols] / aligned_df2[numeric_cols] *1000 # USD(2017) per person

    # Combine numeric and non-numeric columns, preserving the original order
    final_result = pd.concat([aligned_df1[non_numeric_cols], numeric_result], axis=1)
    final_result['Model'] = 'OECD and IIASA'
    final_result['Variable'] = 'GDPpc'
    final_result['Unit'] = 'USD_2017 per person'
    
    return final_result

def get_imf_gr(path_imf, path_iso3_imf):
    
    """
    Function to extract growth rates
    2020-2029 from IMF data
    """
    
    df = pd.read_excel(path_imf)
    country_mapping = pd.read_csv(path_iso3_imf)
    
    # Rename the first column to "Country" for clarity
    df = df.rename(columns={df.columns[0]: "Country"})

    # Step 4: Merge the main dataframe with the country mapping
    df_with_iso3 = df.merge(country_mapping, on="Country", how="inner")

    columns_to_drop =  ["Country"] + [col for col in df_with_iso3.columns if isinstance(col, int) and col < 2020]
    df_with_iso3 = df_with_iso3.drop(columns=columns_to_drop)

    iso3_with_no_data = df_with_iso3.loc[df_with_iso3.isin(["no data"]).any(axis=1), "ISO3"].tolist()
    
    print (f'IMF has data for {len(df_with_iso3)} countries')
    print('Countries dropped cause not all relevant years have data:')
    print(iso3_with_no_data)
    print('These countries will not be updated with IMF data')


    df_with_iso3 = df_with_iso3[~df_with_iso3.isin(["no data"]).any(axis=1)]
    df_with_iso3.set_index("ISO3", inplace=True)
    
    df_with_iso3 = df_with_iso3.astype(float)

    
    return df_with_iso3

def get_data_venezuela(path_ven):
    
    df_ven = pd.read_csv(path_ven)
    
    # Conversion factor from 1990 USD to 2017 USD
    # Based on U.S. GDP implicit price deflator (GDPDEF, index 2017=100) data from FRED
    # https://fred.stlouisfed.org/data/GDPDEF
    # 1990 = 59.30525 
    # 2017 = 100
    factor_1990_to_2017 = 100/59.3025
    df_ven["value"] = df_ven["value"] * factor_1990_to_2017 / 1000.0
    
    # 2. Build 5-year grid and pivot
    target_years = list(range(1950, 2101, 5))
    
    pivot = df_ven.pivot_table(
        index="scenario",
        columns="year",
        values="value",
        aggfunc="first"
    )
    
    pivot_5 = pivot.reindex(columns=target_years)  # keeps exact years like 1975, 1980, ...
    
    # 3. Metadata with same index as pivot (scenario names)
    meta = pd.DataFrame(index=pivot_5.index)
    meta["Model"]   = "GCAM"
    meta["Scenario"]= meta.index
    meta["Region"]  = "Venezuela"
    meta["ISO3"]    = "VEN"
    meta["OMNIA"]   = "LAC"
    meta["Variable"]= "GDP|PPP"
    meta["Unit"]    = "billion USD_2017/yr"
    
    # 4. Concatenate WITHOUT resetting the pivot index (this was the bug before)
    omnia = pd.concat(
        [meta[["Model","Scenario","Region","ISO3","OMNIA","Variable","Unit"]],
         pivot_5],
        axis=1
    )
    
    # 5. Rename year columns to strings and enforce column order
    year_cols = [c for c in omnia.columns if isinstance(c, int)]
    omnia.rename(columns={y: str(y) for y in year_cols}, inplace=True)
    
    ordered_cols = (
        ["Model","Scenario","Region","ISO3","OMNIA","Variable","Unit"] +
        [str(y) for y in target_years]
    )
    omnia = omnia[ordered_cols]

    return omnia

def update_gdp_with_imf(gdp_data, growth_data_imf):
            
    # Calculate year-on-year growth
    gdp_growth = gdp_data.iloc[:, 6:].pct_change(axis=1) * 100

    # Add the non-year columns back to the new dataframe
    growth_data = pd.concat([gdp_data.iloc[:, :6], gdp_growth.iloc[:, 1:]], axis=1)

    growth_data.set_index("ISO3", inplace=True)
    growth_data = growth_data.iloc[:, 5:]

    growth_data.index = growth_data.index.astype(str)
    growth_data_imf.index = growth_data_imf.index.astype(str)
    growth_data.columns = growth_data.columns.astype(str)
    growth_data_imf.columns = growth_data_imf.columns.astype(str)

    growth_data.update(growth_data_imf)

    growth_data = growth_data/100

    # Select years 2020 to 2100
    years = [str(year) for year in range(2020, 2101)]

    # Iterate through each year and update gdp_data
    for year in years:
        prev_year = str(int(year) - 1)  # Previous year
        if prev_year in gdp_data.columns and year in growth_data.columns:
            # Update GDP for the current year
            gdp_data[year] = gdp_data[prev_year] * (1 + growth_data[year].values)
            
    return gdp_data

def calculate_growth_yoy(df, years):
    
    years_str = [str(i) for i in years]
    year_columns = [col for col in df.columns if col.isnumeric()]
    years_drop = [item for item in year_columns if item not in years_str]
    
    df = df.drop(columns=years_drop, errors='ignore')
    df = df.drop(columns=['Variable','Model', 'Unit'], errors='ignore')
    
    df[years_str] = df[years_str].div(df[years_str].shift(axis=1))
    df[str(years[0])] = 1
    
    return df       

def calculate_growth_base_year(df, years):
    """
    Calculates the growth in each column with respect to the base year.
    
    Parameters:
    - df: The DataFrame containing the data (with columns as years and rows as different regions or entities).
    - base_year: The year against which growth will be calculated.

    Returns:
    - A DataFrame with growth percentages calculated with respect to the base year.
    """
    
    years_str = [str(i) for i in years]
    year_columns = [col for col in df.columns if col.isnumeric()]
    years_drop = [item for item in year_columns if item not in years_str]
    
    df = df.drop(columns=years_drop, errors='ignore')
    df = df.drop(columns=['Variable','Model', 'Unit'], errors='ignore')

    # Create a new DataFrame to store the growth results
    growth_df = df.copy()

    # Calculate growth for each column (i.e., year) with respect to the base year
    for year in df.columns:
        if year != years[0]:  # Skip the base year itself
            growth_df[year] = df[year] / df[str(years[0])]

    return growth_df
 
"""
Part 1 - Extracting SSP data
"""

# Read dataset and add ISO3 and OMNIA regions columns
ssp_data = pd.read_csv('inputs/Global data/SSP_database_2024.csv', comment='#')
ssp_data = add_iso3(ssp_data) # Add ISO3
ssp_data = add_omnia_region(ssp_data) # Add OMNIA regions

# Read OMNIA regions map and IIASA countries map
omnia_map = pd.read_csv('inputs/mapping_regions_countries/OMNIA_region_mapping_241120.csv')
ssp_map = pd.read_csv('inputs/mapping_regions_countries/iiasa_country_mapping.csv')

# Getting lists of countries per each dataset in ISO3 format
models_ssp_df = ssp_data['Model'].unique().tolist() # Data sources in the SSP dataset
countries_ssp_iiasa = ssp_data[ssp_data['Model'] == models_ssp_df[0]]['ISO3'].unique().tolist()
countries_ssp_oecd  = ssp_data[ssp_data['Model'] == models_ssp_df[1]]['ISO3'].unique().tolist()
countries_omnia = omnia_map['ISO3'].to_list()
countries_full = list(set(countries_ssp_iiasa + countries_ssp_oecd + countries_omnia))

# Counting counties in each dataset and OMNIA map
print('')
print(f'A total of {len(countries_full)} countries are listed among all sources')
print('')
print (f'SSP - IIASA (population) {len(countries_ssp_iiasa)}')
print (f'SSP - OECD (GDP) {len(countries_ssp_oecd)}')
print (f'OMNIA - Definition of regions {len(countries_omnia)}')
print('')

# Dropping countries that are in SSP dataset but not in OMNIA map
dropped_countries1 = ssp_data.loc[ssp_data['OMNIA'].isna(), 'Region'].unique().tolist()
ssp_data = ssp_data[ssp_data['OMNIA'].notna()]
print(f'{len(dropped_countries1)} countries that are in SSP data but not in OMNIA dropped')
print(dropped_countries1)
print('')

# Dropping countries for which we have population data (IIASA) but not GDP data (OECD)
countries_ssp_iiasa = ssp_data[ssp_data['Model'] == models_ssp_df[0]]['ISO3'].unique().tolist()
countries_ssp_oecd  = ssp_data[ssp_data['Model'] == models_ssp_df[1]]['ISO3'].unique().tolist()
countries_ssp_oecd.append('VEN') # Adding back Venezuela
countries_ssp_oecd = sorted(countries_ssp_oecd) # Adding back Venezuela
diff_countries = list(set(countries_ssp_iiasa)- set(countries_ssp_oecd))
iso3_to_sspnames = ssp_map.set_index('ISO3')['Country'].to_dict()
diff_countries = [iso3_to_sspnames.get(item, item) for item in diff_countries]
ssp_data = ssp_data[ssp_data['ISO3'].isin(countries_ssp_oecd)]
print(f'{len(diff_countries)} countries that are in IIASA data but not in OECD dropped')
print('Venezuela not dropped as data from GCAM will be used')
print(diff_countries)
print('')

# IMF data
path_imf = './inputs/Global data/IMF-WEO2024-GDPproj.xlsx'
path_iso3_imf = './inputs/mapping_regions_countries/imf_country_mapping.csv'
growth_data_imf = get_imf_gr(path_imf, path_iso3_imf)
print('')

# GCAM data for Venezuela
path_ven = './inputs/L102.gdp_mil90usd_Scen_R_VN.csv'
df_ven = get_data_venezuela(path_ven)
ssp_data = ssp_data[df_ven.columns]
ssp_data = pd.concat([ssp_data, df_ven], axis=0, ignore_index=True)

print('We reintroduced GDP data for Venezuela based on GCAM data')
print('')
  
"""
Part 2 - Building year on year projections for OMNIA
"""

# ssp_scen = ['SSP1','SSP2','SSP3','SSP4','SSP5']
ssp_scen = ['SSP2']
years_omnia = [2019,2023,2025,2030,2035,2040,2045,2050,2060,2070,2080,2090,2100]


for scenario in ssp_scen:
    
    # Country level data first - Pop and GDP

    # Population
    pop_data = get_driver_proj(ssp_data, 'Population', scenario)
    columns_to_drop = [str(year) for year in range(1950, 2019)]
    pop_data.drop(columns=columns_to_drop, inplace=True)
    
    # GDP data
    gdp_data = get_driver_proj(ssp_data, 'GDP|PPP', scenario)
    columns_to_drop = [str(year) for year in range(1975, 2019)]
    gdp_data.drop(columns=columns_to_drop, inplace=True)
    rows_with_nan_index = gdp_data[gdp_data.isna().any(axis=1)].index.tolist()
    gdp_data.loc[rows_with_nan_index[0],'2019'] =  gdp_data.loc[rows_with_nan_index[0],'2020']
    gdp_data.loc[rows_with_nan_index[1],'2019'] =  gdp_data.loc[rows_with_nan_index[1],'2020']
    gdp_data = update_gdp_with_imf(gdp_data, growth_data_imf) # Update GDP data with IMF data
    
    # OMNIA regions level - Pop, GDP and GDPpc

    pop_data_regions = pop_data.groupby('OMNIA').sum(numeric_only=True)
    gdp_data_regions = gdp_data.groupby('OMNIA').sum(numeric_only=True)
    gdppc_data_regions = get_gdppercapita_proj(pop_data_regions, gdp_data_regions)

    # Growth year on year
    
    # pop_gr_omnia = calculate_growth_yoy(pop_data_regions, years_omnia)
    # gdp_gr_omnia = calculate_growth_yoy(gdp_data_regions, years_omnia)
    # gdppc_gr_omnia = calculate_growth_yoy(gdppc_data_regions, years_omnia)
    
    # output_excel_path = f'ssps_growth_rates/gr_{scenario}_population_yoy.xlsx'
    # pop_gr_omnia.to_excel(output_excel_path, index=True)
    
    # output_excel_path = f'ssps_growth_rates/gr_{scenario}_gdp_yoy.xlsx'
    # pop_gr_omnia.to_excel(output_excel_path, index=True)

    # output_excel_path = f'ssps_growth_rates/gr_{scenario}_gdppc_yoy.xlsx'
    # pop_gr_omnia.to_excel(output_excel_path, index=True)

    # Growth base year
    
    pop_gr_omnia_by = calculate_growth_base_year(pop_data_regions, years_omnia)    
    gdp_gr_omnia_by = calculate_growth_base_year(gdp_data_regions, years_omnia)
    gdppc_gr_omnia_by = calculate_growth_base_year(gdppc_data_regions, years_omnia)
    
    output_excel_path = f'outputs/gr_{scenario}_population_baseyear.xlsx'
    pop_gr_omnia_by.to_excel(output_excel_path, index=True)
    
    output_excel_path = f'outputs/gr_{scenario}_gdp_baseyear.xlsx'
    gdp_gr_omnia_by.to_excel(output_excel_path, index=True)

    output_excel_path = f'outputs/gr_{scenario}_gdppc_baseyear.xlsx'
    gdppc_gr_omnia_by.to_excel(output_excel_path, index=True)



      

    
    
    
    
    
    
    


