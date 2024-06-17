# Compare Models by Variable Importance

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################


# Code #############################################################################################################
if __name__ == '__main__':


    # Step 1: Read the CSV into a DataFrame
    df = pd.read_csv('/home/s1949330/Documents/scratch/diss_data/model/model_stats.csv')

    # Step 2: Identify and categorize variables
    variable_names = df.iloc[:, 0]
    model_columns = df.iloc[:, 1:]

    # Define categories and their renaming
    categories = {
        'Topography': ['SRTM_'],
        'Reflectance': ['SR_B'],
        'Phenology': ['NDVI'],
        'C-Band': ['VV', 'VH'],
        'L-Band': ['HV', 'HH']
    }

    # Step 3: Initialize DataFrames to store sum importance for each category
    sum_importance_df = pd.DataFrame(index=list(categories.keys()), columns=model_columns.columns)
    sum_importance_adjusted_df = pd.DataFrame(index=list(categories.keys()), columns=model_columns.columns)

    # Step 4: Calculate sum variable importance for each category across all models
    for category_name, substr_list in categories.items():
        category_variables = [var for var in variable_names if any(substr in var for substr in substr_list)]
        if category_variables:
            category_data = df[df.iloc[:, 0].isin(category_variables)]
            category_sum_importance = category_data.iloc[:, 1:].sum()

            # Get number of variables in the category
            num_variables = len(category_variables)
            
            # Divide sum importance by number of variables (adjusted)
            category_sum_importance_adjusted = category_sum_importance / num_variables

            # Assign the summed importance to the corresponding rows in DataFrames
            sum_importance_df.loc[category_name] = category_sum_importance
            sum_importance_adjusted_df.loc[category_name] = category_sum_importance_adjusted

    # Step 5: Transpose the DataFrames for plotting
    sum_importance_df = sum_importance_df.T  # Transpose DataFrame to have models as rows
    sum_importance_adjusted_df = sum_importance_adjusted_df.T  # Transpose DataFrame to have models as rows

    # Step 6: Plot stacked bar charts for both original and adjusted importances
    plt.rcParams['font.family'] = 'Arial'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # Plot original importances
    sum_importance_df.plot(kind='bar', stacked = True, ax = ax1, alpha = 0.75)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Importance (%)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    # Plot adjusted importances
    sum_importance_adjusted_df.plot(kind = 'bar', stacked = True, ax = ax2, legend = False, alpha = 0.75)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Weight-Adjusted Importance')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.show()