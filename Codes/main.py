from data_processing import EVDataProcessor
from visualizations import EVVisualizer


def main():
    # Define dataset path (change this if your file path changes)
    data_path = "EV_Charging_Patterns_Dirty.csv"

    # Create the processor and run the full pipeline step-by-step (chain style)
    processor = (
        EVDataProcessor(data_path)          # Initialize processor with the file path
        .load()                             # Load the CSV into a dataframe
        .rename_columns()                   # Rename columns for readability/consistency
        .drop_user_id()                     # Drop UserId column (not needed)
        .print_uniques_raw()                # Print unique values before cleaning (optional)
        .clean_categoricals_auto()          # Auto-clean categorical columns (normalize + alias + canonical)
        .print_uniques_clean()              # Print unique values after cleaning (optional)
        .handle_missing_and_dropna()        # Print missing info and drop missing rows (same logic)
        .process_datetime_columns()         # Split start/end timestamps into date+time columns
        .clean_symbols_and_features()       # Remove %/$ signs, cast types, create engineered features
        .check_duplicates()                 # Check duplicates and print duplicate row count
        .check_outliers_iqr()               # Check outliers using IQR rule and print counts
    )

    # Extract the final cleaned dataframe
    df_clean = processor.get_df()

    # Create a visualizer with the cleaned dataframe
    viz = EVVisualizer(df_clean)

    # Run all plots in order
    viz.plot_all()


# Run the main function only if this file is executed directly
if __name__ == "__main__":
    main()
