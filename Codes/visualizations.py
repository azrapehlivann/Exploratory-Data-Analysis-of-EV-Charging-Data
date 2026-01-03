import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class EVVisualizer:
    """
    This class is responsible ONLY for plotting.
    Plot code is intentionally kept the same as your original logic
    to preserve identical visual outputs.
    """

    def __init__(self, df: pd.DataFrame):
        # Store the processed dataframe
        self.df = df

    def plot_avg_charging_rate_by_charger_type(self):
        # Filter out rows where ChargerType is "Level" (your original missing-information category)
        df_filtered = self.df[self.df["ChargerType"] != "Level"]

        # Group by charger type and compute mean charging rate
        grouped_df = df_filtered.groupby(["ChargerType"])["ChargingRateKW"].mean().reset_index()

        # Sort descending by charging rate
        sorted_df = grouped_df.sort_values("ChargingRateKW", ascending=False)

        # Create figure with same size as original
        plt.figure(figsize=(12, 6))

        # Bar plot: charger type vs avg charging rate
        sns.barplot(x="ChargerType", y="ChargingRateKW", data=sorted_df)

        # Hard-coded annotation values (kept identical)
        annotations = [27.67, 25.73, 25.52]

        # Place text annotations on top of bars
        for index, value in enumerate(annotations):
            plt.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=10)

        # Title and axis labels
        plt.title("Average Charging Rate by Charger Type")
        plt.xlabel("Charger Type")
        plt.ylabel("Average Charging Rate")

        # Display the plot
        plt.show()

    def plot_avg_cost_by_location(self):
        # Work on a copy (so we don't permanently add extra column to self.df)
        df2 = self.df.copy()

        # Compute cost per kWh for each row
        df2["Cost/EnergyConsumption"] = df2["ChargingCostUSD"] / df2["EnergyConsumedKwh"]

        # Group by location and take mean cost per kWh
        grouped_df = (
            df2.groupby(["ChargingStationLocation"])["Cost/EnergyConsumption"]
            .mean()
            .reset_index()
        )

        # Print grouped dataframe (same behavior as original)
        print(grouped_df)

        # Sort descending by mean cost per kWh
        sorted_df = grouped_df.sort_values("Cost/EnergyConsumption", ascending=False)

        # Create figure with same size as original
        plt.figure(figsize=(12, 6))

        # Bar plot: location vs avg cost per kWh
        sns.barplot(x="ChargingStationLocation", y="Cost/EnergyConsumption", data=sorted_df)

        # Hard-coded annotation values (kept identical)
        annotations = [1.06, 0.99, 0.97, 0.858, 0.857]

        # Place text annotations on bars
        for index, value in enumerate(annotations):
            plt.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

        # Title and axis labels
        plt.title("Average Charging Cost by Country")
        plt.xlabel("Country")
        plt.ylabel("Average Cost per kWh (USD)")

        # Display plot
        plt.show()

    def plot_avg_energy_consumption_by_day(self):
        # Group by DayOfWeek and compute average energy consumed
        energy_by_day = self.df.groupby("DayOfWeek")["EnergyConsumedKwh"].mean().reset_index()

        # Define correct order of week days
        ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # Make DayOfWeek categorical with defined ordering
        energy_by_day["DayOfWeek"] = pd.Categorical(
            energy_by_day["DayOfWeek"], categories=ordered_days, ordered=True
        )

        # Sort by the categorical order
        energy_by_day = energy_by_day.sort_values("DayOfWeek")

        # Create same figure size
        plt.figure(figsize=(10, 6))

        # Line plot with markers (kept same settings, including color="b")
        sns.lineplot(x="DayOfWeek", y="EnergyConsumedKwh", data=energy_by_day, marker="o", color="b")

        # Annotate each point with its value
        for i in range(len(energy_by_day)):
            plt.text(
                energy_by_day["DayOfWeek"].iloc[i],
                energy_by_day["EnergyConsumedKwh"].iloc[i],
                round(energy_by_day["EnergyConsumedKwh"].iloc[i], 2),
                color="black",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Title and axis labels
        plt.title("Average Energy Consumption by Day of the Week")
        plt.xlabel("Day of the Week")
        plt.ylabel("Average Energy Consumption (kWh)")

        # Show plot
        plt.show()

    def plot_vehicle_model_preferences_by_user_type(self):
        # Filter commuter users
        commuter_df = self.df[self.df["UserType"] == "Commuter"]

        # Count vehicle models for commuters
        commuter_model_counts = commuter_df["VehicleModel"].value_counts().sort_values(ascending=True)

        # Filter long distance travelers
        longdist_df = self.df[self.df["UserType"] == "Long Distance Traveler"]

        # Count vehicle models for long distance travelers
        longdist_model_counts = longdist_df["VehicleModel"].value_counts().sort_values(ascending=True)

        # Create two subplots stacked vertically
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))

        # Plot commuter counts as horizontal bars
        commuter_model_counts.plot(kind="barh", color="skyblue", ax=ax[0])

        # Label subplot
        ax[0].set_ylabel("Commuter")

        # Hide axes spines for clean look
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)
        ax[0].spines["left"].set_visible(False)
        ax[0].spines["bottom"].set_visible(False)

        # Remove x tick labels
        ax[0].set_xticklabels([])

        # Plot long distance traveler counts as horizontal bars
        longdist_model_counts.plot(kind="barh", color="skyblue", ax=ax[1])

        # Label subplot
        ax[1].set_ylabel("Long Distance Traveler")

        # Hide axes spines for clean look
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["left"].set_visible(False)
        ax[1].spines["bottom"].set_visible(False)

        # Remove x tick labels
        ax[1].set_xticklabels([])

        # Annotate commuter bars with counts
        for i, v in enumerate(commuter_model_counts):
            ax[0].text(v + 0.1, i, str(v), va="center", ha="left", color="black")

        # Annotate long distance bars with counts
        for i, v in enumerate(longdist_model_counts):
            ax[1].text(v + 0.1, i, str(v), va="center", ha="left", color="black")

        # Add main title
        fig.suptitle("Vehicle Model Preferences of User Types")

        # Show plot
        plt.show()

    def plot_bubble_location_station_count_vs_avg_energy(self):
        # Count unique charging stations per location
        station_counts = self.df.groupby("ChargingStationLocation")["ChargingStationId"].nunique()

        # Compute average energy consumed per location
        bubble_size = self.df.groupby("ChargingStationLocation")["EnergyConsumedKwh"].mean()

        # Normalize bubble sizes to range [100, 1000]
        sizes = np.interp(
            bubble_size.values,
            (bubble_size.values.min(), bubble_size.values.max()),
            (100, 1000),
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # X values are the location names
        x = station_counts.index

        # Y values are station counts
        y = station_counts.values

        # Scatter plot with bubble sizes
        ax.scatter(x, y, s=sizes, alpha=0.6, edgecolors="w", linewidth=2)

        # Title and labels
        ax.set_title("Size of Average Energy Consumption by Location and Station Count")
        ax.set_xlabel("Location")
        ax.set_ylabel("Charging Station Count")

        # Show plot
        plt.show()

    def plot_heatmap_charge_distribution(self):
        # Create a pivot table: counts by day of week and time of day
        heatmap_data = self.df.groupby(["DayOfWeek", "TimeOfDay"]).size().unstack(fill_value=0)

        # Create heatmap figure
        plt.figure(figsize=(10, 6))

        # Heatmap with no annotations and "Blues" colormap (kept same)
        sns.heatmap(heatmap_data, annot=False, cmap="Blues")

        # Titles and labels
        plt.title("Charge Distribution by Day of the Week and Time of Day", fontsize=14)
        plt.xlabel("Time of Day", fontsize=12)
        plt.ylabel("Day of the Week", fontsize=12)

        # Tight layout for spacing
        plt.tight_layout()

        # Show plot
        plt.show()

    def plot_energy_by_vehicle_model_and_time_of_day(self):
        # Group by vehicle model and time of day, compute mean charging rate and mean energy consumed
        grouped_data_model_time = (
            self.df.groupby(["VehicleModel", "TimeOfDay"])
            .agg({"ChargingRateKW": "mean", "EnergyConsumedKwh": "mean"})
            .reset_index()
        )

        # Create figure
        plt.figure(figsize=(14, 8))

        # Bar plot with hue by time of day (kept identical)
        sns.barplot(
            data=grouped_data_model_time,
            x="VehicleModel",
            y="EnergyConsumedKwh",
            hue="TimeOfDay",
        )

        # Title and labels
        plt.title("Energy Consumed by Vehicle Model and Time of Day", fontsize=14)
        plt.xlabel("Vehicle Model", fontsize=12)
        plt.ylabel("Avg. Energy Consumed (kWh)", fontsize=12)

        # Place legend outside plot area
        plt.legend(title="Time of Day", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Adjust layout
        plt.tight_layout()

        # Show plot
        plt.show()

    def plot_all(self):
        """Run all plots in the same order as the original script."""
        self.plot_avg_charging_rate_by_charger_type()
        self.plot_avg_cost_by_location()
        self.plot_avg_energy_consumption_by_day()
        self.plot_vehicle_model_preferences_by_user_type()
        self.plot_bubble_location_station_count_vs_avg_energy()
        self.plot_heatmap_charge_distribution()
        self.plot_energy_by_vehicle_model_and_time_of_day()
