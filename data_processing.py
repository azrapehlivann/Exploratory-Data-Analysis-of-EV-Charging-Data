import re
import numpy as np
import pandas as pd


class EVDataProcessor:
    """
    This class handles the full data processing pipeline:
    - Read CSV
    - Rename columns
    - Drop unnecessary columns
    - Clean categorical columns automatically (normalize + alias + canonical)
    - Handle missing values (drop rows)
    - Split datetime strings into date/time columns
    - Remove symbols like '%' and '$' and convert to numeric
    - Create engineered features
    - Perform duplicate and outlier checks
    """

    def __init__(self, data_path: str):
        # Store dataset path
        self.data_path = data_path

        # DataFrame will be stored here after loading
        self.df: pd.DataFrame | None = None

        # Column renaming dictionary (kept identical to original logic)
        self.rename_map = {
            "user id": "UserId",
            "VehicleModel": "VehicleModel",
            "BatteryCapacitykWh": "BatteryCapacityKWh",
            "Charging_StationID": "ChargingStationId",
            "Charging StationLocation": "ChargingStationLocation",
            "Charging StartTime": "ChargingStartTime",
            "Charging EndTime": "ChargingEndTime",
            "Energy ConsumedKWh": "EnergyConsumedKwh",
            "ChargingDuration_hours": "ChargingDurationHours",
            "ChargingRateKW": "ChargingRateKW",
            "ChargingCostUSD": "ChargingCostUSD",
            "TimeofDay": "TimeOfDay",
            "DayofWeek": "DayOfWeek",
            "State_of_Charge_Start%": "StateOfChargeStart%",
            "State_of_Charge_End%": "StateOfChargeEnd%",
            "Distance_Driven_km": "DistanceDrivenKm",
            "TemperatureC": "TemperatureC",
            "VehicleAge_years": "VehicleAgeYears",
            "ChargerType": "ChargerType",
            "UserType": "UserType",
        }

        # List of categorical columns we want to clean
        self.cat_cols = [
            "VehicleModel",
            "ChargingStationLocation",
            "TimeOfDay",
            "DayOfWeek",
            "ChargerType",
            "UserType",
        ]

        # ---- SMALL alias dictionaries (only truly incorrect variants / typos) ----

        # Vehicle model typos/variants -> corrected normalized form
        self.VEHICLE_ALIASES = {
            "audi e-tro": "audi e-tron",
            "tesla model": "tesla model 3",
            "chevy bol": "chevy bolt",
            "nissan lea": "nissan leaf",
            "hyundai kon": "hyundai kona",
        }

        # Location typos/variants -> corrected normalized form
        self.LOCATION_ALIASES = {
            "los angele": "los angeles",
            "new yor": "new york",
            "san francisc": "san francisco",
            "chicag": "chicago",
            "seattl": "seattle",
        }

        # Time-of-day typos/variants -> corrected normalized form
        self.TIME_ALIASES = {
            "mornin": "morning",
            "afternoo": "afternoon",
            "evenin": "evening",
            "nigh": "night",
        }

        # Day-of-week typos/variants -> corrected normalized form
        self.DAY_ALIASES = {
            "monda": "monday",
            "tuesda": "tuesday",
            "wednesda": "wednesday",
            "thursda": "thursday",
            "frida": "friday",
            "saturda": "saturday",
            "sunda": "sunday",
        }

        # Charger type typos/variants -> corrected normalized form
        self.CHARGER_ALIASES = {
            "dc fast charge": "dc fast charger",
        }

        # User type typos/variants -> corrected normalized form
        self.USER_ALIASES = {
            "commute": "commuter",
            "long-distance travele": "long distance traveler",
            "long-distance traveler": "long distance traveler",
        }

        # ---- CANONICAL dictionaries (final labels, casing guaranteed) ----
        # These ensure output categories match what your plots expect.

        # Vehicle canonical labels
        self.VEHICLE_CANON = {
            "tesla model 3": "Tesla Model 3",
            "chevy bolt": "Chevy Bolt",
            "nissan leaf": "Nissan Leaf",
            "hyundai kona": "Hyundai Kona",
            "audi e-tron": "Audi e-Tron",
        }

        # Location canonical labels
        self.LOCATION_CANON = {
            "los angeles": "Los Angeles",
            "new york": "New York",
            "san francisco": "San Francisco",
            "chicago": "Chicago",
            "seattle": "Seattle",
        }

        # Time-of-day canonical labels
        self.TIME_CANON = {
            "morning": "Morning",
            "afternoon": "Afternoon",
            "evening": "Evening",
            "night": "Night",
        }

        # Day-of-week canonical labels
        self.DAY_CANON = {
            "monday": "Monday",
            "tuesday": "Tuesday",
            "wednesday": "Wednesday",
            "thursday": "Thursday",
            "friday": "Friday",
            "saturday": "Saturday",
            "sunday": "Sunday",
        }

        # Charger type canonical labels
        # IMPORTANT: "Level" must stay as "Level" because you filter it out in a plot.
        self.CHARGER_CANON = {
            "level 1": "Level 1",
            "level 2": "Level 2",
            "dc fast charger": "Dc Fast Charger",
            "level": "Level",
        }

        # User type canonical labels
        self.USER_CANON = {
            "commuter": "Commuter",
            "long distance traveler": "Long Distance Traveler",
        }

    # ---------------------------------------------------------------------
    # Core utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _normalize_text(x):
        """
        Normalize text values:
        - Convert None/NaN to np.nan
        - Convert to string, strip spaces
        - Lowercase everything
        - Remove trailing '#' characters (e.g., "Seattle#" -> "Seattle")
        - Collapse multiple spaces into single space
        """
        # If value is None, keep as missing
        if x is None:
            return np.nan

        # If value is a float NaN, keep as missing
        if isinstance(x, float) and np.isnan(x):
            return np.nan

        # Convert value to string, trim spaces, and lowercase
        s = str(x).strip().lower()

        # If pandas converted NaN to "nan" string, treat it as missing
        if s == "nan":
            return np.nan

        # Remove trailing '#' signs (one or more)
        s = re.sub(r"[#]+$", "", s).strip()

        # Replace multiple whitespace characters with a single space
        s = re.sub(r"\s+", " ", s)

        # Return normalized text
        return s

    @staticmethod
    def _apply_aliases(norm_series: pd.Series, aliases: dict) -> pd.Series:
        """
        Apply alias corrections to a normalized series.
        Example: "audi e-tro" -> "audi e-tron"
        """
        return norm_series.replace(aliases)

    @staticmethod
    def _to_canonical(norm_series: pd.Series, canon: dict) -> pd.Series:
        """
        Convert normalized values into canonical display labels.
        If a value is not found in canon, keep it unchanged
        (so you can notice unexpected categories in unique prints).
        """

        def mapper(v):
            # Preserve missing values
            if v is np.nan:
                return np.nan
            if isinstance(v, float) and np.isnan(v):
                return np.nan
            # Map using canonical dictionary (or keep original)
            return canon.get(v, v)

        return norm_series.map(mapper)

    def _clean_categorical_column(self, col: str, aliases: dict, canon: dict) -> None:
        """
        Clean one categorical column using:
        1) fill missing with "Unknown" temporarily
        2) normalize (strip/lower/remove '#')
        3) alias fixes (typos)
        4) canonical mapping (final labels with correct casing)
        5) convert "unknown" to NaN so missing stays missing
        """

        # Fill missing entries with "Unknown" first (similar intent to original)
        self.df[col] = self.df[col].fillna("Unknown")

        # Normalize values
        s = self.df[col].map(self._normalize_text)

        # Apply alias corrections (typos / variants)
        s = self._apply_aliases(s, aliases)

        # Convert to canonical labels
        s = self._to_canonical(s, canon)

        # Convert "unknown" back into missing value (NaN)
        s = s.replace("unknown", np.nan)
        s = s.replace("Unknown", np.nan)

        # Write cleaned series back into dataframe
        self.df[col] = s

    # ---------------------------------------------------------------------
    # Pipeline steps (chainable)
    # ---------------------------------------------------------------------

    def load(self):
        """Load the CSV file into self.df."""
        self.df = pd.read_csv(self.data_path)
        return self

    def rename_columns(self):
        """Rename columns using rename_map."""
        self._ensure_df()
        self.df = self.df.rename(columns=self.rename_map)
        return self

    def drop_user_id(self):
        """Drop UserId column if it exists."""
        self._ensure_df()
        if "UserId" in self.df.columns:
            self.df.drop("UserId", axis=1, inplace=True)
        return self

    def print_uniques_raw(self):
        """Print unique values BEFORE cleaning (to inspect dirty categories)."""
        self._ensure_df()
        print(self.df["VehicleModel"].unique())
        print(self.df["ChargingStationLocation"].unique())
        print(self.df["TimeOfDay"].unique())
        print(self.df["DayOfWeek"].unique())
        print(self.df["ChargerType"].unique())
        print(self.df["UserType"].unique())
        return self

    def clean_categoricals_auto(self):
        """Clean all categorical columns using the automated approach."""
        self._ensure_df()

        # Clean vehicle model strings
        self._clean_categorical_column("VehicleModel", self.VEHICLE_ALIASES, self.VEHICLE_CANON)

        # Clean location strings
        self._clean_categorical_column(
            "ChargingStationLocation", self.LOCATION_ALIASES, self.LOCATION_CANON
        )

        # Clean time-of-day strings
        self._clean_categorical_column("TimeOfDay", self.TIME_ALIASES, self.TIME_CANON)

        # Clean day-of-week strings
        self._clean_categorical_column("DayOfWeek", self.DAY_ALIASES, self.DAY_CANON)

        # Clean charger type strings
        self._clean_categorical_column("ChargerType", self.CHARGER_ALIASES, self.CHARGER_CANON)

        # Clean user type strings
        self._clean_categorical_column("UserType", self.USER_ALIASES, self.USER_CANON)

        return self

    def print_uniques_clean(self):
        """Print unique values AFTER cleaning (to confirm standardization)."""
        self._ensure_df()
        print(self.df["VehicleModel"].unique())
        print(self.df["ChargingStationLocation"].unique())
        print(self.df["TimeOfDay"].unique())
        print(self.df["DayOfWeek"].unique())
        print(self.df["ChargerType"].unique())
        print(self.df["UserType"].unique())
        return self

    def handle_missing_and_dropna(self):
        """
        Handle missing values like the original code:
        - print missing counts
        - count rows where ALL key categorical and datetime columns are missing
        - print the hard-coded missing percentage from the original
        - drop rows containing any NaN
        """
        self._ensure_df()

        # Print missing counts for each column
        print(self.df.isna().sum())

        # Find rows where these columns are all missing simultaneously
        is_ALL_na = self.df[
            self.df[
                [
                    "VehicleModel",
                    "ChargerType",
                    "ChargingStationLocation",
                    "TimeOfDay",
                    "DayOfWeek",
                    "UserType",
                    "ChargingStartTime",
                    "ChargingEndTime",
                ]
            ]
            .isna()
            .all(axis=1)
        ]

        # Print how many such fully-missing rows exist
        print(len(is_ALL_na))

        # Print same percentage as original code (to keep outputs consistent)
        print(150 / 1050 * 100)

        # Drop all rows with any missing values
        self.df = self.df.dropna()
        return self

    def process_datetime_columns(self):
        """
        Split ChargingStartTime and ChargingEndTime into separate Date and Time columns,
        convert to proper datetime types, drop old columns, and rename new columns.
        """
        self._ensure_df()

        # Split start timestamp into date part and time part
        self.df["ChargingStartDateOnly"] = self.df["ChargingStartTime"].str.split(" ").str[0]
        self.df["ChargingStartTimeOnly"] = self.df["ChargingStartTime"].str.split(" ").str[1]

        # Split end timestamp into date part and time part
        self.df["ChargingEndDateOnly"] = self.df["ChargingEndTime"].str.split(" ").str[0]
        self.df["ChargingEndTimeOnly"] = self.df["ChargingEndTime"].str.split(" ").str[1]

        # Remove fractional seconds from end time if present (e.g., "12:30:00.0" -> "12:30:00")
        self.df["ChargingEndTimeOnly"] = self.df["ChargingEndTimeOnly"].str.replace(
            r"\..*$", "", regex=True
        )

        # Convert date strings to date objects
        self.df["ChargingStartDateOnly"] = pd.to_datetime(self.df["ChargingStartDateOnly"]).dt.date
        self.df["ChargingEndDateOnly"] = pd.to_datetime(self.df["ChargingEndDateOnly"]).dt.date

        # Convert time strings to time objects with given format
        self.df["ChargingStartTimeOnly"] = pd.to_datetime(
            self.df["ChargingStartTimeOnly"], format="%H:%M:%S"
        ).dt.time

        self.df["ChargingEndTimeOnly"] = pd.to_datetime(
            self.df["ChargingEndTimeOnly"], format="%H:%M:%S"
        ).dt.time

        # Drop original combined datetime string columns
        self.df = self.df.drop(["ChargingStartTime", "ChargingEndTime"], axis=1)

        # Rename new columns to final desired names
        self.df = self.df.rename(
            columns={
                "ChargingStartDateOnly": "ChargingStartDate",
                "ChargingStartTimeOnly": "ChargingStartTime",
                "ChargingEndDateOnly": "ChargingEndDate",
                "ChargingEndTimeOnly": "ChargingEndTime",
            }
        )

        return self

    def clean_symbols_and_features(self):
        """
        Remove symbols and convert types:
        - Remove '%' from SoC columns and convert to float
        - Remove '$' from cost column and convert to float
        - Remove 'S' from station id and keep as string
        Then create:
        - ChargeDifference% = end - start
        - ChargingDurationMinutes = hours * 60
        Finally print data types (original behavior).
        """
        self._ensure_df()

        # Convert StateOfChargeStart% from strings like "20%" to float 20.0
        self.df["StateOfChargeStart%"] = (
            self.df["StateOfChargeStart%"].astype(str).str.replace("%", "").astype(float)
        )

        # Convert StateOfChargeEnd% from strings like "80%" to float 80.0
        self.df["StateOfChargeEnd%"] = (
            self.df["StateOfChargeEnd%"].astype(str).str.replace("%", "").astype(float)
        )

        # Convert ChargingCostUSD from strings like "$10.5" to float 10.5
        self.df["ChargingCostUSD"] = (
            self.df["ChargingCostUSD"].astype(str).str.replace("$", "").astype(float)
        )

        # Remove "S" from station ID (e.g., "S123" -> "123")
        self.df["ChargingStationId"] = self.df["ChargingStationId"].astype(str).str.replace("S", "")

        # Create charge difference feature
        self.df["ChargeDifference%"] = self.df["StateOfChargeEnd%"] - self.df["StateOfChargeStart%"]

        # Convert charging duration hours to minutes
        self.df["ChargingDurationMinutes"] = self.df["ChargingDurationHours"] * 60

        # Print dataframe dtypes (kept same as original output)
        print(self.df.dtypes)

        return self

    def check_duplicates(self):
        """Check duplicates and print how many duplicated rows exist."""
        self._ensure_df()

        # Select all duplicated rows (keep=False shows all duplicates)
        duplicated_rows_all = self.df[self.df.duplicated(keep=False)]

        # Print number of duplicated rows
        print(len(duplicated_rows_all))

        return self

    def check_outliers_iqr(self):
        """
        Check outliers using the 1.5*IQR rule for each numeric column,
        and print outlier counts (kept identical logic).
        """
        self._ensure_df()

        # Select numeric columns only
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns

        # Dictionary storing outlier counts per numeric column
        outliers_count = {}

        # Loop through each numeric column
        for column in numeric_columns:
            # Compute first quartile (25th percentile)
            Q1 = self.df[column].quantile(0.25)

            # Compute third quartile (75th percentile)
            Q3 = self.df[column].quantile(0.75)

            # Compute IQR
            IQR = Q3 - Q1

            # Define lower bound for outliers
            lower_bound = Q1 - 1.5 * IQR

            # Define upper bound for outliers
            upper_bound = Q3 + 1.5 * IQR

            # Select outliers
            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]

            # Store outlier count
            outliers_count[column] = len(outliers)

        # Print outlier counts for each numeric column
        for column, count in outliers_count.items():
            print(f"Outliers in {column}: {count}")

        return self

    def get_df(self) -> pd.DataFrame:
        """Return the processed dataframe."""
        self._ensure_df()
        return self.df

    def _ensure_df(self):
        """Internal guard to ensure df is loaded before operations."""
        if self.df is None:
            raise ValueError("DataFrame not loaded. Call .load() first.")
