# Exploratory Data Analysis of EV Charging Data

This repository contains a Python-based data cleaning + exploratory data analysis (EDA) project focused on understanding **electric vehicle (EV) charging session patterns**. The project studies charging behavior across vehicle models, user types, time/day patterns, charger types, locations, and pricing. :contentReference[oaicite:7]{index=7}

---

## Project Summary

As EV adoption grows, understanding charging behavior is important for improving charging infrastructure planning and user experience. This project analyzes a dataset of EV charging sessions and produces visual insights using Python libraries such as **NumPy, Matplotlib, and Seaborn**. :contentReference[oaicite:8]{index=8}

The workflow includes:
- Data tidying & cleaning (string standardization, missing values, datetime processing, type conversions)
- Exploratory data analysis with descriptive stats
- Visualization-driven answers to **7 research questions** :contentReference[oaicite:9]{index=9}

---

## Dataset

Each row represents a **unique EV charging session**. The report states the dataset contains **900 observations** after cleaning. :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11}

The variables include:
- Vehicle and user attributes (vehicle model, battery capacity, vehicle age, user type, etc.)
- Charging session details (start/end time, duration, energy consumed, charging rate, charging cost)
- Context variables (location, time of day, day of week, temperature) :contentReference[oaicite:12]{index=12}

A detailed variable list + measurement scales is provided in the report. :contentReference[oaicite:13]{index=13}

---

## Repository Structure

Suggested structure (matches the modular/OOP refactor):

```text
.
├── data_processing.py        # Data loading + cleaning pipeline (OOP)
├── visualization.py          # All plots (kept identical logic/output)
├── main.py                   # Runs the full pipeline and generates visuals
├── data/
│   └── EV_Charging_Patterns_Dirty.csv
└── report/
    └── final templatee 11.pdf
