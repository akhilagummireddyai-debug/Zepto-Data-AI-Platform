import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD TITANIC DATASET ONCE
# ============================================================

print("=" * 60)
print("LOADING TITANIC DATASET")
print("=" * 60)

df = sns.load_dataset("titanic")

print("Dataset loaded successfully.")
print("Original shape:", df.shape)

# Save raw dataset immediately as offline fallback
df.to_csv("titanic.csv", index=False)

print("titanic.csv saved successfully.")


# ============================================================
# 2. DATA PROFILING
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFO")
print("=" * 60)

df.info()

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe(include="all"))

print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print(df.shape)


# ============================================================
# 3. MISSING VALUE PERCENTAGES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE PERCENTAGES")
print("=" * 60)

missing_percentage = (
    df.isnull().sum() / len(df)
) * 100

missing_percentage = missing_percentage[
    missing_percentage > 0
]

print(missing_percentage)


# ============================================================
# 4. MISSING VALUE HANDLING
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE HANDLING")
print("=" * 60)


# ---------------- AGE ----------------

age_missing = (
    df["age"].isnull().sum() / len(df)
) * 100

print(f"Age missing: {age_missing:.2f}%")

if age_missing < 5:

    df = df.dropna(subset=["age"])

    print(
        "Age: rows dropped because missing percentage "
        "is below 5%."
    )

elif age_missing <= 30:

    df["age"] = df["age"].fillna(
        df["age"].median()
    )

    print(
        "Age: median imputation because missing "
        "percentage is between 5% and 30%."
    )

else:

    df["age"] = df["age"].fillna(
        df["age"].median()
    )

    print(
        "Age: median imputation used because "
        "missing percentage is high."
    )


# ---------------- EMBARKED ----------------

embarked_missing = (
    df["embarked"].isnull().sum() / len(df)
) * 100

print(
    f"Embarked missing: "
    f"{embarked_missing:.2f}%"
)

if embarked_missing < 5:

    df = df.dropna(subset=["embarked"])

    print(
        "Embarked: rows dropped because missing "
        "percentage is below 5%."
    )

elif embarked_missing <= 30:

    df["embarked"] = df["embarked"].fillna(
        df["embarked"].mode()[0]
    )

    print(
        "Embarked: mode imputation because missing "
        "percentage is between 5% and 30%."
    )


# ---------------- DECK ----------------

deck_missing = (
    df["deck"].isnull().sum() / len(df)
) * 100

print(
    f"Deck missing: "
    f"{deck_missing:.2f}%"
)

if deck_missing > 30:

    # IMPORTANT:
    # deck is categorical in Seaborn dataset.
    # Convert to object before adding "Missing".
    df["deck"] = df["deck"].astype(object)

    df["deck"] = df["deck"].fillna(
        "Missing"
    )

    print(
        "Deck: missing percentage is above 30%, "
        "so missing values are encoded as "
        "a separate 'Missing' category."
    )


# ---------------- EMBARK TOWN ----------------

if "embark_town" in df.columns:

    embark_town_missing = (
        df["embark_town"].isnull().sum()
        / len(df)
    ) * 100

    print(
        f"Embark_town missing: "
        f"{embark_town_missing:.2f}%"
    )

    if embark_town_missing > 0:

        df["embark_town"] = (
            df["embark_town"]
            .astype(object)
            .fillna(
                df["embark_town"]
                .mode()[0]
            )
        )


print("\nMissing values after cleaning:")

print(df.isnull().sum())


# ============================================================
# CREATE CHART FOLDER
# ============================================================

os.makedirs("charts", exist_ok=True)


# ============================================================
# 5. AGE HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["age"],
    bins=30,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "charts/age_histogram.png"
)

plt.show()


# ============================================================
# AGE BOX PLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["age"]
)

plt.title("Age Box Plot")
plt.xlabel("Age")

plt.tight_layout()

plt.savefig(
    "charts/age_boxplot.png"
)

plt.show()


# ============================================================
# AGE IQR OUTLIERS
# ============================================================

Q1_age = df["age"].quantile(0.25)
Q3_age = df["age"].quantile(0.75)

IQR_age = Q3_age - Q1_age

lower_age = Q1_age - 1.5 * IQR_age
upper_age = Q3_age + 1.5 * IQR_age

age_outliers = df[
    (df["age"] < lower_age) |
    (df["age"] > upper_age)
]

print("\nAge IQR outliers:", len(age_outliers))


# ============================================================
# 6. FARE HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["fare"],
    bins=30,
    kde=True
)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "charts/fare_histogram.png"
)

plt.show()


# ============================================================
# FARE BOX PLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["fare"]
)

plt.title("Fare Box Plot")
plt.xlabel("Fare")

plt.tight_layout()

plt.savefig(
    "charts/fare_boxplot.png"
)

plt.show()


# ============================================================
# FARE IQR OUTLIERS
# ============================================================

Q1_fare = df["fare"].quantile(0.25)
Q3_fare = df["fare"].quantile(0.75)

IQR_fare = Q3_fare - Q1_fare

lower_fare = Q1_fare - 1.5 * IQR_fare
upper_fare = Q3_fare + 1.5 * IQR_fare

fare_outliers = df[
    (df["fare"] < lower_fare) |
    (df["fare"] > upper_fare)
]

print(
    "Fare IQR outliers:",
    len(fare_outliers)
)


# ============================================================
# FARE MEAN, MEDIAN, MODE
# ============================================================

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

print("\n" + "=" * 60)
print("FARE STATISTICS")
print("=" * 60)

print("Mean:", fare_mean)
print("Median:", fare_median)
print("Mode:", fare_mode)


if fare_mean > fare_median > fare_mode:

    print(
        "Fare distribution is right-skewed "
        "because Mean > Median > Mode."
    )

elif fare_mean < fare_median < fare_mode:

    print(
        "Fare distribution is left-skewed "
        "because Mean < Median < Mode."
    )

else:

    print(
        "Fare distribution is approximately "
        "symmetric based on mean, median and mode."
    )


# ============================================================
# 7. SURVIVAL RATE BY SEX
# ============================================================

print("\n" + "=" * 60)
print("SURVIVAL RATE BY SEX")
print("=" * 60)

male_rate = df[
    df["sex"] == "male"
]["survived"].mean()

female_rate = df[
    df["sex"] == "female"
]["survived"].mean()

print(
    f"Male survival rate: "
    f"{male_rate:.4f}"
)

print(
    f"Female survival rate: "
    f"{female_rate:.4f}"
)


# ============================================================
# SURVIVAL RATE BY PCLASS
# ============================================================

print("\n" + "=" * 60)
print("SURVIVAL RATE BY PCLASS")
print("=" * 60)

for pclass in sorted(
    df["pclass"].unique()
):

    rate = df[
        df["pclass"] == pclass
    ]["survived"].mean()

    print(
        f"Class {pclass}: "
        f"{rate:.4f}"
    )


# ============================================================
# SURVIVAL RATE BY SEX + PCLASS
# ============================================================

print("\n" + "=" * 60)
print("SURVIVAL RATE BY SEX AND PCLASS")
print("=" * 60)

for sex in df["sex"].unique():

    for pclass in sorted(
        df["pclass"].unique()
    ):

        rate = df[
            (df["sex"] == sex) &
            (df["pclass"] == pclass)
        ]["survived"].mean()

        print(
            f"{sex}, Class {pclass}: "
            f"{rate:.4f}"
        )


# ============================================================
# 8. CORRELATION MATRIX
# EXACTLY SIX REQUIRED COLUMNS
# ============================================================

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = df[
    correlation_columns
].corr()

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

print(correlation_matrix)


# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Titanic Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    "charts/correlation_heatmap.png"
)

plt.show()


# ============================================================
# TWO STRONGEST CORRELATIONS
# ============================================================

corr_pairs = []

for i in range(
    len(correlation_columns)
):

    for j in range(
        i + 1,
        len(correlation_columns)
    ):

        col1 = correlation_columns[i]
        col2 = correlation_columns[j]

        value = correlation_matrix.loc[
            col1,
            col2
        ]

        corr_pairs.append(
            (
                col1,
                col2,
                value,
                abs(value)
            )
        )


corr_pairs.sort(
    key=lambda x: x[3],
    reverse=True
)

print("\n" + "=" * 60)
print("TWO STRONGEST CORRELATIONS")
print("=" * 60)

for pair in corr_pairs[:2]:

    print(
        f"{pair[0]} vs {pair[1]}: "
        f"{pair[2]:.4f}"
    )


# ============================================================
# 9. MULTIVARIATE CHART 1
# SURVIVAL BY SEX
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="sex",
    y="survived"
)

plt.title(
    "Survival Rate by Sex"
)

plt.ylabel("Survival Rate")

plt.tight_layout()

plt.savefig(
    "charts/survival_by_sex.png"
)

plt.show()


# ============================================================
# MULTIVARIATE CHART 2
# SURVIVAL BY CLASS
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived"
)

plt.title(
    "Survival Rate by Passenger Class"
)

plt.ylabel("Survival Rate")

plt.tight_layout()

plt.savefig(
    "charts/survival_by_class.png"
)

plt.show()


# ============================================================
# MULTIVARIATE CHART 3
# SEX + CLASS
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title(
    "Survival Rate by Sex and Class"
)

plt.ylabel("Survival Rate")

plt.tight_layout()

plt.savefig(
    "charts/survival_sex_class.png"
)

plt.show()


# ============================================================
# MULTIVARIATE CHART 4
# AGE + FARE + SURVIVAL
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived"
)

plt.title(
    "Age vs Fare by Survival"
)

plt.tight_layout()

plt.savefig(
    "charts/age_fare_survival.png"
)

plt.show()


# ============================================================
# MULTIVARIATE CHART 5
# FARE + CLASS + SURVIVAL
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="pclass",
    y="fare",
    hue="survived"
)

plt.title(
    "Fare Distribution by Class and Survival"
)

plt.tight_layout()

plt.savefig(
    "charts/fare_class_survival.png"
)

plt.show()


# ============================================================
# 10. EXPLORATORY STANDARDIZATION
# ============================================================

print("\n" + "=" * 60)
print("STANDARDIZATION CHECK")
print("=" * 60)

# Before standardization

print("\nBefore standardization:")

print(
    "Age mean:",
    df["age"].mean()
)

print(
    "Age std:",
    df["age"].std()
)

print(
    "Fare mean:",
    df["fare"].mean()
)

print(
    "Fare std:",
    df["fare"].std()
)


# Z-score standardization

df["age_z"] = (
    df["age"] - df["age"].mean()
) / df["age"].std()

df["fare_z"] = (
    df["fare"] - df["fare"].mean()
) / df["fare"].std()


# After standardization

print("\nAfter standardization:")

print(
    "Age z-score mean:",
    df["age_z"].mean()
)

print(
    "Age z-score std:",
    df["age_z"].std()
)

print(
    "Fare z-score mean:",
    df["fare_z"].mean()
)

print(
    "Fare z-score std:",
    df["fare_z"].std()
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "Cleaned rows:",
    len(df)
)

print(
    "Offline dataset:",
    "analytics/titanic.csv"
)

print(
    "Charts folder:",
    "analytics/charts/"
)

print(
    "\nPart A completed successfully."
)