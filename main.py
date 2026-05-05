import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandasql import sqldf

# ── Step 1: Load Data ──────────────────────────────────────────────────────────
df = pd.read_csv('titanic.csv', index_col=0)
print("Step 1 – First 5 rows:")
print(df.head(), "\n")

# ── Step 2: Slice by gender / age ─────────────────────────────────────────────
women_and_children_df = df[(df['Sex'] == 'female') | (df['Age'] <= 15)]
adult_males_df = df[(df['Sex'] == 'male') & (df['Age'] > 15)]

print(f"Step 2 – Women & children: {len(women_and_children_df)} rows | Adult males: {len(adult_males_df)} rows\n")

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 8))
adult_males_df['Survived'].hist(ax=axes[0])
axes[0].set_title('Male Adults (over 15)')
axes[0].set_xlabel('Survived?')
axes[0].set_ylabel('Count')
women_and_children_df['Survived'].hist(ax=axes[1], color='pink')
axes[1].set_title('Women and Children (15 and under)')
axes[1].set_xlabel('Survived?')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig('step2_gender_age_survival.png', dpi=100)
plt.close()

# ── Step 3: Slice by passenger class ──────────────────────────────────────────
first_class_df = df[df['Pclass'] == '1']
second_third_class_df = df[df['Pclass'] != '1']

print(f"Step 3 – First class: {len(first_class_df)} rows | 2nd/3rd class: {len(second_third_class_df)} rows\n")

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 8))
first_class_df['Survived'].hist(ax=axes[0], color='gold')
axes[0].set_title('First Class')
axes[0].set_xlabel('Survived?')
axes[0].set_ylabel('Count')
second_third_class_df['Survived'].hist(ax=axes[1], color='grey')
axes[1].set_title('Everyone Else')
axes[1].set_xlabel('Survived?')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig('step3_class_survival.png', dpi=100)
plt.close()

# ── Step 4: .query() – PassengerId >= 500 ─────────────────────────────────────
query_string = 'PassengerId >= 500'
high_passenger_number_df = df.query(query_string)

print(f"Step 4 – Passengers with ID >= 500: {len(high_passenger_number_df)} rows")
print(high_passenger_number_df.head(), "\n")

# ── Step 5: .query() – females and children under 15 ─────────────────────────
query_string = 'Sex == "female" or Age < 15'
female_children_df = df.query(query_string)

print(f"Step 5 – Females + children under 15: {len(female_children_df)} rows")
print(female_children_df.head(), "\n")

# ── Step 6: .eval() – create Age_x_Fare column ────────────────────────────────
df = df.eval('Age_x_Fare = Age * Fare')

print("Step 6 – DataFrame with Age_x_Fare column:")
print(df[['Name', 'Age', 'Fare', 'Age_x_Fare']].head(), "\n")

# ── Step 7: pandasql lambda ────────────────────────────────────────────────────
pysqldf = lambda q: sqldf(q, globals())

# ── Step 8: SQL – first 10 passenger names ────────────────────────────────────
query1 = "SELECT Name FROM df LIMIT 10"
passenger_names = pysqldf(query1)

print("Step 8 – First 10 passenger names:")
print(passenger_names, "\n")

# ── Step 9: SQL – surviving males (name + fare, first 30) ─────────────────────
query2 = "SELECT Name, Fare FROM df WHERE Sex='male' AND Survived=1 LIMIT 30"
sql_surviving_males = pysqldf(query2)

print(f"Step 9 – Surviving males (first 30): {len(sql_surviving_males)} rows")
print(sql_surviving_males.head(), "\n")

# ── Step 10: SQL GROUP BY – female survival by class ──────────────────────────
query3 = "SELECT Pclass, COUNT(*) as 'Count(*)' FROM df WHERE Sex='female' AND Survived=1 GROUP BY Pclass"
query4 = "SELECT Pclass, COUNT(*) as 'Count(*)' FROM df WHERE Sex='female' AND Survived=0 GROUP BY Pclass"

survived_females_by_pclass_df = pysqldf(query3)
died_females_by_pclass_df = pysqldf(query4)

print("Step 10 – Female survivors by class:")
print(survived_females_by_pclass_df)
print("\nStep 10 – Female casualties by class:")
print(died_females_by_pclass_df, "\n")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
survived_females_by_pclass_df.set_index('Pclass')['Count(*)'].plot(kind='barh', ax=axes[0])
axes[0].set_title('Distribution of Classes for Female Survivors')
died_females_by_pclass_df.set_index('Pclass')['Count(*)'].plot(kind='barh', ax=axes[1])
axes[1].set_title('Distribution of Classes for Female Casualties')
plt.tight_layout()
plt.savefig('step10_female_class_survival.png', dpi=100)
plt.close()

print("All steps complete. Charts saved as PNG files.")
