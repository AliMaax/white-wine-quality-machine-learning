import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('winequality-white.csv', sep=';')

# Create a Boxplot for Total Sulfur Dioxide vs Quality
plt.figure(figsize=(10, 6))
sns.boxplot(x='quality', y='total sulfur dioxide', data=df, palette='coolwarm')

# Titles and Labels
plt.title('Illustrative Figure 4: Total Sulfur Dioxide across Quality Ratings', fontsize=15)
plt.xlabel('Quality Score', fontsize=12)
plt.ylabel('Total Sulfur Dioxide (mg/dm³)', fontsize=12)

# Save for your report
plt.savefig('illustrative_figure_4.png', dpi=300)
plt.show()