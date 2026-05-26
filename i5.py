import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('winequality-white.csv', sep=';')

# Create Illustrative Figure 5: Box plot of Residual Sugar vs Quality
plt.figure(figsize=(10, 6))
sns.boxplot(x='quality', y='residual sugar', data=df, palette='YlGnBu')

# Titles and Labels
plt.title('Illustrative Figure 5: Residual Sugar across Quality Ratings', fontsize=15)
plt.xlabel('Quality Score', fontsize=12)
plt.ylabel('Residual Sugar (g/dm³)', fontsize=12)

# Cap the Y-axis slightly to make the main boxes readable (ignoring extreme 60+ outliers)
plt.ylim(0, 30) 
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.savefig('illustrative_figure_5.png', dpi=300)
plt.show()