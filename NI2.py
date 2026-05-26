import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('winequality-white.csv', sep=';')

# Create Non-Illustrative Figure 2
plt.figure(figsize=(10, 6))
sns.scatterplot(x='citric acid', y='quality', data=df, alpha=0.3, color='green')

# Add titles and labels
plt.title('Non-Illustrative Figure 2: Citric Acid vs. Quality', fontsize=15)
plt.xlabel('Citric Acid (g/dm³)', fontsize=12)
plt.ylabel('Quality Score', fontsize=12)

# Save the figure
plt.savefig('non_illustrative_2.png', dpi=300)
plt.show()