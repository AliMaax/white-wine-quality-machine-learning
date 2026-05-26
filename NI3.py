import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('winequality-white.csv', sep=';')

# Create Non-Illustrative Figure 3
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sulphates', y='quality', data=df, alpha=0.3, color='red')

# Add titles and labels
plt.title('Non-Illustrative Figure 3: Sulphates vs. Quality', fontsize=15)
plt.xlabel('Sulphates (g/dm³)', fontsize=12)
plt.ylabel('Quality Score', fontsize=12)

# Save the figure
plt.savefig('non_illustrative_3.png', dpi=300)
plt.show()