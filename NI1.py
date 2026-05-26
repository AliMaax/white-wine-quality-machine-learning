import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('winequality-white.csv', sep=';')

# Create Non-Illustrative Figure 1
plt.figure(figsize=(10, 6))
sns.scatterplot(x='pH', y='quality', data=df, alpha=0.3, color='orange')

# Add titles and labels
plt.title('Non-Illustrative Figure 1: pH vs. Quality', fontsize=15)
plt.xlabel('pH Level', fontsize=12)
plt.ylabel('Quality Score', fontsize=12)

# Save the figure
plt.savefig('non_illustrative_1.png', dpi=300)
plt.show()