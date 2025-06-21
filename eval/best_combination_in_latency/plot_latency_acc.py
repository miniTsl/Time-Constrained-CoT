import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV and setup
df = pd.read_csv('math500_best.csv')
plt.figure(figsize=(15, 10))
plt.rcParams['axes.linewidth'] = 2

# Create color palette
model_prompt_pairs = df.groupby(['Model', 'Prompt_Type'])
colors = sns.color_palette('Set2', n_colors=len(model_prompt_pairs))

# Plot continuous lines with markers
for (model, prompt_type), color in zip(model_prompt_pairs.groups.keys(), colors):
    data = df[(df['Model'] == model) & (df['Prompt_Type'] == prompt_type)]
    plt.plot(data['Latency'], 
            data['Accuracy'], 
            label=f"{model.split('/')[-1]}-{prompt_type}",
            color=color,
            linewidth=5,
            marker='o',
            markersize=3,  # Reduced marker size
            markeredgewidth=0.5,  # Thinner marker edge
            markeredgecolor='white')
    
# Enhanced customization
plt.xlabel('Latency (seconds)', fontsize=14, weight='bold')
plt.ylabel('Accuracy (%)', fontsize=14, weight='bold')
plt.title('Model Performance: Latency vs Accuracy', fontsize=16, weight='bold',pad=20)

# Improve grid and ticks
plt.grid(True, linestyle='--', alpha=0.7, which='both')
plt.minorticks_on()
plt.tick_params(axis='both', which='major', labelsize=12, length=6, width=2)
plt.tick_params(axis='both', which='minor', length=3, width=1)

# Enhanced legend
plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left',borderaxespad=0.,fontsize=12,frameon=True,fancybox=True,shadow=True,ncol=1)

# Set spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# Adjust layout and save
plt.tight_layout()
plt.savefig('latency_accuracy_plot.png', 
            bbox_inches='tight',
            dpi=300,
            facecolor='white',
            edgecolor='none')
plt.close()