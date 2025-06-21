import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV and setup
df = pd.read_csv('math500_bestmodels.csv')
df2 = pd.read_csv('math500_best.csv')
plt.figure(figsize=(90, 60))
plt.rcParams['axes.linewidth'] = 3  # Increased line width

# Create color palette
model_prompt_pairs = df.groupby(['Model', 'Prompt_Type'])
colors = sns.color_palette('Set2', n_colors=len(model_prompt_pairs))

# Plot continuous lines with markers
for (model, prompt_type), color in zip(model_prompt_pairs.groups.keys(), colors):
    data = df[(df['Model'] == model) & (df['Prompt_Type'] == prompt_type)]
    plt.plot(data['Latency'], 
            data['Accuracy'], 
            label=f"{model.split('/')[-1]} + {prompt_type.split('-')[-1]}",
            color=color,
            linewidth=45,  # Thicker lines
            marker='o',
            markersize=25,  # Larger markers
            markeredgewidth=1,
            markeredgecolor='white')

plt.plot(df2['Latency'], 
        df2['Accuracy'], 
        label="Best Combination",
        color='red',
        linewidth=40,  # Thicker lines
        marker='o',
        markersize=32,  # Larger markers
        markeredgewidth=1,
        markeredgecolor='white')

# Enhanced customization with larger fonts
plt.xlabel('Latency (seconds)', fontsize=250, weight='bold')
plt.ylabel('Accuracy (%)', fontsize=250, weight='bold')
plt.title('Model Performance: Latency vs Accuracy', fontsize=250, weight='bold', pad=20)

# Remove grid and adjust ticks
plt.grid(False)  # Remove grid
plt.minorticks_on()
plt.tick_params(axis='both', which='major', labelsize=210, length=8, width=3)
plt.tick_params(axis='both', which='minor', length=25, width=2)

# Move legend inside
plt.legend(loc='lower right', fontsize=190, frameon=True, fancybox=True, shadow=True, ncol=1)

# Set spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('latency_accuracy_plot.pdf', 
            bbox_inches='tight',
            dpi=2400,
            facecolor='white',
            edgecolor='none')
plt.close()