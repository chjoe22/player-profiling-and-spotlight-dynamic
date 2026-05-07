import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def make_emotion_pie_charts(input_folder: str, output_folder: str):
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("pastel")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if "_weighted" not in file_name or not file_name.endswith(".csv"):
                continue

            csv_path = os.path.join(root, file_name)
            episode_label = file_name.split('_')[0].lower()
            episode_output_dir = os.path.join(output_folder, f"episode_{episode_label}")
            
            if not os.path.exists(episode_output_dir):
                os.makedirs(episode_output_dir)

            try:
                df = pd.read_csv(csv_path, header=0)
                person_col = df.columns[0]
                emotion_col = df.columns[1]
            except Exception as e:
                print(f"Error: {e}")
                continue

            group_data = df[emotion_col].value_counts()
            save_pie(group_data, f"Episode {episode_label} - Group Summary",
                    os.path.join(episode_output_dir, "total_group_emotions.png"), colors)

            for person in df[person_col].unique():
                person_data = df[df[person_col] == person][emotion_col].value_counts()
                safe_name = "".join(x for x in str(person).lower() if x.isalnum())
                
                save_pie(person_data, f"Episode {episode_label} - {person}",
                        os.path.join(episode_output_dir, f"{safe_name}_emotions.png"), colors)

def save_pie(data, title, output_path, colors):
    data = data[data > 0]
    
    if data.empty:
        return

    total = data.sum()
    limit = total * 0.01 
    small_slices = data[data < limit]
    if len(small_slices) > 1:
        data = data[data >= limit]
        data['other'] = small_slices.sum()

    plt.figure(figsize=(12, 7))
    
    patches, texts, autotexts = plt.pie(
        data, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        pctdistance=0.85,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )

    plt.legend(
        patches, 
        data.index, 
        title="Emotions", 
        loc="center left", 
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    input = "../../resources/results/combined" 
    output = "../../resources/emotion_output/" 
    make_emotion_pie_charts(input, output)