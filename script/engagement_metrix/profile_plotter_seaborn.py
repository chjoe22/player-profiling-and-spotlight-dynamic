import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_barplots(parent_folder):
    # Loop through all subfolders and files
    for root, dirs, files in os.walk(parent_folder):
        for file in files:

            # Only process CSV files ending with "profile.csv"
            if file.lower().endswith("profile.csv"):

                csv_path = os.path.join(root, file)

                try:
                    # Read CSV
                    df = pd.read_csv(csv_path)

                    # Use first row
                    row = df.iloc[0]

                    # Convert data for plotting
                    plot_df = pd.DataFrame({
                        "Category": row.index,
                        "Value": row.values
                    })

                    # Create plot
                    plt.figure(figsize=(4, 5))

                    ax = sns.barplot(
                        data=plot_df,
                        x="Category",
                        y="Value"
                    )

                    # Keep 0 centered
                    max_abs = max(abs(plot_df["Value"].min()), abs(plot_df["Value"].max()))
                    ax.set_ylim(-max_abs, max_abs)

                    # Draw zero line
                    ax.axhline(0, color="black", linewidth=1)

                    # Add value labels on bars
                    for container in ax.containers:
                        ax.bar_label(
                            container,
                            fmt="%.2f",   # Number format
                            padding=3
                        )

                    limit = max_abs * 1.25
                    ax.set_ylim(-limit, limit)

                    # Plot title = filename without extension
                    folder_name = os.path.basename(root)
                    base_name = os.path.splitext(file)[0]

                    plt.title(f"{folder_name} - {base_name}")

                    plt.tight_layout()

                    # Save PNG in same folder
                    folder_name = os.path.basename(root)
                    base_name = os.path.splitext(file)[0]
                    png_filename = f"{folder_name}_{base_name}.png"
                    png_path = os.path.join(root, png_filename)

                    plt.savefig(png_path)
                    plt.close()

                    print(f"Saved: {png_path}")

                except Exception as e:
                    print(f"Failed processing {csv_path}: {e}")


if __name__ == "__main__":
    # Replace with your parent folder path
    parent_folder = "../../resources/speaker_stats"

    create_barplots(parent_folder)