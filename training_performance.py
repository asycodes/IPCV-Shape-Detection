import matplotlib.pyplot as plt
import os

stages = [0, 1, 2]
tpr = [1, 1, 1]
fpr = [1, 0.0164799, 0.000523583]
output_folder = "graphs"

plt.figure(figsize=(10, 6))
plt.plot(stages, tpr, marker="o", label="True Positive Rate (TPR)", color="blue")
plt.plot(stages, fpr, marker="s", label="False Positive Rate (FPR)", color="orange")
plt.title("Training Performance: TPR and FPR Across Stages", fontsize=14)
plt.xlabel("Training Stage", fontsize=12)
plt.ylabel("Rate", fontsize=12)
plt.xticks(stages)
plt.legend()
plt.grid()
output_path = os.path.join(output_folder, "training_performance.png")
plt.savefig(output_path, format="png")
print(f"Graph saved to {output_path}")
plt.show()
