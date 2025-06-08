from functions2 import *

# ‣ “one group, many metrics”  (train vs eval case)
plot_curves(
    group_names = ["correctly disambiguated PCA 1 camera train eval same position but world vs camera coords"],
    y_keys      = ["rollout/Success", "eval/Success"],
    y_labels    = ["Train", "Eval"],
    key_x       = "_step",
    filename    = "disambiguated_pca_test_train_vs_eval_success.pdf"
)
