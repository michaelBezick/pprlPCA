from functions2 import *

group_names= ["PCA_DS_easy","baseline_DS_easier"]

# ‣ “one group, many metrics”  (train vs eval case)
plot_curves_binned_bootstrap(
    group_names = group_names,
    y_keys      = ["eval/roll-30deg/success_rate"],
    y_labels    = ["Eval Success"],
    key_x       = "_step",
    filename    = "./DS/+30_degree_roll.pdf",
    color_cycle=False,
    # cutoff = 4000,
    title = "+30 Degree Roll",
)

plot_curves_binned_bootstrap(
    group_names = group_names,
    y_keys      = ["eval/roll+15deg/success_rate"],
    y_labels    = ["Eval Success"],
    key_x       = "_step",
    filename    = "./DS/+15_degree_roll.pdf",
    color_cycle=False,
    # cutoff = 4000,
    title = "+15 Degree Roll",
)

plot_curves_binned_bootstrap(
    group_names = group_names,
    y_keys      = ["eval/shift+z+150/success_rate"],
    y_labels    = ["Eval Success"],
    key_x       = "_step",
    filename    = "./DS/+z50.pdf",
    color_cycle=False,
    # cutoff = 4000,
    title = "Shift 50mm +z",
)

plot_curves_binned_bootstrap(
    group_names = group_names,
    y_keys      = ["eval/shift+y+150/success_rate"],
    y_labels    = ["Eval Success"],
    key_x       = "_step",
    filename    = "./DS/+y50.pdf",
    color_cycle=False,
    # cutoff = 4000,
    title = "Shift 50mm +y",
)

plot_curves_binned_bootstrap(
    group_names = group_names,
    y_keys      = ["eval/shift+x+150/success_rate"],
    y_labels    = ["Eval Success"],
    key_x       = "_step",
    filename    = "./DS/+x50.pdf",
    color_cycle=False,
    # cutoff = 4000,
    title = "Shift 50mm +x",
)

plot_curves_binned_bootstrap(
    group_names = group_names,
    y_keys      = ["eval/along_view+150/success_rate"],
    y_labels    = ["Eval Success"],
    key_x       = "_step",
    filename    = "./DS/along_view+50.pdf",
    color_cycle=False,
    # cutoff = 4000,
    title = "Shift Backwards on Camera Axis 50mm",
)
