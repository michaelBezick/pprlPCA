import collections
import itertools

import matplotlib.pyplot as plt
import numpy as np
import wandb


def get_runs_by_group(
    group_name, entity="michael-bezick-purdue-university", project="pprl"
):
    api = wandb.Api()
    return [run for run in api.runs(f"{entity}/{project}") if run.group == group_name]


def get_xy(run, key_x, key_y):
    """
    Return two aligned lists taken from a single run’s history.
    NOTE: key_x must be included in the history() call!
    """
    xs, ys = [], []
    # history() is lazy; ask for both columns up front
    for _, row in run.history(keys=[key_x, key_y]).iterrows():
        xs.append(row[key_x])
        ys.append(row[key_y])
    return xs, ys




def plot_curves(
    group_names,
    y_keys,
    key_x="_step",
    filename="plot.pdf",
    group_labels=None,
    y_labels=None,
    num_bootstrap=1_000,
    palette="tab10",
    color_cycle=True,
):
    """
    Parameters
    ----------
    group_names : list[str]          – one or more W&B run groups
    y_keys      : list[str]          – one or more metric names in run.history
    key_x       : str               – common x‑axis key (default: '_step')
    filename    : str               – output path
    group_labels: list[str] | None  – pretty labels for groups (len = len(group_names))
    y_labels    : list[str] | None  – pretty labels for metrics (len = len(y_keys))
    num_bootstrap : int             – repetitions for CI
    palette     : str | list        – colour cycle (matplotlib‑style)
    """
    assert group_names, "Give at least one group"
    assert y_keys, "Give at least one y‑axis key"

    if group_labels and len(group_labels) != len(group_names):
        raise ValueError("group_labels len must match group_names len")
    if y_labels and len(y_labels) != len(y_keys):
        raise ValueError("y_labels len must match y_keys len")

    colour_cycle = plt.get_cmap(palette).colors if isinstance(palette, str) else palette

    fig, ax = plt.subplots(figsize=(10, 6))

    # Outer loop: groups ▸ colours
    for g_idx, g in enumerate(group_names):
        runs = get_runs_by_group(g)
        if not runs:
            print(f"[warning] group “{g}” has no runs – skipping")
            continue

        base_color = colour_cycle[g_idx % len(colour_cycle)]
        # For visibility, each metric in the same group gets the same colour
        # but different line style.
        line_styles = itertools.cycle(["-", "--", "-.", ":"])

        # Inner loop: metrics ▸ linestyle
        for m_idx, key_y in enumerate(y_keys):
            style = next(line_styles)
            if color_cycle:
                base_color = colour_cycle[m_idx % len(colour_cycle)]

            # aggregate over runs ― identical to your existing logic
            buckets = collections.defaultdict(list)
            for run in runs:
                xs, ys = get_xy(run, key_x, key_y)
                for x, y in zip(xs, ys):
                    buckets[x].append(y)

            xs_sorted = sorted(buckets.keys())
            means, lowers, uppers = [], [], []
            for x in xs_sorted:
                ys = buckets[x]
                means.append(np.mean(ys))
                boot = np.random.choice(ys, (num_bootstrap, len(ys)), replace=True)
                boot_means = boot.mean(axis=1)
                lowers.append(np.percentile(boot_means, 2.5))
                uppers.append(np.percentile(boot_means, 97.5))

            xs_a = np.asarray(xs_sorted)
            means = np.asarray(means)
            lowers = np.asarray(lowers)
            uppers = np.asarray(uppers)

            label_g = group_labels[g_idx] if group_labels else g
            label_y = y_labels[m_idx] if y_labels else key_y
            full_label = (
                f"{label_g} – {label_y}"
                if len(group_names) > 1 or len(y_keys) > 1
                else label_y
            )

            ax.plot(xs_a, means, linestyle=style, color=base_color, label=full_label)
            ax.fill_between(xs_a, lowers, uppers, alpha=0.25, color=base_color)

    ax.set_xlabel(key_x)
    # Use generic ylabel when several metrics share the axis
    ax.set_ylabel("metric value" if len(y_keys) > 1 else y_keys[0])
    ax.set_title("WandB curves with 95 % bootstrap CI")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    print(f"✓ saved → {filename}")
