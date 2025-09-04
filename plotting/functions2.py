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

import collections
import itertools
import numpy as np
import matplotlib.pyplot as plt

def _bin_key(x, bin_width):
    return int(np.floor(float(x) / bin_width)) * bin_width

def plot_curves_binned_bootstrap(
    group_names,
    y_keys,
    key_x="_step",
    filename="plot_binned_bootstrap.pdf",
    group_labels=None,
    y_labels=None,
    palette="tab10",
    color_cycle=True,
    cutoff=None,
    title=None,
    bin_width=400,
    num_bootstrap=500,
    random_state=None,   # set int for reproducibility
):
    """
    Bins by `_step` in fixed-width bins, pools all raw values across runs in each bin,
    and computes bootstrap CIs of the mean per bin (2.5/97.5 percentiles), matching
    your original bootstrap approach but with binning.
    """
    assert group_names, "Give at least one group"
    assert y_keys, "Give at least one y-axis key"
    if group_labels and len(group_labels) != len(group_names):
        raise ValueError("group_labels len must match group_names len")
    if y_labels and len(y_labels) != len(y_keys):
        raise ValueError("y_labels len must match y_keys len")

    if random_state is not None:
        rng = np.random.default_rng(random_state)
        choice = rng.choice
    else:
        choice = np.random.choice

    colour_cycle = plt.get_cmap(palette).colors if isinstance(palette, str) else palette
    fig, ax = plt.subplots(figsize=(10, 6))

    for g_idx, g in enumerate(group_names):
        runs = get_runs_by_group(g)
        if not runs:
            print(f"[warning] group “{g}” has no runs – skipping")
            continue

        base_color = colour_cycle[g_idx % len(colour_cycle)]
        line_styles = itertools.cycle(["-", "--", "-.", ":"])

        for m_idx, key_y in enumerate(y_keys):
            style = next(line_styles)
            if color_cycle:
                base_color = colour_cycle[m_idx % len(colour_cycle)]

            # Collect raw values per bin across all runs
            bin_to_vals = collections.defaultdict(list)
            for run in runs:
                xs, ys = get_xy(run, key_x, key_y)
                for x, y in zip(xs, ys):
                    if y is None:
                        continue
                    try:
                        xv = float(x); yv = float(y)
                    except Exception:
                        continue
                    if cutoff is not None and xv > cutoff:
                        continue
                    b = _bin_key(xv, bin_width)
                    bin_to_vals[b].append(yv)

            if not bin_to_vals:
                print(f"[warning] metric “{key_y}” empty for group “{g}” – skipping")
                continue

            bins_sorted = sorted(bin_to_vals.keys())
            x_centers = np.array([b + bin_width / 2.0 for b in bins_sorted], dtype=float)

            means, lowers, uppers = [], [], []
            for b in bins_sorted:
                vals = np.asarray(bin_to_vals[b], dtype=float)
                if len(vals) == 0 or not np.isfinite(vals).any():
                    means.append(np.nan); lowers.append(np.nan); uppers.append(np.nan); continue
                means.append(float(np.mean(vals)))

                # Bootstrap mean per bin — same style as your original (pooled resampling)
                boots = choice(vals, size=(num_bootstrap, len(vals)), replace=True)
                boot_means = boots.mean(axis=1)
                lowers.append(float(np.percentile(boot_means, 2.5)))
                uppers.append(float(np.percentile(boot_means, 97.5)))

            means = np.asarray(means, dtype=float)
            lowers = np.asarray(lowers, dtype=float)
            uppers = np.asarray(uppers, dtype=float)

            label_g = group_labels[g_idx] if group_labels else g
            label_y = y_labels[m_idx] if y_labels else key_y
            full_label = (
                f"{label_g} – {label_y}"
                if (len(group_names) > 1 or len(y_keys) > 1)
                else label_y
            )

            ax.plot(x_centers, means, linestyle=style, color=base_color, label=full_label)
            ax.fill_between(x_centers, lowers, uppers, alpha=0.25, color=base_color)

    ax.set_xlabel(f"{key_x} (binned, width={bin_width})")
    ax.set_ylabel("metric value" if len(y_keys) > 1 else y_keys[0])
    ax.set_title(title or "W&B curves – bootstrapped mean per bin")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    print(f"✓ saved → {filename}")


import collections
import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_curves_binned(
    group_names,
    y_keys,
    key_x="_step",
    filename="plot_binned.pdf",
    group_labels=None,
    y_labels=None,
    palette="tab10",
    color_cycle=True,
    cutoff=None,
    title=None,
    bin_width=200,
    show_ci=True,          # 95% CI from per-run means; off by default
    ci_method="stderr",     # "stderr" (normal approx) or "percentile"
):
    """
    Plot per-bin mean across runs using _step bins of width `bin_width`.
    Aggregation is two-level:
      (i) per-run mean within each bin, then
      (ii) mean of those per-run means across runs.

    This avoids overweighting runs with denser logging.
    """
    assert group_names, "Give at least one group"
    assert y_keys, "Give at least one y-axis key"
    if group_labels and len(group_labels) != len(group_names):
        raise ValueError("group_labels len must match group_names len")
    if y_labels and len(y_labels) != len(y_keys):
        raise ValueError("y_labels len must match y_keys len")

    colour_cycle = plt.get_cmap(palette).colors if isinstance(palette, str) else palette
    fig, ax = plt.subplots(figsize=(10, 6))

    for g_idx, g in enumerate(group_names):
        runs = get_runs_by_group(g)
        if not runs:
            print(f"[warning] group “{g}” has no runs – skipping")
            continue

        base_color = colour_cycle[g_idx % len(colour_cycle)]
        line_styles = itertools.cycle(["-", "--", "-.", ":"])

        for m_idx, key_y in enumerate(y_keys):
            style = next(line_styles)
            if color_cycle:
                base_color = colour_cycle[m_idx % len(colour_cycle)]

            # 1) Per-run binning → per-run bin means
            run_bin_means = []     # list of dicts: {bin_start: mean_value}
            all_bins = set()
            for run in runs:
                xs, ys = get_xy(run, key_x, key_y)
                # collect raw values per bin for this run
                per_bin_vals = collections.defaultdict(list)
                for x, y in zip(xs, ys):
                    if y is None: 
                        continue
                    try:
                        x = float(x); y = float(y)
                    except Exception:
                        continue
                    if cutoff is not None and x > cutoff:
                        continue
                    b = _bin_key(x, bin_width)
                    per_bin_vals[b].append(y)
                # collapse to per-run mean in each bin
                per_run_means = {b: float(np.mean(v)) for b, v in per_bin_vals.items() if len(v) > 0}
                run_bin_means.append(per_run_means)
                all_bins.update(per_run_means.keys())

            if not all_bins:
                print(f"[warning] metric “{key_y}” empty for group “{g}” – skipping")
                continue

            # 2) Across-run aggregation per bin
            bins_sorted = sorted(all_bins)
            x_centers = np.array([b + bin_width/2.0 for b in bins_sorted], dtype=float)

            per_bin_across_run_vals = []
            for b in bins_sorted:
                vals = [rbm[b] for rbm in run_bin_means if b in rbm]  # per-run means available for this bin
                per_bin_across_run_vals.append(vals)

            means = np.array([np.mean(v) if len(v) else np.nan for v in per_bin_across_run_vals], dtype=float)

            # Optional CI band based on across-run variation
            lowers = uppers = None
            if show_ci:
                lowers, uppers = [], []
                for vals in per_bin_across_run_vals:
                    n = len(vals)
                    if n >= 2:
                        v = np.array(vals, float)
                        if ci_method == "percentile":
                            lowers.append(np.percentile(v, 2.5))
                            uppers.append(np.percentile(v, 97.5))
                        else:
                            # normal approx on mean: mean ± 1.96 * s/√n
                            m = v.mean(); s = v.std(ddof=1)
                            delta = 1.96 * (s / np.sqrt(n))
                            lowers.append(m - delta); uppers.append(m + delta)
                    elif n == 1:
                        lowers.append(vals[0]); uppers.append(vals[0])
                    else:
                        lowers.append(np.nan); uppers.append(np.nan)
                lowers = np.array(lowers, float); uppers = np.array(uppers, float)

            label_g = group_labels[g_idx] if group_labels else g
            label_y = y_labels[m_idx] if y_labels else key_y
            full_label = f"{label_g} – {label_y}" if (len(group_names) > 1 or len(y_keys) > 1) else label_y

            ax.plot(x_centers, means, linestyle=style, color=base_color, label=full_label)
            if show_ci:
                ax.fill_between(x_centers, lowers, uppers, alpha=0.25, color=base_color)

    ax.set_xlabel(f"{key_x} (binned, width={bin_width})")
    ax.set_ylabel("metric value" if len(y_keys) > 1 else y_keys[0])
    ax.set_title(title or "W&B curves – per-run means, binned by _step")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    print(f"✓ saved → {filename}")

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
    cutoff = None,
    title = None,
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
        print(g, runs)
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
                    if (cutoff):
                        if (x > cutoff):
                            continue
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
    if title:
        ax.set_title(title)
    else:
        ax.set_title("WandB curves with 95 % bootstrap CI")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    print(f"✓ saved → {filename}")

def plot_curves_ema(
    group_names,
    y_keys,
    key_x="_step",
    filename="plot_ema.pdf",
    group_labels=None,
    y_labels=None,
    palette="tab10",
    color_cycle=True,
    cutoff=None,
    title=None,
    ema_alpha=0.2,           # 0<alpha<=1; higher = more smoothing weight on recent points
    plot_raw_mean=True,     # set True to also draw the unsmoothed mean
):
    """
    Plot per-step mean across runs, with optional exponential moving average (EMA) smoothing.
    """
    assert group_names, "Give at least one group"
    assert y_keys, "Give at least one y-axis key"

    if group_labels and len(group_labels) != len(group_names):
        raise ValueError("group_labels len must match group_names len")
    if y_labels and len(y_labels) != len(y_keys):
        raise ValueError("y_labels len must match y_keys len")

    colour_cycle = plt.get_cmap(palette).colors if isinstance(palette, str) else palette

    def ema(arr, alpha):
        if len(arr) == 0:
            return np.array([], dtype=float)
        out = np.empty(len(arr), dtype=float)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    fig, ax = plt.subplots(figsize=(10, 6))

    for g_idx, g in enumerate(group_names):
        runs = get_runs_by_group(g)
        print(g, runs)
        if not runs:
            print(f"[warning] group “{g}” has no runs – skipping")
            continue

        base_color = colour_cycle[g_idx % len(colour_cycle)]
        line_styles = itertools.cycle(["-", "--", "-.", ":"])

        for m_idx, key_y in enumerate(y_keys):
            style = next(line_styles)
            if color_cycle:
                base_color = colour_cycle[m_idx % len(colour_cycle)]

            # Collect all runs into buckets keyed by x
            buckets = collections.defaultdict(list)
            for run in runs:
                xs, ys = get_xy(run, key_x, key_y)
                for x, y in zip(xs, ys):
                    if cutoff is not None and x > cutoff:
                        continue
                    if y is not None and not (isinstance(y, float) and np.isnan(y)):
                        buckets[x].append(float(y))

            if not buckets:
                print(f"[warning] metric “{key_y}” empty for group “{g}” – skipping")
                continue

            xs_sorted = sorted(buckets.keys())
            means = np.array([np.mean(buckets[x]) for x in xs_sorted], dtype=float)
            xs_a = np.asarray(xs_sorted, dtype=float)

            smoothed = ema(means, ema_alpha)

            label_g = group_labels[g_idx] if group_labels else g
            label_y = y_labels[m_idx] if y_labels else key_y
            full_label = (
                f"{label_g} – {label_y}"
                if len(group_names) > 1 or len(y_keys) > 1
                else label_y
            )

            if plot_raw_mean:
                ax.plot(xs_a, means, linestyle=":", linewidth=1.0,
                        color=base_color, alpha=0.6, label=f"{full_label} (mean)")

            ax.plot(xs_a, smoothed, linestyle=style, linewidth=2.0,
                    color=base_color, label=f"{full_label} (EMA α={ema_alpha})")

    ax.set_xlabel(key_x)
    ax.set_ylabel("metric value" if len(y_keys) > 1 else y_keys[0])
    ax.set_title(title or "WandB curves – EMA-smoothed mean (no CI)")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    print(f"✓ saved → {filename}")
