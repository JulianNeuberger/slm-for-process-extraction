import math
import pathlib
import typing

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tabulate
from matplotlib import patches

from data.convert import sanitize_doc
from data.pet import PetImporter, PetDocument
from eval import importing
from eval.radar import spider_plot


def _set_theme():
    sns.set_theme(
        rc={
            "figure.autolayout": False,
            "font.family": ["Computer Modern", "CMU Serif", "cmu", "serif"],
            "font.serif": ["Computer Modern", "CMU Serif", "cmu"],
            # 'text.usetex': True
        }
    )
    matplotlib.rcParams.update(
        {
            "figure.autolayout": False,
            "font.family": ["Computer Modern", "CMU Serif", "cmu", "serif"],
            "font.serif": ["Computer Modern", "CMU Serif", "cmu"],
            # 'text.usetex': True
        }
    )
    sns.set_style(
        rc={
            "font.family": ["Computer Modern", "CMU Serif", "cmu", "serif"],
            "font.serif": ["Computer Modern", "CMU Serif", "cmu"],
            # 'text.usetex': True
        }
    )
    sns.set(font="CMU Serif", font_scale=1.25)
    plt.rcParams["font.family"] = "CMU Serif"


def plot_subsets(base_path: pathlib.Path, models: typing.Dict[str, typing.Dict]):
    n_rows = math.ceil(len(models) / 2) + 1
    n_cols = 2
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                            sharex="col",
                            gridspec_kw={
                                "height_ratios": [0.3] + [1] * (n_rows-1),
                                #"hspace": 1, "wspace": 1
                            }, figsize=(9, 1 + 2 * (n_rows-1)))

    axs = list(axs.flat)
    legend_axs = axs[:2]
    # combine top row
    legend_axs[0].remove()
    legend_axs[1].remove()
    legend_ax = fig.add_subplot(n_rows, n_cols, (1, 2))

    plot_axs = axs[2:]

    ax: plt.Axes
    for model, ax in zip(models, plot_axs):
        subset_df = importing.import_subset_experiment(base_path / models[model]["path"], models[model]["importer"])
        baselines_df = importing.import_relative_experiments(base_path / models[model]["path"],
                                                             experiments=["pet-cv", "synth-cv"],
                                                             baseline="pet-cv",
                                                             import_fn=models[model]["importer"])
        synth_baseline = baselines_df[baselines_df["experiment"] == "synth-cv"]["f1"]
        pet_baseline = baselines_df[baselines_df["experiment"] == "pet-cv"]["f1"]
        sns.lineplot(data=subset_df, x="amount", y="score", ax=ax, label="Partial Synthetic")
        ax.scatter(subset_df["amount"], subset_df["score"], alpha=0.75, s=7, marker="o")
        ax.plot(subset_df["amount"], [synth_baseline] * len(subset_df["amount"]), label="Synthetic", linestyle="--")
        ax.plot(subset_df["amount"], [pet_baseline] * len(subset_df["amount"]), label="PET", linestyle="-.")
        ax.collections[0].set_label("95% confidence interval")
        ax.set_title(models[model]["title"])
        ax.set_xticks(ax.get_xticks(), [f"{x:.0%}" for x in ax.get_xticks()])
        ax.set_xlabel("Dataset Size")
        ax.set_yticks(ax.get_yticks(), [f"{x:.0f}%" for x in ax.get_yticks()])
        ax.set_ylabel("F1")
        ax.get_legend().remove()

    legend_handles, legend_labels = plot_axs[0].get_legend_handles_labels()
    legend_ax.grid(False)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.set_facecolor((0, 0, 0, 0))
    for ax in reversed(plot_axs[len(models):]):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor((0, 0, 0, 0))
        ax.remove()
    legend_ax.legend(legend_handles, legend_labels, ncols=4, labelspacing=0, borderpad=0.2, loc="upper right", bbox_to_anchor=(1,1))

    fig: plt.Figure
    fig.suptitle("")
    fig.tight_layout()
    fig_folder = pathlib.Path(__file__).parent.parent.parent / "figures" / "subsets"
    fig_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_folder / f"{'-'.join(models)}.png")
    fig.savefig(fig_folder / f"{'-'.join(models)}.pdf")


def print_results(base_path: pathlib.Path, experiments: typing.List[str], baseline: str,
                  models: typing.Dict[str, typing.Dict]):
    for model, import_fn in models.items():
        df = importing.import_relative_experiments(base_path / models[model]["path"], experiments, baseline,
                                                   models[model]["importer"])
        print(f"--- {model}")
        print(tabulate.tabulate(df, headers="keys", tablefmt="psql"))
        print()
        print()


def get_data_statistics(base_path: pathlib.Path) -> pd.DataFrame:
    def count_types(ds: typing.List[PetDocument]) -> typing.Tuple[typing.Dict[str, int], typing.Dict[str, int]]:
        ent_counts = {}
        rel_counts = {}
        for d in ds:
            for r in d.relations:
                if r.type not in rel_counts:
                    rel_counts[r.type] = 0
                rel_counts[r.type] += 1
            for m in d.mentions:
                if m.type not in ent_counts:
                    ent_counts[m.type] = 0
                ent_counts[m.type] += 1
        return ent_counts, rel_counts


    def span_length(ds: typing.List[PetDocument]) -> typing.Tuple[typing.Dict[str, typing.List[int]], typing.Dict[str, typing.List[int]]]:
        ent_len = {}
        rel_len = {}

        for d in ds:
            for r in d.relations:
                if r.type not in rel_len:
                    rel_len[r.type] = []
                head_start = d.mentions[r.head_mention_index].token_document_indices[0]
                tail_start = d.mentions[r.tail_mention_index].token_document_indices[0]
                rel_len[r.type].append(abs(head_start - tail_start))
            for m in d.mentions:
                if m.type not in ent_len:
                    ent_len[m.type] = []
                ent_len[m.type].append(m.token_document_indices[-1] - m.token_document_indices[0] + 1)
        return ent_len, rel_len

    data = {
        "dataset": [],
        "target": [],
        "type": [],
        "count": [],
        "rel_count": [],
        "avg_length": []
    }

    for ds_name, label in {"pet": "PET", "all": "Synthetic Data"}.items():
        data_set = PetImporter(base_path / f"{ds_name}.jsonl").do_import()
        for doc in data_set:
            sanitize_doc(doc)
        e_counts, r_counts = count_types(data_set)
        e_lengths, r_lengths = span_length(data_set)
        for e_type in e_counts:
            e_count = e_counts[e_type]
            e_lens = e_lengths[e_type]
            data["dataset"].append(label)
            data["target"].append("entities")
            data["type"].append(e_type)
            data["count"].append(e_count)
            data["rel_count"].append(e_count / sum(e_counts.values()))
            data["avg_length"].append(sum(e_lens) / len(e_lens))
        for r_type in r_counts:
            r_count = r_counts[r_type]
            r_lens = r_lengths[r_type]
            data["dataset"].append(label)
            data["target"].append("relations")
            data["type"].append(r_type)
            data["count"].append(r_count)
            data["rel_count"].append(r_count / sum(r_counts.values()))
            data["avg_length"].append(sum(r_lens) / len(r_lens))

    return pd.DataFrame(data)



if __name__ == "__main__":
    root_dir = pathlib.Path(__file__).parent.parent.parent / "resources"

    def plot_document_lengths():
        base_path = root_dir / "data" / "original"
        data = {
            "dataset": [],
            "length": []
        }
        colors = sns.color_palette()
        handles = []
        for i, (ds_name, label) in enumerate({"pet": "PET", "all": "Synthetic"}.items()):
            handles.append(patches.Patch(facecolor=colors[i], alpha=0.5, fill=True, label=label))
            data_set = PetImporter(base_path / f"{ds_name}.jsonl").do_import()
            for doc in data_set:
                sanitize_doc(doc)
                data["dataset"].append(label)
                data["length"].append(len(doc.tokens))

        df = pd.DataFrame(data)
        fig = plt.figure(figsize=(5, 3))

        ax = sns.kdeplot(df, x="length", hue="dataset", fill=True, alpha=0.5, common_norm=False, legend=False)

        ax.legend(handles=handles)

        fig.tight_layout()
        fig_folder = pathlib.Path(__file__).parent.parent.parent / "figures" / "statistics"
        fig_folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_folder / f"doc-lengths.png")
        fig.savefig(fig_folder / f"doc-lengths.pdf")

    def plot_ablation():
        base_path = root_dir / "results"

        models = {
            "plmarker": {"importer": importing.import_plmarker, "title": "PL-Marker", "path": "plmarker"},
            "ace": {"importer": importing.import_ace, "title": "ACE", "path": "ace"},
            "unirel": {"importer": importing.import_uni_rel, "title": "UniRel", "path": "unirel"},
            "piqn": {"importer": importing.import_piqn, "title": "PIQN", "path": "piqn"},
        }

        experiments = ["sbvr", "image", "combined", "no_hints", "full", "fine-tune"]
        baseline = "sbvr"

        print_results(base_path, experiments, baseline, models)
        # plot_subsets(base_path, {
        #     "ace": models["ace"],
        #     "piqn": models["piqn"],
        #     "plmarker": models["plmarker"],
        #     "unirel": models["unirel"],
        # })

        # experiments = ["synth-cv", "no-hints"]
        # baseline = "synth-cv"
        #
        # print_results(base_path, experiments, baseline, {
        #     "ace": {"importer": importing.import_ace, "title": "ACE", "path": "ace"},
        #     "unirel": {"importer": importing.import_uni_rel, "title": "UniRel", "path": "unirel"},
        #     "piqn": {"importer": importing.import_piqn, "title": "PIQN", "path": "piqn"},
        # })


    def plot_statistics():
        base_path = root_dir / "data" / "original"
        df = get_data_statistics(base_path)

        print_df = df.copy()
        print_df["rel_count"] = df["rel_count"].apply(lambda x: f"{x:.1%}")
        print("--- entity stats")
        print(tabulate.tabulate(print_df[print_df["target"] == "entities"], headers="keys", tablefmt="psql"))
        print()
        print("--- relation stats")
        print(tabulate.tabulate(print_df[print_df["target"] == "relations"], headers="keys", tablefmt="psql"))

        fig_folder = pathlib.Path(__file__).parent.parent.parent / "figures" / "statistics"
        fig_folder.mkdir(parents=True, exist_ok=True)

        fig, _ = spider_plot(df, x="type", y="rel_count", subplots="target", hue="dataset", figsize=(10, 5.0), y_format=lambda y: f"{y:.1%}")
        fig.savefig(fig_folder / f"type-statistics.png")
        fig.savefig(fig_folder / f"type-statistics.pdf")

        fig, _ = spider_plot(df, x="type", y="avg_length", subplots="target", hue="dataset", figsize=(10, 5.0), y_format=lambda y: f"{y:.1f}")
        fig.savefig(fig_folder / f"len-statistics.png")
        fig.savefig(fig_folder / f"len-statistics.pdf")


    def main():
        plot_ablation()
        # plot_statistics()
        # plot_document_lengths()


    _set_theme()
    main()
