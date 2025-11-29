import dataclasses
import pathlib
import typing

import numpy as np
import seaborn
from matplotlib import pyplot as plt

resources_folder = pathlib.Path(__file__).parent.parent / "resources"


@dataclasses.dataclass
class PowerAndTimings:
    powers: typing.List[float]
    times: typing.List[float]
    total_power: float
    runtime_seconds: float


def trapezoidal_integration(data: typing.List[typing.Tuple[float, float]]) -> float:
    total = 0
    for i in range(len(data) - 1):
        a, fa = data[i]
        b, fb = data[i + 1]

        # rectangle
        total += (b - a) * fa
        # triangle
        total += 0.5 * (b - a) * (fb - fa)
    return total / (60 * 60)


def read_power_file(power_file_path: pathlib.Path) -> typing.Dict[str, PowerAndTimings]:
    readings_by_doc = {}
    with open(power_file_path, "r") as power_file:
        for l in power_file:
            l = l.strip()
            if len(l) == 0:
                continue
            timestamp, doc_id, power_draw = l.split("\t")
            if doc_id not in readings_by_doc:
                readings_by_doc[doc_id] = []
            readings_by_doc[doc_id].append((float(timestamp), float(power_draw)))
    return {
        d: PowerAndTimings(
            powers=list(x[1] for x in readings),
            times=list(x[0] for x in readings),
            runtime_seconds=readings[-1][0] - readings[0][0],
            total_power=trapezoidal_integration(readings)
        )
        for d, readings in readings_by_doc.items()
    }

def main():
    max_len = max(len(a.name) for a in (resources_folder / "results").iterdir()) + 5
    print(f"{'Model':<{max_len}}\tPower [Wh]\t{'std':>5}\ttime [s]\t{'std':>6}")
    for mode in ["re", "md"]:
        for approach in (resources_folder / "results").iterdir():
            power_file = approach / mode / "power.csv"
            if not power_file.exists():
                if mode == "md" and (approach.name == "unirel" or approach.name == "plmarker"):
                    continue
                if mode == "re" and (approach.name == "piqn" or approach.name == "ace"):
                    continue
                #print(f"Skipping {approach}, no recorded power measures")
                power_file = approach / "power.csv"
            if not power_file.exists():
                continue
            readings_by_doc = read_power_file(power_file)
            powers = np.array([r.total_power for r in readings_by_doc.values()])
            times = np.array([r.runtime_seconds for r in readings_by_doc.values()])
            print(f"{mode + ' | ' + approach.name:<{max_len}}\t{np.mean(powers):10.2f}\t{np.std(powers):5.2f}\t{np.mean(times):8.2f}\t{np.std(times):6.2f}")
            if "doc-10.14" in readings_by_doc:
                seaborn.set_style("whitegrid")
                plt.figure(figsize=(10.2, 2.3))
                xs = np.array(readings_by_doc["doc-10.14"].times) - readings_by_doc["doc-10.14"].times[0]
                ys = np.array(readings_by_doc["doc-10.14"].powers)
                seaborn.lineplot(x=xs, y=ys)
                plt.ylabel("Power Draw [W]")
                plt.xlabel("Time [s]")
                (resources_folder / "figures" / "power").mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig(resources_folder / "figures" / "power" / f"{approach.name}.pdf")
                plt.savefig(resources_folder / "figures" / "power" / f"{approach.name}.png")

main()