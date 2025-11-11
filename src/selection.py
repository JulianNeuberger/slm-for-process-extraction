import csv
import enum
import json
import pathlib
import typing

import langdetect
import networkx as nx
from tqdm import tqdm

import conversion
import mappings
import patterns


class ConstraintState(enum.Enum):
    HOLDS = 0
    VIOLATED = 1
    UNDECIDED = 2


class BaseConstraint:
    def __init__(self):
        self._state = ConstraintState.UNDECIDED

    def holds_finally(self) -> ConstraintState:
        raise NotImplementedError()

    def update(self, shape: typing.Dict) -> ConstraintState:
        raise NotImplementedError()


class StencilConstraint(BaseConstraint):
    def __init__(self, *, allowed_stencils: typing.Set[str], disallowed_stencils: typing.Set[str]):
        super().__init__()
        self._stencils = allowed_stencils
        self._disallowed_stencils = disallowed_stencils

    def update(self, shape: typing.Dict):
        if "stencil" not in shape or "id" not in shape["stencil"]:
            self._state = ConstraintState.VIOLATED
            return self._state
        if shape["stencil"]["id"] in self._disallowed_stencils:
            self._state = ConstraintState.VIOLATED
            return self._state
        if shape["stencil"]["id"] in self._stencils:
            self._state = ConstraintState.UNDECIDED
            return self._state
        print(f"+++ Unknown stencil {shape['stencil']['id']}")
        return self._state

    def holds_finally(self) -> ConstraintState:
        if self._state == ConstraintState.UNDECIDED:
            self._state = ConstraintState.HOLDS
        return self._state


class LabelLengthConstraint(BaseConstraint):
    def __init__(self, activity_stencils: typing.List[str]):
        super().__init__()
        self._activity_stencils = activity_stencils

    def update(self, shape: typing.Dict):
        if shape["stencil"]["id"] not in self._activity_stencils:
            return self._state
        if "properties" not in shape:
            return self._state
        if "name" not in shape["properties"]:
            return self._state
        if len(shape["properties"]["name"]) <= 1:
            self._state = ConstraintState.VIOLATED
        return self._state

    def holds_finally(self) -> ConstraintState:
        if self._state == ConstraintState.UNDECIDED:
            self._state = ConstraintState.HOLDS
        return self._state


class LanguageConstraint(BaseConstraint):
    def __init__(self, allowed_languages: typing.List[str]):
        super().__init__()
        self._text = ""
        self._languages = allowed_languages
        POSSIBLE_LANGS = ["af", "ar", "bg", "bn", "ca", "cs",
                          "cy", "da", "de", "el", "en", "es",
                          "et", "fa", "fi", "fr", "gu", "he",
                          "hi", "hr", "hu", "id", "it", "ja",
                          "kn", "ko", "lt", "lv", "mk", "ml",
                          "mr", "ne", "nl", "no", "pa", "pl",
                          "pt", "ro", "ru", "sk", "sl", "so",
                          "sq", "sv", "sw", "ta", "te", "th",
                          "tl", "tr", "uk", "ur", "vi", "zh-cn",
                          "zh-tw"]
        for l in allowed_languages:
            assert l in POSSIBLE_LANGS, f"Unsupported language: {l}, choose one of: {POSSIBLE_LANGS}"

    def update(self, shape: typing.Dict):
        if "properties" not in shape:
            return self._state
        if "name" not in shape["properties"]:
            return self._state
        label = shape["properties"]["name"]
        if label.strip() == "":
            return self._state
        self._text += label + " "
        return self._state

    def holds_finally(self) -> ConstraintState:
        try:
            detected = langdetect.detect(self._text)
        except langdetect.LangDetectException:
            return ConstraintState.VIOLATED
        if detected not in self._languages:
            return ConstraintState.VIOLATED
        return ConstraintState.HOLDS

class RequiredStencilConstraint(BaseConstraint):
    def __init__(self, required_stencils: typing.Set[str]):
        super().__init__()
        self._stencils = required_stencils
        self._seen = set()

    def update(self, shape: typing.Dict):
        self._seen.add(shape["stencil"]["id"])
        if len(self._stencils - self._seen):
            self._state = ConstraintState.HOLDS
        return self._state

    def holds_finally(self):
        if len(self._stencils - self._seen):
            self._state = ConstraintState.HOLDS
        return self._state

class ElementOccurrencesConstraint(BaseConstraint):
    def __init__(self,
                 element_min_max: typing.Dict[str, typing.Tuple[int, int]],
                 stencil_mapping: mappings.MappingCollection):
        super().__init__()
        self._mapping = stencil_mapping.all
        self._element_min_max = element_min_max
        self._element_occurrences = {}

    def update(self, shape: typing.Dict):
        if shape["stencil"]["id"] not in self._mapping:
            return self._state
        stencil = self._mapping[shape["stencil"]["id"]]
        if stencil not in self._element_occurrences:
            self._element_occurrences[stencil] = 0
        self._element_occurrences[stencil] += 1
        return self._state

    def holds_finally(self) -> ConstraintState:
        for stencil, (lo, hi) in self._element_min_max.items():
            if stencil not in self._element_occurrences:
                return ConstraintState.VIOLATED
            if self._element_occurrences[stencil] < lo:
                return ConstraintState.VIOLATED
            if self._element_occurrences[stencil] > hi:
                return ConstraintState.VIOLATED
        return ConstraintState.HOLDS


class ConnectivityConstraint(BaseConstraint):
    def __init__(self, stencil_mapping: typing.Dict[str, str]):
        super().__init__()
        self._stencil_mapping = stencil_mapping
        self._root_seen = False

    def update(self, shape: typing.Dict):
        if self._root_seen:
            print("WARN: Connectivity Constraint was called twice, "
                  "possible with a non-root stencil, which will "
                  "result in undefined behavior.")
        self._root_seen = True
        g = conversion.sam_json_to_networkx(shape, self._stencil_mapping)
        if nx.is_empty(g):
            return ConstraintState.VIOLATED
        if not nx.is_connected(g.to_undirected()):
            return ConstraintState.VIOLATED
        return ConstraintState.HOLDS

    def holds_finally(self):
        print(f"WARN: Called holds_finally on {self.__class__.__name__}, "
              f"which should not happen, "
              f"the first shape should evaluate this constraint.")
        raise AssertionError()


class ReachabilityConstraint(BaseConstraint):
    def __init__(self, stencil_mapping: mappings.MappingCollection):
        super().__init__()
        self._stencil_mapping = stencil_mapping
        self._root_seen = False

    def update(self, shape: typing.Dict):
        if self._root_seen:
            print("WARN: ExplicitActorConstraint was called twice, "
                  "possible with a non-root stencil, which will "
                  "result in undefined behavior.")
        self._root_seen = True
        g = conversion.sam_json_to_networkx(shape, self._stencil_mapping.behaviour)
        start_candidates = [n for n, degree in g.in_degree if degree == 0]
        for n in g.nodes:
            if not any(nx.has_path(g, s, n) for s in start_candidates):
                self._state = ConstraintState.VIOLATED
                return self._state
        self._state = ConstraintState.HOLDS
        return self._state


class ExplicitActorConstraint(BaseConstraint):
    def __init__(self, stencil_mapping: mappings.MappingCollection, actor_type: str, checked_types: typing.Set[str]):
        super().__init__()
        self._stencil_mapping = stencil_mapping
        self._root_seen = False
        self._actor_type = actor_type
        self._checked_types = checked_types

    def update(self, shape: typing.Dict):
        if self._root_seen:
            print("WARN: ExplicitActorConstraint was called twice, "
                  "possible with a non-root stencil, which will "
                  "result in undefined behavior.")
        self._root_seen = True
        g = conversion.sam_json_to_networkx(shape, self._stencil_mapping.all)
        for node, node_type in g.nodes(data="type"):
            if node_type not in self._checked_types:
                continue

            actor = patterns.get_actor(g, node, actor_type=self._actor_type)
            if actor is None:
                # print(f"Missing actor for '{g.nodes[node]['label']} ({node_type})'")
                # print("Connected to :")
                # for n in patterns.neighbors(g, node, direction="ignore"):
                #     print(f"\t{g.nodes[n]['label']} ({g.nodes[n]['type']})")
                return ConstraintState.VIOLATED
        return ConstraintState.HOLDS


class ProcessStartConstraint(BaseConstraint):
    def __init__(self, stencil_mapping: typing.Dict[str, str], allowed_start_types: typing.Set[str]):
        super().__init__()
        self._stencil_mapping = stencil_mapping
        self._allowed_start_types = allowed_start_types
        self._root_seen = False

    def update(self, shape: typing.Dict):
        if self._root_seen:
            print("WARN: Connectivity Constraint was called twice, "
                  "possible with a non-root stencil, which will "
                  "result in undefined behavior.")
        self._root_seen = True
        g = conversion.sam_json_to_networkx(shape, self._stencil_mapping)
        for node, node_type in g.nodes(data="type"):
            if node_type not in self._stencil_mapping.values():
                continue
            if node_type in self._allowed_start_types:
                continue
            if g.in_degree(node) == 0:
                return ConstraintState.VIOLATED
        return ConstraintState.HOLDS

    def holds_finally(self):
        print(f"WARN: Called holds_finally on {self.__class__.__name__}, "
              f"which should not happen, "
              f"the first shape should evaluate this constraint.")
        raise AssertionError()


def filter_models(constraint_factories: typing.List[typing.Callable[[], typing.List[BaseConstraint]]],
                  *,
                  output_file_path: pathlib.Path,
                  input_file_path: pathlib.Path):

    def traverse(_shape: typing.Dict,
                 _constraints: typing.List[BaseConstraint],
                 _violations: typing.Dict[str, int]) -> bool:
        satisfied_constraints = []
        for constraint in _constraints:
            constraint_state = constraint.update(_shape)
            if constraint_state == ConstraintState.VIOLATED:
                if constraint.__class__.__name__ not in _violations:
                    _violations[constraint.__class__.__name__] = 0
                _violations[constraint.__class__.__name__] += 1
                return False
            if constraint_state == ConstraintState.HOLDS:
                # remove constraint, no need to check again
                satisfied_constraints.append(constraint)
        for s in satisfied_constraints:
            constraints.remove(s)

        for child in _shape.get("childShapes", []):
            should_continue = traverse(child, _constraints, _violations)
            if not should_continue:
                return False
        return True

    plausible_models = []
    total_num_models = 0

    constraint_names = [c.__class__.__name__ for cf in constraint_factories for c in cf()]

    violations = {c: 0 for c in constraint_names}
    violations.update({
        "NotBPMN": 0,
        "Empty": 0
    })
    with open(input_file_path, "r", encoding="utf8") as f:
        csvfile = csv.reader(f, delimiter=",", quotechar='"')
        header = next(csvfile)
        for row in tqdm(csvfile):
            total_num_models += 1
            all_hold = True
            for constraint_factory in constraint_factories:
                if not all_hold:
                    break
                constraints = constraint_factory()

                if row[8] != "http://b3mn.org/stencilset/bpmn2.0#":
                    violations["NotBPMN"] += 1
                    all_hold = False
                    break
                sam_json = json.loads(row[4])
                if "childShapes" not in sam_json:
                    violations["Empty"] += 1
                    all_hold = False
                    break

                plausible = traverse(sam_json, constraints, violations)
                if not plausible:
                    all_hold = False
                    break

                for c in constraints:
                    if c.holds_finally() != ConstraintState.HOLDS:
                        all_hold = False
                        violations[c.__class__.__name__] += 1
                        break

            if all_hold:
                plausible_models.append(row)

    print(f"Number of plausible models: {len(plausible_models)} of a total of {total_num_models}")
    print("Violations per constraint:")
    print("--------------------------")
    for c, v in sorted(violations.items(), key=lambda x: x[1]):
        print(f"{c}: {v}")
    print("--------------------------")
    print(f"Total of {sum(violations.values())}")

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, "w", encoding="utf8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(plausible_models)
    violations_file_name = f"{output_file_path.stem}-violations.json"
    violations_file_path = output_file_path.parent / violations_file_name
    with open(violations_file_path, "w", encoding="utf8") as f:
        json.dump(violations, f)



if __name__ == "__main__":
    def _basic_constraints():
        mapping = mappings.SapSamMappingCollection()
        return [
            StencilConstraint(allowed_stencils=set(mapping.all.keys()) | mapping.ignored,
                              disallowed_stencils=mapping.disallowed),
            LabelLengthConstraint(activity_stencils=["Task"]),
            LanguageConstraint(["en"]),
            ElementOccurrencesConstraint({
                "Activity": (8, 40),
                "StartEvent": (1, 1),
                "EndEvent": (1, 1),
                "Gateway": (1, 10),
                "Actor": (1, 10),
            }, stencil_mapping=mappings.SimpleSapSamCollection()),
        ]

    def _structure_constraints():
        mapping = mappings.SapSamMappingCollection()
        return [
            ConnectivityConstraint(mapping.behaviour),
            ProcessStartConstraint(mapping.behaviour, {"StartEvent"}),
            ExplicitActorConstraint(mapping, actor_type="Actor", checked_types={"Activity"}),
            ReachabilityConstraint(mapping)
        ]

    def main():
        csv.field_size_limit(2147483647)

        raw_models_dir = pathlib.Path(__file__).parent.parent / "resources" / "models" / "raw"
        plausible_models_dir = pathlib.Path(__file__).parent.parent / "resources" / "models" / "selected"

        for models_file_path in raw_models_dir.iterdir():
            out_file_path = plausible_models_dir / models_file_path.name
            if out_file_path.exists():
                print(f"Skipping {models_file_path.name} as it already exists")
                continue
            filter_models([_basic_constraints, _structure_constraints],
                          input_file_path=models_file_path,
                          output_file_path=out_file_path)
        print("All done!")

        all_violations = {}
        num_models = 0
        for models_file_path in plausible_models_dir.iterdir():
            if models_file_path.suffix == ".json":
                with open(models_file_path, "r", encoding="utf8") as f:
                    violations = json.load(f)
                    for k, v in violations.items():
                        if k not in all_violations:
                            all_violations[k] = 0
                        all_violations[k] += v
            elif models_file_path.suffix == ".csv":
                print(models_file_path)
                with open(models_file_path, "r", encoding="utf8") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for _ in reader:
                        num_models += 1
            else:
                raise AssertionError(models_file_path)
        print()
        print(f"Number of selected models: {num_models}")
        print("Violations per constraint:")
        print("--------------------------")
        for c, v in sorted(all_violations.items(), key=lambda x: x[1]):
            print(f"{c}: {v}")
        print("--------------------------")
        print(f"Total of {sum(all_violations.values())}")


    main()


