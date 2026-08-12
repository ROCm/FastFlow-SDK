# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builds the FastFlowLM kernel qualification recipe from measured facts.

The recipe is indexed by kernel implementation rather than by model family,
because identical xclbin bytes are one implementation no matter how many
families ship them. It carries generators rather than tensors, so the buffers
and oracles are computed at qualification time instead of shipped.

Two things are deliberately separated here. `build_recipe` turns measured facts
into cases and is pure, so it can be tested without a build tree or an NPU.
Asset generation supplies the measured build outputs, because only they know the
digests that bind a case to the exact bytes it ran against.

The facts live with the qualification product in
`fastflowlm-kernel-capabilities-v1.json`; native tests consume that same
contract. They are measurements, not derivations: no
part of an xclbin says which transaction builder reaches it, which Q4NX chunk
shape its packing uses, or which order it writes its output in.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

PRODUCTION_SHAPE_KEY = "production"
WIDE_SHAPE_KEY = "wide"

# Dequant runs the two populations that move the arithmetic; MM adds the subset
# every family agrees on, which is the only one a shared implementation can be
# held to at the tightest bound.
DEQUANT_PROFILES = ("shared-subset", "full-nibble", "full-affine")
MM_PROFILES = ("shared-subset", "full-nibble", "full-affine")

# Salts pick deterministic activation content. The first is the resident case's
# own input; the others are rewritten underneath it during readback rounds.
BASE_SALT = 41
LEGACY_SALT = 29
READBACK_SALTS = (46, 54)


class RecipeError(RuntimeError):
    pass


def _generator(id, **parameters):
    return {
        "id": id,
        "parameters": {k: v for k, v in parameters.items() if v is not None},
    }


def _shape(facts, key):
    if key == PRODUCTION_SHAPE_KEY:
        return dict(facts["shape"])
    if key == WIDE_SHAPE_KEY:
        return dict(facts["implementations"]["wide_shape"])
    raise RecipeError(f"unknown shape key: {key}")


def _buffer(stage, name):
    record = next(
        (record for record in stage["buffers"] if record["name"] == name),
        None,
    )
    if record is None:
        raise RecipeError(f"{stage['name']} stage has no {name!r} buffer")
    return record


def _case(cases, name):
    case = next((case for case in cases if case["name"] == name), None)
    if case is None:
        raise RecipeError(f"recipe has no case named {name!r}")
    return case


def _dequant_stage(entry, profile):
    shape = entry["shape"]
    m, k = shape["m"], shape["k"]
    blocks = entry["facts"]["q4nx_blocks"]
    zero = profile == "zero"
    return {
        "name": "dequant",
        "implementation": entry["id"],
        "buffers": [
            {
                "name": "output",
                "direction": "output",
                "size": 2 * m * k,
                "expected": _generator("zero", size=2 * m * k)
                if zero
                else _generator(
                    "q4nx-dequant-expected",
                    m=m,
                    k=k,
                    profile=profile,
                    semantics=entry["facts"]["dequant_semantics"],
                    order=entry["facts"]["dequant_output_order"],
                ),
                "comparison": {"kind": "exact"},
            },
            {
                "name": "packed_weight",
                "direction": "input",
                "size": m * k * 5 // 8,
                "source": _generator("zero", size=m * k * 5 // 8)
                if zero
                else _generator(
                    "q4nx-packed",
                    m=m,
                    k=k,
                    profile=profile,
                    row_block=blocks["row_block"],
                    col_block=blocks["col_block"],
                    parallel=blocks["parallel"],
                ),
            },
            {
                "name": "parameters",
                "direction": "input",
                "size": 4096,
                "source": _generator("zero", size=4096),
            },
        ],
    }


def _execution(entry, execution):
    if execution is None:
        return entry
    try:
        variant = next(
            variant
            for variant in entry.get("execution_variants", ())
            if variant["id"] == execution
        )
    except StopIteration as error:
        raise RecipeError(
            f"{entry['id']} has no execution variant {execution!r}"
        ) from error
    return {
        **entry,
        "shape": variant["shape"],
        "facts": {**entry["facts"], "builder": variant["builder"]},
    }


def _mm_stage(entry, profile, policy, salt=BASE_SALT, execution=None):
    entry = _execution(entry, execution)
    shape = entry["shape"]
    m, k, n = shape["m"], shape["k"], shape["n"]
    zero = profile == "zero"
    return {
        "name": "mm",
        "implementation": entry["id"],
        **({} if execution is None else {"execution": execution}),
        "buffers": [
            {
                "name": "output",
                "direction": "output",
                "size": 2 * m * n,
                "expected": _generator("zero", size=2 * m * n)
                if zero
                else _generator(
                    "gemm-reference", m=m, k=k, n=n, profile=profile, salt=salt
                ),
                "comparison": {"kind": "exact"}
                if zero
                else {
                    "kind": "bf16-tolerance",
                    "atol": policy["gemm_absolute_tolerance"][profile],
                    "rtol": policy["relative_tolerance"],
                },
            },
            {
                "name": "weight",
                "direction": "input",
                "size": 2 * m * k + 4096,
                "source": _generator("zero", size=2 * m * k + 4096)
                if zero
                else _generator("mm-weight", m=m, k=k, profile=profile),
            },
            {
                "name": "activation",
                "direction": "input",
                "size": 2 * k * n,
                "source": _generator("zero", size=2 * k * n)
                if zero
                else _generator(
                    "mm-activation-layout",
                    k=k,
                    n=n,
                    salt=salt,
                    builder=entry["facts"]["builder"],
                ),
            },
        ],
    }


def _chained_case(
    dequant, mm, profile, policy, mm_execution=None, salt=BASE_SALT
):
    """Binds a dequant output straight into the MM weight, with no host round trip.

    The dequant kernel writes its result permuted and the MM kernel reads its
    weight row-major, so the chain is not the plain product. Measured on device,
    it is the product of the weight exactly as the first stage wrote it, which is
    why the oracle stays computable instead of becoming a recorded baseline.
    """
    m, k = dequant["shape"]["m"], dequant["shape"]["k"]
    selected_mm = _execution(mm, mm_execution)
    n = selected_mm["shape"]["n"]
    shared = {"name": "weight", "size": 2 * m * k + 4096}

    first = _dequant_stage(dequant, profile)
    # The output plays the MM weight role, so it takes that size and carries no
    # expectation of its own: the chain is judged on the product it feeds.
    first_output = _buffer(first, "output")
    first_output.clear()
    first_output.update({
        **shared,
        "direction": "output",
        "preserved": _generator(
            "chained-weight",
            m=m,
            k=k,
            profile=profile,
            semantics=dequant["facts"]["dequant_semantics"],
            order=dequant["facts"]["dequant_output_order"],
        ),
    })

    second = _mm_stage(
        mm, profile, policy, salt=salt, execution=mm_execution
    )
    second_output = _buffer(second, "output")
    second_output["expected"] = _generator(
        "chained-reference",
        m=m,
        k=k,
        n=n,
        profile=profile,
        salt=salt,
        semantics=dequant["facts"]["dequant_semantics"],
        order=dequant["facts"]["dequant_output_order"],
    )
    second_output["comparison"] = {
        "kind": "bf16-tolerance",
        "atol": policy["chained_absolute_tolerance"][profile],
        "rtol": policy["relative_tolerance"],
    }
    second_weight = _buffer(second, "weight")
    second_weight.clear()
    second_weight.update({
        **shared,
        "direction": "input",
        "source": {"stage_output": "dequant.weight"},
    })
    return {
        "name": (
            f"{dequant['id']}+{mm['id']}"
            f"{'' if mm_execution is None else '/' + mm_execution}/{profile}"
        ),
        "stages": [first, second],
    }


def _readback_variant(entry, policy, salt, execution=None):
    stage = _mm_stage(
        entry, "shared-subset", policy, salt=salt, execution=execution
    )
    return {
        "stage": "mm",
        "buffer": "activation",
        "source": _buffer(stage, "activation")["source"],
        "expected": _buffer(stage, "output")["expected"],
    }


def _consumers(catalogue, reached):
    """Which MM implementation consumes each dequant implementation's output.

    Not a choice: every dequant implementation shares model families with
    exactly one MM implementation, so the catalogue already says which pair runs
    together in production.
    """
    mm = {
        record["id"]: set(record["families"])
        for record in catalogue
        if record["kernel"] == "mm" and record["id"] in reached
    }
    pairs = {}
    for record in catalogue:
        if record["kernel"] != "dequant" or record["id"] not in reached:
            continue
        matches = [
            partner
            for partner, families in mm.items()
            if families & set(record["families"])
        ]
        if not matches:
            # The matching shipped MM implementation is a declared coverage
            # limit (for example, because FastFlowLM exposes no supported
            # transaction-builder API). The dequant implementation is still
            # independently useful; only its end-to-end chain is unavailable.
            continue
        if len(matches) != 1:
            raise RecipeError(
                f"{record['id']} shares families with {len(matches)} MM "
                "implementations, so the chain it feeds is no longer implied by "
                "the catalogue"
            )
        pairs[record["id"]] = matches[0]
    return pairs


def build_recipe(capabilities, catalogue, digests):
    """Turns measured facts and a shipped-implementation catalogue into a recipe.

    `catalogue` is one record per unique implementation, as the implementation
    probe reports them. `digests` supplies the per-implementation identity a
    build knows and these facts do not: the xclbin, transaction, and XADX
    digests that bind a case to the exact bytes it ran against.
    """
    measured = capabilities["implementations"]
    reached = measured["entries"]
    policy = capabilities["recipe_policy"]
    undeclared = set(measured["undeclared_builders"])

    implementations, cases, limits = [], [], []
    entries = {}
    for record in catalogue:
        id = record["id"]
        facts = reached.get(id)
        if facts is None:
            reason = measured["unreached"].get(id)
            if reason is None:
                raise RecipeError(
                    f"implementation {id} has neither measured facts nor a "
                    "declared coverage limit"
                )
            limits.append(
                {
                    "id": id,
                    "scope": "dispatch",
                    "kernel": record["kernel"],
                    "representative": record["representative"],
                    "xclbin_sha256": record["xclbin_sha256"],
                    "families": record["families"],
                    "reason": reason,
                }
            )
            continue
        if id not in digests:
            raise RecipeError(f"no build digests for implementation {id}")
        expected_id = f"{record['kernel']}-{record['xclbin_sha256'][:16]}"
        if id != expected_id:
            raise RecipeError(
                f"implementation ID {id} does not match its source xclbin "
                f"identity {expected_id}"
            )
        digest = dict(digests[id])
        variant_digests = digest.pop("execution_variants", {})
        execution_variants = []
        for variant in facts.get("execution_variants", ()):
            variant_id = variant["id"]
            if variant_id not in variant_digests:
                raise RecipeError(
                    f"no build digests for {id} execution variant {variant_id}"
                )
            execution_variants.append(
                {
                    "id": variant_id,
                    "builder": variant["builder"],
                    "shape": _shape(capabilities, variant["shape"]),
                    "export": "MLIR_AIE",
                    **variant_digests[variant_id],
                }
            )
        entry = {
            "id": id,
            "kernel": record["kernel"],
            "builder": facts["builder"],
            "builder_declared": facts["builder"] not in undeclared,
            "shape": _shape(capabilities, facts["shape"]),
            "xclbin_bytes": record["xclbin_bytes"],
            "export": "MLIR_AIE",
            "representative": record["representative"],
            "families": record["families"],
            **digest,
            **(
                {"execution_variants": execution_variants}
                if execution_variants
                else {}
            ),
        }
        entries[id] = {**entry, "facts": facts}
        implementations.append(entry)

        staged = entries[id]

        def stage(profile, staged=staged, kernel=record["kernel"]):
            if kernel == "dequant":
                return _dequant_stage(staged, profile)
            return _mm_stage(staged, profile, policy)

        cases.append(
            {"name": f"{id}/zero-baseline", "stages": [stage("zero")]}
        )
        if "numeric_limit" in facts:
            limits.append(
                {
                    "id": id,
                    "scope": "numeric",
                    "kernel": record["kernel"],
                    "representative": record["representative"],
                    "xclbin_sha256": record["xclbin_sha256"],
                    "families": record["families"],
                    "reason": facts["numeric_limit"],
                }
            )
            continue
        profiles = (
            DEQUANT_PROFILES if record["kernel"] == "dequant" else MM_PROFILES
        )
        for profile in profiles:
            cases.append(
                {"name": f"{id}/{profile}", "stages": [stage(profile)]}
            )
        if record["kernel"] == "mm":
            # Alternate inputs rewritten into a resident buffer between
            # dispatches, which is what the readback rounds exercise.
            shared_subset = _case(cases, f"{id}/shared-subset")
            shared_subset["readback_variants"] = [
                _readback_variant(staged, policy, salt)
                for salt in READBACK_SALTS
            ]
            for variant in staged.get("execution_variants", ()):
                execution = variant["id"]
                for profile in ("zero", *MM_PROFILES):
                    cases.append(
                        {
                            "name": f"{id}/{execution}/{profile}",
                            "stages": [
                                _mm_stage(
                                    staged,
                                    profile,
                                    policy,
                                    salt=LEGACY_SALT,
                                    execution=execution,
                                )
                            ],
                        }
                    )
                variant_shared = _case(
                    cases, f"{id}/{execution}/shared-subset"
                )
                variant_shared["readback_variants"] = [
                    _readback_variant(staged, policy, salt, execution)
                    for salt in READBACK_SALTS
                ]

    for dequant, mm in _consumers(catalogue, reached).items():
        if "numeric_limit" in reached[mm]:
            continue
        for profile in policy["chained_absolute_tolerance"]:
            if profile not in DEQUANT_PROFILES:
                continue
            cases.append(
                _chained_case(entries[dequant], entries[mm], profile, policy)
            )
        for variant in entries[mm].get("execution_variants", ()):
            for profile in MM_PROFILES:
                cases.append(
                    _chained_case(
                        entries[dequant],
                        entries[mm],
                        profile,
                        policy,
                        variant["id"],
                        LEGACY_SALT,
                    )
                )

    return {
        "schema_version": 1,
        "product": "rocm-npu-fastflowlm-kernel-recipe",
        "target": "npu2",
        "generator_version": "kernel-recipe-generator-v1",
        "implementations": implementations,
        "coverage_limits": limits,
        "cases": cases,
    }


def attest_recipe_generators(recipe, materialize):
    """Pins every generated source and oracle to the bytes it must produce.

    `materialize(specification, expected_size)` returns bytes. Keeping the
    callback outside this module lets pure tests use a bounded stand-in while
    asset production calls the installed recipe_generators implementation.
    """
    seen = {}

    def attest(specification, expected_size):
        key = json.dumps(specification, sort_keys=True, separators=(",", ":"))
        prior = seen.get(key)
        if prior is not None and prior != expected_size:
            raise RecipeError(
                "one generator specification is used with inconsistent sizes: "
                f"{prior} and {expected_size}"
            )
        seen[key] = expected_size
        data = materialize(specification, expected_size)
        if len(data) != expected_size:
            raise RecipeError(
                f"generator {specification['id']} produced {len(data)} bytes; "
                f"expected {expected_size}"
            )
        specification["sha256"] = hashlib.sha256(data).hexdigest()

    for case in recipe["cases"]:
        expected_sizes = []
        buffers_by_identity = {}
        for stage in case["stages"]:
            for buffer in stage["buffers"]:
                buffers_by_identity[(stage["name"], buffer["name"])] = buffer
                source = buffer.get("source")
                if source is not None and "stage_output" not in source:
                    attest(source, buffer["size"])
                expected = buffer.get("expected")
                if expected is not None:
                    expected_sizes.append(buffer["size"])
                    attest(expected, buffer["size"])
                preserved = buffer.get("preserved")
                if preserved is not None:
                    attest(preserved, buffer["size"])
        if len(expected_sizes) != 1:
            raise RecipeError(
                f"case {case['name']} must declare exactly one expected output"
            )
        for variant in case.get("readback_variants", ()):
            buffer = buffers_by_identity.get(
                (variant["stage"], variant["buffer"])
            )
            if buffer is None:
                raise RecipeError(
                    f"case {case['name']} readback names unknown buffer "
                    f"{variant['buffer']}"
                )
            attest(variant["source"], buffer["size"])
            attest(variant["expected"], expected_sizes[0])
    return recipe


def validate_recipe(recipe, schema_path):
    import jsonschema

    jsonschema.Draft202012Validator(
        json.loads(Path(schema_path).read_text(encoding="utf-8"))
    ).validate(recipe)
    implementation_ids = [entry["id"] for entry in recipe["implementations"]]
    case_names = [case["name"] for case in recipe["cases"]]
    limit_ids = [
        (limit["id"], limit["scope"]) for limit in recipe["coverage_limits"]
    ]
    if len(implementation_ids) != len(set(implementation_ids)):
        raise RecipeError("recipe implementation IDs are not unique")
    if len(case_names) != len(set(case_names)):
        raise RecipeError("recipe case names are not unique")
    if len(limit_ids) != len(set(limit_ids)):
        raise RecipeError("recipe coverage-limit identities are invalid")
    dispatched = set(implementation_ids)
    for case in recipe["cases"]:
        stage_names = []
        staged_outputs = {}
        expectation_count = 0
        for stage in case["stages"]:
            if stage["implementation"] not in dispatched:
                raise RecipeError(
                    f"{case['name']} dispatches an implementation the recipe omits"
                )
            stage_name = stage["name"]
            stage_names.append(stage_name)
            buffer_names = [buffer["name"] for buffer in stage["buffers"]]
            if len(buffer_names) != len(set(buffer_names)):
                raise RecipeError(
                    f"{case['name']}/{stage_name} buffer names are not unique"
                )
            for buffer in stage["buffers"]:
                if buffer["direction"] == "output":
                    staged_outputs[f"{stage_name}.{buffer['name']}"] = buffer["size"]
                source = buffer.get("source", {})
                if "stage_output" in source:
                    alias = source["stage_output"]
                    if (
                        alias not in staged_outputs
                        or staged_outputs[alias] != buffer["size"]
                    ):
                        raise RecipeError(
                            f"{case['name']} has an invalid stage-output alias: {alias}"
                        )
                expectation_count += int("expected" in buffer)
        if len(stage_names) != len(set(stage_names)):
            raise RecipeError(f"{case['name']} stage names are not unique")
        if expectation_count != 1:
            raise RecipeError(
                f"{case['name']} must declare exactly one expected output"
            )
    covered = {
        stage["implementation"]
        for case in recipe["cases"]
        if not case["name"].endswith("zero-baseline")
        for stage in case["stages"]
    }
    if not covered <= dispatched:
        raise RecipeError("a case dispatches an implementation the recipe omits")
    return {
        "implementations": len(recipe["implementations"]),
        "cases": len(recipe["cases"]),
        "chained": sum(1 for case in recipe["cases"] if len(case["stages"]) > 1),
        "numeric_coverage": len(covered),
        "coverage_limits": len(recipe["coverage_limits"]),
    }
