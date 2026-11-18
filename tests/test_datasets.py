"""Tests for dataset registry and loading from CSV."""

from pathlib import Path

import pytest

from impenetra.datasets import (
    DATASET_REGISTRY,
    DATASETS,
    get_dataset_info,
    list_datasets,
    load_all_datasets,
    load_guardrail_dataset,
    parse_boolean,
)


def test_registry_has_expected_datasets():
    assert "on_charter" in DATASET_REGISTRY
    assert "infused_on_charter" in DATASET_REGISTRY
    assert "adversarial" in DATASET_REGISTRY
    assert "off_charter" in DATASET_REGISTRY
    assert "infused_off_charter" in DATASET_REGISTRY
    assert "intent_manipulation" in DATASET_REGISTRY
    assert "xstest" in DATASET_REGISTRY

    # Obsolete datasets must be removed
    assert "clinc150" not in DATASET_REGISTRY
    assert "toxic_chat" not in DATASET_REGISTRY


def test_list_datasets():
    datasets = list_datasets()
    assert len(datasets) == 7
    names = [d.name for d in datasets]
    assert "on_charter" in names
    assert "infused_on_charter" in names
    assert "adversarial" in names
    assert "off_charter" in names
    assert "infused_off_charter" in names
    assert "intent_manipulation" in names
    assert "xstest" in names


def test_get_dataset_info():
    info = get_dataset_info("on_charter")
    assert info.name == "on_charter"
    assert info.path.exists()
    assert info.task == "culinary_intent"

    info_infused = get_dataset_info("infused_on_charter")
    assert info_infused.name == "infused_on_charter"
    assert info_infused.path.exists()
    assert info_infused.task == "infused_culinary_intent"

    info_infused_off = get_dataset_info("infused_off_charter")
    assert info_infused_off.name == "infused_off_charter"
    assert info_infused_off.path.exists()
    assert info_infused_off.task == "infused_utility_theft"

    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset_info("nonexistent_dataset")


def test_load_on_charter_dataset():
    rows = load_guardrail_dataset("on_charter")
    assert len(rows) == 12
    # All rows should have expected_blocked = False and type = positive
    assert all(r["expected_blocked"] is False for r in rows)
    assert all(r["type"] == "positive" for r in rows)

    # First row is a pure culinary query
    assert "honeycrisp apple" in rows[0]["prompt"]
    assert rows[0]["type"] == "positive"


def test_load_infused_on_charter_dataset():
    rows = load_guardrail_dataset("infused_on_charter")
    assert len(rows) == 12
    # All rows should have expected_blocked = False and type = positive
    assert all(r["expected_blocked"] is False for r in rows)
    assert all(r["type"] == "positive" for r in rows)

    # Verify that prompts look like negative ones (jailbreaks, injections, toxins, etc.)
    prompts_text = " ".join(r["prompt"].lower() for r in rows)
    assert "developer mode" in prompts_text
    assert "dan" in prompts_text
    assert "override" in prompts_text
    assert "solanine" in prompts_text
    assert "salmonella" in prompts_text


def test_load_adversarial_dataset():
    rows = load_guardrail_dataset("adversarial")
    assert len(rows) == 15
    # All adversarial probes should be expected_blocked = True and negative
    assert all(r["expected_blocked"] is True for r in rows)
    assert all(r["type"] == "negative" for r in rows)


def test_load_off_charter_dataset():
    rows = load_guardrail_dataset("off_charter")
    assert len(rows) == 8
    # All off-charter queries should be expected_blocked = True and negative
    assert all(r["expected_blocked"] is True for r in rows)
    assert all(r["type"] == "negative" for r in rows)


def test_load_infused_off_charter_dataset():
    rows = load_guardrail_dataset("infused_off_charter")
    assert len(rows) == 10
    # All infused off-charter queries should be expected_blocked = True and negative
    assert all(r["expected_blocked"] is True for r in rows)
    assert all(r["type"] == "negative" for r in rows)

    # Verify that prompts look like culinary / positive ones on the surface
    prompts_text = " ".join(r["prompt"].lower() for r in rows)
    assert "bakery" in prompts_text
    assert "restaurant" in prompts_text
    assert "chef" in prompts_text
    assert "bread" in prompts_text


def test_load_intent_manipulation_dataset():
    rows = load_guardrail_dataset("intent_manipulation")
    assert len(rows) == 15
    # All intent manipulation probes should be expected_blocked = True
    assert all(r["expected_blocked"] is True for r in rows)
    assert all(r["type"] == "negative" for r in rows)


def test_load_xstest_dataset():
    rows = load_guardrail_dataset("xstest")
    assert len(rows) == 32
    assert all(r["dataset"] == "xstest" for r in rows)
    pos_rows = [r for r in rows if not r["expected_blocked"]]
    neg_rows = [r for r in rows if r["expected_blocked"]]
    assert len(pos_rows) == 16
    assert len(neg_rows) == 16
    assert all(r["type"] == "positive" for r in pos_rows)
    assert all(r["type"] == "negative" for r in neg_rows)


def test_load_all_datasets():
    all_data = load_all_datasets()
    assert set(all_data.keys()) == {
        "on_charter",
        "infused_on_charter",
        "adversarial",
        "off_charter",
        "infused_off_charter",
        "intent_manipulation",
        "xstest",
    }
    assert len(all_data["on_charter"]) == 12
    assert len(all_data["infused_on_charter"]) == 12
    assert len(all_data["adversarial"]) == 15
    assert len(all_data["off_charter"]) == 8
    assert len(all_data["infused_off_charter"]) == 10
    assert len(all_data["intent_manipulation"]) == 15
    assert len(all_data["xstest"]) == 32


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset or file"):
        load_guardrail_dataset("nonexistent_dataset")


def test_custom_csv_loading(tmp_path: Path):
    custom_csv = tmp_path / "custom.csv"
    custom_csv.write_text(
        "prompt,expected_blocked,type,subtype\n"
        '"Test positive query",false,positive,test_sub\n'
        '"Test negative attack",true,negative,test_adv\n',
        encoding="utf-8",
    )
    rows = load_guardrail_dataset(custom_csv)
    assert len(rows) == 2
    assert rows[0]["expected_blocked"] is False
    assert rows[0]["type"] == "positive"
    assert rows[1]["expected_blocked"] is True
    assert rows[1]["type"] == "negative"


def test_parse_boolean_strict():
    assert parse_boolean(True) is True
    assert parse_boolean(False) is False
    assert parse_boolean("true") is True
    assert parse_boolean("TRUE") is True
    assert parse_boolean("True") is True
    assert parse_boolean("false") is False
    assert parse_boolean("FALSE") is False
    assert parse_boolean("False") is False

    # Loose strings must strictly fail
    for invalid in ["1", "0", "yes", "no", "block", "allow", "negative", "invalid_val", "", 42]:
        with pytest.raises(ValueError, match="Invalid boolean value"):
            parse_boolean(invalid)


def test_missing_required_column_raises(tmp_path: Path):
    bad_csv = tmp_path / "missing_col.csv"
    bad_csv.write_text("prompt,type,subtype\n'Hello',positive,test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_guardrail_dataset(bad_csv)


def test_empty_prompt_raises(tmp_path: Path):
    bad_csv = tmp_path / "empty_prompt.csv"
    bad_csv.write_text(
        'prompt,expected_blocked,type,subtype\n"",false,positive,test\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="'prompt' cannot be empty"):
        load_guardrail_dataset(bad_csv)


def test_invalid_type_raises(tmp_path: Path):
    bad_csv = tmp_path / "invalid_type.csv"
    bad_csv.write_text(
        "prompt,expected_blocked,type,subtype\n'Hello',false,unknown_type,test\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid 'type'"):
        load_guardrail_dataset(bad_csv)

    for disallow in ["super_positive", "super_negative"]:
        csv_p = tmp_path / f"invalid_{disallow}.csv"
        csv_p.write_text(
            f"prompt,expected_blocked,type,subtype\n'Test',false,{disallow},test\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Invalid 'type'"):
            load_guardrail_dataset(csv_p)


def test_type_and_expected_blocked_mismatch_raises(tmp_path: Path):
    # positive with expected_blocked=true must fail
    bad_csv1 = tmp_path / "mismatch1.csv"
    bad_csv1.write_text(
        "prompt,expected_blocked,type,subtype\n'Apple',true,positive,test\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must have expected_blocked=false"):
        load_guardrail_dataset(bad_csv1)

    # negative with expected_blocked=false must fail
    bad_csv2 = tmp_path / "mismatch2.csv"
    bad_csv2.write_text(
        "prompt,expected_blocked,type,subtype\n'Bomb',false,negative,test\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must have expected_blocked=true"):
        load_guardrail_dataset(bad_csv2)


def test_datasets_array_and_loader():
    assert DATASETS == [
        "on_charter",
        "infused_on_charter",
        "adversarial",
        "off_charter",
        "infused_off_charter",
        "intent_manipulation",
        "xstest",
    ]

    # Verify that load_guardrail_dataset loads every dataset from the array
    for name in DATASETS:
        rows = load_guardrail_dataset(name)
        assert len(rows) > 0
        assert all(
            {"prompt", "expected_blocked", "type", "subtype"}.issubset(r.keys()) for r in rows
        )
