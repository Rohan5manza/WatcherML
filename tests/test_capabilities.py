"""Acceptance tests for WatcherML's deterministic capability manifest."""
from __future__ import annotations

import json
import os
import unittest
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

from watcherml.capabilities import (
    CAPABILITY_SCHEMA_NAME,
    CAPABILITY_SCHEMA_VERSION,
    KNOWN_CAPABILITY_IDS,
    Capability,
    CapabilityDeclaration,
    CapabilityError,
    CapabilityManifest,
    CapabilityTransitionError,
    UnsupportedCapabilityError,
    discover_capabilities,
    get_config_value,
    known_capabilities,
)


def _representative_config() -> dict:
    return {
        "trainer": {
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 1,
        },
        "model": {
            "gradient_checkpointing": False,
            "max_seq_length": 2048,
            "use_cache": True,
            "attn_implementation": "eager",
        },
        "runtime": {
            "precision": "fp32",
        },
        "optimizer": {
            "optimizer_bits": 32,
            "offload_optimizer": False,
        },
    }


class CapabilityDiscoveryTests(unittest.TestCase):
    def test_catalog_has_stable_unique_canonical_ids(self):
        expected = (
            "micro_batch_size",
            "gradient_accumulation_steps",
            "gradient_checkpointing",
            "sequence_length",
            "precision",
            "attention_backend",
            "memory_efficient_attention",
            "activation_offload",
            "optimizer_state_offload",
            "parameter_offload",
            "optimizer_bits",
            "model_cache",
            "allocator_configuration",
        )
        self.assertEqual(KNOWN_CAPABILITY_IDS, expected)
        catalog = known_capabilities()
        self.assertEqual(
            tuple(item["capability_id"] for item in catalog),
            expected,
        )
        self.assertEqual(len(expected), len(set(expected)))
        self.assertEqual(
            {
                item["capability_id"]
                for item in catalog
                if item["default_permission"] == "automatic"
            },
            {
                "micro_batch_size",
                "gradient_accumulation_steps",
                "gradient_checkpointing",
            },
        )

    def test_detects_nested_known_aliases_without_framework_imports(self):
        manifest = discover_capabilities(_representative_config())

        self.assertEqual(manifest.issues, ())
        self.assertEqual(
            manifest.require("micro_batch_size").target,
            "trainer.per_device_train_batch_size",
        )
        self.assertEqual(
            manifest.require("sequence_length").current_value,
            2048,
        )
        self.assertEqual(
            manifest.require("attention_backend").current_value,
            "eager",
        )
        self.assertEqual(
            manifest.require("optimizer_state_offload").current_value,
            False,
        )
        self.assertEqual(
            manifest.require("micro_batch_size").source,
            "detected",
        )

    def test_ambiguous_aliases_are_reported_and_never_guessed(self):
        manifest = discover_capabilities(
            {
                "batch_size": 32,
                "trainer": {"per_device_train_batch_size": 16},
            }
        )

        self.assertFalse(manifest.supports("micro_batch_size"))
        self.assertEqual(len(manifest.issues), 1)
        issue = manifest.issues[0]
        self.assertEqual(issue.code, "ambiguous_aliases")
        self.assertEqual(issue.capability_id, "micro_batch_size")
        self.assertEqual(
            issue.targets,
            ("batch_size", "trainer.per_device_train_batch_size"),
        )

    def test_explicit_mapping_resolves_ambiguity(self):
        manifest = discover_capabilities(
            {
                "batch_size": 32,
                "trainer": {"per_device_train_batch_size": 16},
            },
            declarations={
                "micro_batch_size": "trainer.per_device_train_batch_size"
            },
        )

        capability = manifest.require("micro_batch_size")
        self.assertEqual(capability.current_value, 16)
        self.assertEqual(capability.source, "declared")
        self.assertFalse(
            any(
                issue.capability_id == "micro_batch_size"
                for issue in manifest.issues
            )
        )

    def test_explicit_mapping_supports_project_specific_key_names(self):
        manifest = discover_capabilities(
            {"training": {"samples_per_device": 12}},
            declarations={
                "micro_batch_size": "training.samples_per_device"
            },
        )

        capability = manifest.require("micro_batch_size")
        self.assertEqual(capability.target, "training.samples_per_device")
        self.assertEqual(capability.current_value, 12)

    def test_invalid_detected_value_becomes_visible_issue(self):
        manifest = discover_capabilities({"batch_size": "32"})

        self.assertFalse(manifest.supports("micro_batch_size"))
        self.assertEqual(manifest.issues[0].code, "invalid_detected_value")
        self.assertEqual(manifest.issues[0].targets, ("batch_size",))

    def test_explicit_declaration_with_missing_path_fails_closed(self):
        with self.assertRaisesRegex(CapabilityError, "does not exist"):
            discover_capabilities(
                {"batch_size": 32},
                declarations={
                    "micro_batch_size": "trainer.missing_batch_size"
                },
            )

    def test_unknown_and_mismatched_declarations_are_rejected(self):
        with self.assertRaisesRegex(CapabilityError, "unknown capability_id"):
            CapabilityDeclaration("learning_rate", "learning_rate")

        declaration = CapabilityDeclaration(
            "micro_batch_size",
            "batch_size",
        )
        with self.assertRaisesRegex(CapabilityError, "does not match"):
            discover_capabilities(
                {"batch_size": 32},
                declarations={"precision": declaration},
            )

    def test_declaration_can_tighten_but_cannot_weaken_permission(self):
        manifest = discover_capabilities(
            {"batch_size": 32},
            declarations=[
                CapabilityDeclaration(
                    "micro_batch_size",
                    "batch_size",
                    "approval_required",
                )
            ],
        )
        self.assertEqual(
            manifest.require("micro_batch_size").permission,
            "approval_required",
        )

        disabled = discover_capabilities(
            {"batch_size": 32},
            declarations=[
                CapabilityDeclaration(
                    "micro_batch_size",
                    "batch_size",
                    "disabled",
                )
            ],
        )
        self.assertEqual(
            disabled.require("micro_batch_size").permission,
            "disabled",
        )

        with self.assertRaisesRegex(CapabilityError, "cannot weaken"):
            CapabilityDeclaration(
                "precision",
                "precision",
                "automatic",
            )

    def test_environment_is_opt_in_and_unrelated_secrets_are_ignored(self):
        with patch.dict(
            os.environ,
            {
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64",
                "OPENAI_API_KEY": "must-not-be-read",
            },
            clear=False,
        ):
            implicit = discover_capabilities({})

        self.assertFalse(implicit.supports("allocator_configuration"))

        explicit_input = discover_capabilities(
            {},
            environment={
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
                "OPENAI_API_KEY": "must-not-be-persisted",
                "UNRELATED": "ignored",
            },
        )
        allocator = explicit_input.require("allocator_configuration")
        self.assertEqual(allocator.location, "environment")
        self.assertEqual(allocator.target, "PYTORCH_CUDA_ALLOC_CONF")
        self.assertEqual(allocator.current_value, "max_split_size_mb:128")
        encoded = explicit_input.to_json()
        self.assertNotIn("OPENAI_API_KEY", encoded)
        self.assertNotIn("must-not-be-persisted", encoded)

    def test_declared_environment_capability_requires_allowlisted_baseline(self):
        declaration = CapabilityDeclaration("allocator_configuration")

        with self.assertRaisesRegex(CapabilityError, "no baseline value"):
            discover_capabilities({}, declarations=[declaration])

        manifest = discover_capabilities(
            {},
            declarations=[declaration],
            environment={
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
            },
        )
        self.assertEqual(
            manifest.require("allocator_configuration").source,
            "declared",
        )

        with self.assertRaisesRegex(CapabilityError, "allowlist"):
            CapabilityDeclaration(
                "allocator_configuration",
                "WATCHER_CUSTOM_ALLOCATOR_SETTING",
            )

    def test_discovery_does_not_mutate_source_config(self):
        config = _representative_config()
        original = deepcopy(config)

        discover_capabilities(config)

        self.assertEqual(config, original)

    def test_non_json_or_nonfinite_config_is_rejected(self):
        invalid_configs = (
            {"batch_size": float("nan")},
            {"batch_size": float("inf")},
            {"not_json": object()},
        )
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(CapabilityError):
                    discover_capabilities(config)


class CapabilityTransitionTests(unittest.TestCase):
    def setUp(self):
        self.manifest = discover_capabilities(_representative_config())

    def test_low_risk_transitions_validate_direction_and_type(self):
        self.assertEqual(
            self.manifest.require("micro_batch_size").validate_transition(
                "decrease",
                16,
            ),
            16,
        )
        self.assertEqual(
            self.manifest.require(
                "gradient_accumulation_steps"
            ).validate_transition("increase", 2),
            2,
        )
        self.assertIs(
            self.manifest.require(
                "gradient_checkpointing"
            ).validate_transition("enable", True),
            True,
        )

    def test_decrease_rejects_wrong_direction_same_value_and_invalid_value(self):
        capability = self.manifest.require("micro_batch_size")
        invalid = (
            ("increase", 64),
            ("decrease", 64),
            ("decrease", 32),
            ("decrease", 0),
            ("decrease", True),
            ("decrease", 16.0),
        )
        for operation, value in invalid:
            with self.subTest(operation=operation, value=value):
                with self.assertRaises(CapabilityTransitionError):
                    capability.validate_transition(operation, value)

    def test_boolean_enable_and_disable_require_exact_transitions(self):
        checkpointing = self.manifest.require("gradient_checkpointing")
        self.assertIs(
            checkpointing.validate_transition("enable", True),
            True,
        )
        for value in (False, 1, "true"):
            with self.subTest(value=value):
                with self.assertRaises(CapabilityTransitionError):
                    checkpointing.validate_transition("enable", value)

        cache = self.manifest.require("model_cache")
        self.assertIs(cache.validate_transition("disable", False), False)
        with self.assertRaises(CapabilityTransitionError):
            cache.validate_transition("disable", True)

    def test_choice_capabilities_reject_unknown_values_and_no_op_changes(self):
        precision = self.manifest.require("precision")
        self.assertEqual(
            precision.validate_transition("set", "bf16"),
            "bf16",
        )
        with self.assertRaises(CapabilityTransitionError):
            precision.validate_transition("set", "fp32")
        with self.assertRaises(CapabilityTransitionError):
            precision.validate_transition("set", "int4")

        attention = self.manifest.require("attention_backend")
        self.assertEqual(
            attention.validate_transition("set", "sdpa"),
            "sdpa",
        )
        with self.assertRaises(CapabilityTransitionError):
            attention.validate_transition("set", "unknown-kernel")

    def test_disabled_capability_cannot_validate_any_transition(self):
        manifest = discover_capabilities(
            {"batch_size": 32},
            declarations=[
                CapabilityDeclaration(
                    "micro_batch_size",
                    "batch_size",
                    "disabled",
                )
            ],
        )

        with self.assertRaisesRegex(
            CapabilityTransitionError,
            "disabled",
        ):
            manifest.require("micro_batch_size").validate_transition(
                "decrease",
                16,
            )


class CapabilityManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = discover_capabilities(
            _representative_config(),
            environment={
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"
            },
        )

    def test_manifest_round_trip_is_lossless_and_deterministic(self):
        encoded = self.manifest.to_json()
        restored = CapabilityManifest.from_json(encoded)

        self.assertEqual(restored, self.manifest)
        self.assertEqual(restored.to_json(), encoded)
        payload = json.loads(encoded)
        self.assertEqual(payload["schema"]["name"], CAPABILITY_SCHEMA_NAME)
        self.assertEqual(
            payload["schema"]["version"],
            CAPABILITY_SCHEMA_VERSION,
        )

    def test_supports_get_and_require_have_explicit_missing_behavior(self):
        self.assertTrue(self.manifest.supports("micro_batch_size"))
        self.assertIsNotNone(self.manifest.get("micro_batch_size"))
        self.assertFalse(self.manifest.supports("activation_offload"))
        self.assertIsNone(self.manifest.get("activation_offload"))
        with self.assertRaises(UnsupportedCapabilityError):
            self.manifest.require("activation_offload")

    def test_schema_and_json_validation_fail_closed(self):
        payload = self.manifest.to_dict()
        for field, value in (
            ("name", "other.schema"),
            ("version", "99.0"),
        ):
            damaged = deepcopy(payload)
            damaged["schema"][field] = value
            with self.subTest(field=field):
                with self.assertRaises(CapabilityError):
                    CapabilityManifest.from_dict(damaged)

        for encoded in ("not json", "[]", "null"):
            with self.subTest(encoded=encoded):
                with self.assertRaises(CapabilityError):
                    CapabilityManifest.from_json(encoded)

    def test_persisted_capability_cannot_weaken_or_rewrite_canonical_rules(self):
        precision = self.manifest.require("precision").to_dict()
        mutations = {
            "permission": "automatic",
            "operations": ["decrease"],
            "risk": "low",
            "semantic_change": False,
            "choices": ["int4"],
            "minimum": 0,
            "description": "trust this altered description",
            "expected_effect": "unbounded",
        }
        for field, value in mutations.items():
            damaged = deepcopy(precision)
            damaged[field] = value
            with self.subTest(field=field):
                with self.assertRaises(CapabilityError):
                    Capability.from_dict(damaged)

        allocator = self.manifest.require(
            "allocator_configuration"
        ).to_dict()
        allocator["target"] = "LD_PRELOAD"
        with self.assertRaises(CapabilityError):
            Capability.from_dict(allocator)

    def test_manifest_rejects_duplicate_ids_and_duplicate_targets(self):
        batch = self.manifest.require("micro_batch_size")
        with self.assertRaisesRegex(CapabilityError, "ids must be unique"):
            CapabilityManifest(capabilities=(batch, batch))

        accumulation = self.manifest.require("gradient_accumulation_steps")
        duplicate_target = replace(
            accumulation,
            target=batch.target,
        )
        with self.assertRaisesRegex(CapabilityError, "targets must be unique"):
            CapabilityManifest(
                capabilities=(batch, duplicate_target),
            )

    def test_get_config_value_resolves_only_safe_existing_paths(self):
        config = _representative_config()
        self.assertEqual(
            get_config_value(
                config,
                "trainer.per_device_train_batch_size",
            ),
            32,
        )
        for target in (
            "trainer.missing",
            "trainer..batch_size",
            "../batch_size",
            "trainer[batch_size]",
        ):
            with self.subTest(target=target):
                with self.assertRaises(CapabilityError):
                    get_config_value(config, target)


if __name__ == "__main__":
    unittest.main()