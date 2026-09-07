"""Deterministic corpus generation and mutation for benchmarks."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class CorpusProfile:
    """Profile for corpus generation."""

    name: str
    file_count: int
    avg_bytes_per_file: int
    language_mix: Dict[str, float] = field(
        default_factory=lambda: {
            "java": 0.4,
            "python": 0.2,
            "markdown": 0.15,
            "yaml": 0.1,
            "json": 0.1,
            "other": 0.05,
        }
    )
    include_tests: bool = True
    include_docs: bool = True
    include_ignored_dirs: bool = True
    seed: int = 42


# Predefined corpus profiles
CORPUS_TINY = CorpusProfile(name="tiny", file_count=20, avg_bytes_per_file=500, seed=1)
CORPUS_SMALL = CorpusProfile(name="small", file_count=100, avg_bytes_per_file=1500, seed=2)
CORPUS_MEDIUM = CorpusProfile(name="medium", file_count=1000, avg_bytes_per_file=2000, seed=3)
CORPUS_LARGE = CorpusProfile(name="large", file_count=10000, avg_bytes_per_file=3000, seed=4)


@dataclass
class CorpusManifest:
    """Manifest describing a generated corpus."""

    profile: str
    root: Path
    files_total: int
    indexable_files: int
    bytes_total: int
    query_terms: List[str] = field(default_factory=list)


@dataclass
class MutationPlan:
    """Plan for corpus mutation."""

    add_files: int = 0
    change_files: int = 0
    delete_files: int = 0
    preserve_size: bool = False
    force_mtime_tick: bool = False


# Mutation presets
MUTATION_NOOP = MutationPlan()
MUTATION_ONE_CHANGED = MutationPlan(change_files=1)
MUTATION_ONE_ADDED_ONE_DELETED = MutationPlan(add_files=1, delete_files=1)
MUTATION_ONE_PERCENT_CHURN = MutationPlan(change_files=1)  # Caller should scale


# Templates for realistic code content
JAVA_CLASS_TEMPLATE = """package com.example.{package_name};

import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * {class_name} - Service for managing business logic.
 */
@Service
public class {class_name} {{
    private final {repository_name} repository;
    private final Map<String, Object> config;

    @Autowired
    public {class_name}({repository_name} repository, Map<String, Object> config) {{
        this.repository = repository;
        this.config = config;
    }}

    /**
     * Process invoice with given parameters.
     */
    public {return_type} process{invoice_type}({param_type} request) {{
        // Validate input
        if (request == null) {{
            throw new IllegalArgumentException("Request cannot be null");
        }}

        // Fetch existing records
        Optional<{entity_type}> existing = repository.findById(request.getId());
        if (existing.isEmpty()) {{
            return {return_type}.failure("Not found");
        }}

        // Apply business logic
        {entity_type} entity = existing.get();
        entity.setStatus("{status_value}");
        entity.setModifiedAt(System.currentTimeMillis());

        // Save and return
        repository.save(entity);
        return {return_type}.success(entity);
    }}

    /**
     * Rollback a previously completed operation.
     */
    public void rollback(Long id, String reason) {{
        {entity_type} entity = repository.findById(id)
            .orElseThrow(() -> new RuntimeException("Entity not found: " + id));
        entity.setStatus("ROLLBACK");
        entity.setRollbackReason(reason);
        entity.setRolledBackAt(System.currentTimeMillis());
        repository.save(entity);
    }}

    /**
     * Reconcile ledger entries for tenant.
     */
    public List<ReconciliationResult> reconcileLedger(String tenantId) {{
        List<{entity_type}> entries = repository.findByTenantId(tenantId);
        return entries.stream()
            .map(e -> new ReconciliationResult(e.getId(), e.getAmount(), e.getStatus()))
            .toList();
    }}
}}
"""

JAVA_TEST_TEMPLATE = """package com.example.{package_name};

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class {class_name}Test {{

    @Mock
    private {repository_name} repository;

    @InjectMocks
    private {class_name} service;

    @Test
    void testProcess{invoice_type}Success() {{
        // Given
        {entity_type} entity = new {entity_type}();
        entity.setId(1L);
        when(repository.findById(1L)).thenReturn(Optional.of(entity));
        when(repository.save(any())).thenReturn(entity);

        // When
        var result = service.process{invoice_type}(new Request(1L));

        // Then
        assertTrue(result.isSuccess());
        verify(repository, times(1)).save(any());
    }}

    @Test
    void testRollback() {{
        // Given
        {entity_type} entity = new {entity_type}();
        entity.setId(1L);
        when(repository.findById(1L)).thenReturn(Optional.of(entity));

        // When
        service.rollback(1L, "Test rollback reason");

        // Then
        verify(repository).save(argThat(e ->
            "ROLLBACK".equals(((Entity)e).getStatus())
        ));
    }}
}}
"""

PYTHON_SERVICE_TEMPLATE = '''"""Service for managing {module_name} business logic."""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger("{module_name}")


@dataclass
class {class_name}Service:
    """Service class for {module_name} operations."""

    def __init__(self, repository, config: Dict[str, Any]):
        self.repository = repository
        self.config = config
        self.logger = logging.getLogger("{module_name}")

    def process_{invoice_name}(self, request: Dict) -> Dict:
        """Process an invoice request."""
        if not request:
            raise ValueError("Request cannot be empty")

        existing = self.repository.find_by_id(request["id"])
        if not existing:
            return {{"status": "error", "message": "Not found"}}

        existing["status"] = "{status_value}"
        existing["modified_at"] = self._timestamp()
        self.repository.save(existing)

        self.logger.info(f"Processed invoice {{request['id']}}")
        return {{"status": "success", "data": existing}}

    def rollback_operation(self, operation_id: int, reason: str) -> None:
        """Rollback a previously completed operation."""
        entity = self.repository.find_by_id(operation_id)
        if not entity:
            raise ValueError(f"Entity not found: {{operation_id}}")

        entity["status"] = "ROLLBACK"
        entity["rollback_reason"] = reason
        entity["rolled_back_at"] = self._timestamp()
        self.repository.save(entity)

    def reconcile_ledger(self, tenant_id: str) -> List[Dict]:
        """Reconcile ledger entries for a tenant."""
        entries = self.repository.find_by_tenant_id(tenant_id)
        results = []
        for entry in entries:
            results.append({{
                "id": entry["id"],
                "amount": entry["amount"],
                "status": entry["status"]
            }})
        return results

    @staticmethod
    def _timestamp() -> int:
        """Get current timestamp."""
        import time
        return int(time.time())
'''


MARKDOWN_DOC_TEMPLATE = """# {title}

{description}

## Overview

This document describes the {module_name} functionality and its usage.

## Features

- Feature A: Processing of {invoice_type} requests
- Feature B: Automatic rollback on failure
- Feature C: Ledger reconciliation

## Usage

```python
service = {class_name}Service(repository, config)
result = service.process_{invoice_name}({{"id": 1}})
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| timeout | int | 30 | Request timeout in seconds |
| retries | int | 3 | Number of retry attempts |

## Examples

### Processing an Invoice

```python
request = {{"id": 1, "type": "{invoice_type}", "amount": 100.00}}
result = service.process_{invoice_name}(request)
```

### Rolling Back

```python
service.rollback_operation(1, "Customer requested cancellation")
```

## API Reference

### process_{invoice_name}(request)

Process a {invoice_type} request.

**Parameters:**
- `request` (dict): The request object

**Returns:**
- `dict`: Result with status and data

### rollback_operation(operation_id, reason)

Rollback a completed operation.

**Parameters:**
- `operation_id` (int): The operation ID
- `reason` (str): Reason for rollback
"""


YAML_CONFIG_TEMPLATE = """# Configuration for {app_name}
application:
  name: {app_name}
  version: 1.0.0
  environment: {environment}

database:
  host: localhost
  port: 5432
  name: {db_name}
  pool_size: 10
  timeout: 30

services:
  {service_name}:
    enabled: true
    timeout: {timeout}
    retry_attempts: 3
    circuit_breaker:
      enabled: true
      threshold: 5
      timeout: 60

logging:
  level: INFO
  format: json
  outputs:
    - console
    - file: /var/log/{app_name}.log
"""


QUERY_TERMS = [
    "PaymentController",
    "PaymentRollbackService",
    "tenant ledger reconciliation",
    "generated invoice",
    "invoice processing rollback",
    "reconciliation service",
    "tax calculation",
    "payment processing",
    "ledger entry",
    "invoice controller",
]


def _deterministic_random(seed: int) -> random.Random:
    """Create a deterministic random generator."""
    return random.Random(seed)


def _hash_text(text: str, seed: int) -> str:
    """Create a deterministic hash from text and seed."""
    combined = f"{text}:{seed}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _generate_java_content(rng: random.Random, idx: int, pkg: str) -> str:
    """Generate Java class content."""
    class_name = f"{pkg.title().replace('_', '')}{idx:04d}Service"
    repo_name = f"{pkg.title().replace('_', '')}{idx:04d}Repository"
    invoice_types = ["Invoice", "Payment", "Transaction", "Ledger"]
    statuses = ["COMPLETED", "PENDING", "PROCESSING", "APPROVED"]
    invoice_type = rng.choice(invoice_types)
    status = rng.choice(statuses)
    pkg_path = pkg.replace("_", "/")

    return JAVA_CLASS_TEMPLATE.format(
        package_name=pkg_path,
        class_name=class_name,
        repository_name=repo_name,
        return_type="Result",
        param_type="InvoiceRequest",
        entity_type="InvoiceEntity",
        status_value=status,
        invoice_type=invoice_type,
    )


def _generate_java_test(rng: random.Random, idx: int, pkg: str) -> str:
    """Generate Java test class content."""
    class_name = f"{pkg.title().replace('_', '')}{idx:04d}Service"
    repo_name = f"{pkg.title().replace('_', '')}{idx:04d}Repository"
    invoice_types = ["Invoice", "Payment", "Transaction"]
    invoice_type = rng.choice(invoice_types)
    pkg_path = pkg.replace("_", "/")

    return JAVA_TEST_TEMPLATE.format(
        package_name=pkg_path,
        class_name=class_name,
        repository_name=repo_name,
        invoice_type=invoice_type,
        entity_type="Entity",
    )


def _generate_python_content(rng: random.Random, idx: int, module: str) -> str:
    """Generate Python module content."""
    class_name = f"{module.title().replace('_', '')}{idx:04d}"
    invoice_names = ["invoice", "payment", "transaction", "ledger"]
    statuses = ["completed", "pending", "processing"]

    return PYTHON_SERVICE_TEMPLATE.format(
        module_name=module,
        class_name=class_name,
        invoice_name=rng.choice(invoice_names),
        status_value=rng.choice(statuses),
    )


def _generate_markdown_content(rng: random.Random, idx: int) -> str:
    """Generate Markdown documentation content."""
    titles = [
        "Invoice Processing Guide",
        "Payment Service Documentation",
        "Ledger Reconciliation",
        "Tax Calculation Reference",
    ]
    descriptions = [
        "Comprehensive guide to invoice processing workflows.",
        "Payment processing service documentation and API reference.",
        "Instructions for reconciling ledger entries across tenants.",
        "Tax calculation rules and configuration options.",
    ]
    modules = ["invoice", "payment", "ledger", "tax"]
    invoice_types_list = ["Invoice", "Payment", "Transaction"]

    return MARKDOWN_DOC_TEMPLATE.format(
        title=rng.choice(titles),
        description=rng.choice(descriptions),
        module_name=rng.choice(modules),
        class_name=f"Service{idx:04d}",
        invoice_name=rng.choice(invoice_types_list).lower(),
        invoice_type=rng.choice(invoice_types_list),
    )


def _generate_yaml_content(rng: random.Random, idx: int) -> str:
    """Generate YAML configuration content."""
    apps = ["invoice-service", "payment-api", "ledger-sync", "tax-calculator"]
    services = ["processor", "validator", "reconciler"]
    environments = ["development", "staging", "production"]

    return YAML_CONFIG_TEMPLATE.format(
        app_name=rng.choice(apps),
        service_name=rng.choice(services),
        environment=rng.choice(environments),
        db_name="appdb",
        timeout=rng.randint(10, 120),
    )


def _generate_json_content(rng: random.Random, idx: int) -> str:
    """Generate JSON data file content."""
    import json

    entries = []
    for i in range(rng.randint(5, 20)):
        entries.append(
            {
                "id": idx * 100 + i,
                "type": rng.choice(["invoice", "payment", "refund"]),
                "amount": round(rng.uniform(10, 10000), 2),
                "status": rng.choice(["completed", "pending", "failed"]),
                "tenant_id": f"tenant_{rng.randint(1, 10)}",
            }
        )
    return json.dumps({"entries": entries, "count": len(entries)}, indent=2)


def generate_corpus(root: Path, profile: CorpusProfile) -> CorpusManifest:
    """
    Generate a deterministic corpus under the given root path.

    Creates a realistic repo structure with Java, Python, Markdown, YAML, and JSON files.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    rng = _deterministic_random(profile.seed)

    # Calculate language distribution
    lang_counts: Dict[str, int] = {}
    remaining = profile.file_count
    for lang, ratio in sorted(profile.language_mix.items(), key=lambda x: -x[1]):
        if remaining <= 0:
            break
        count = int(profile.file_count * ratio)
        count = min(count, remaining)
        if count > 0:
            lang_counts[lang] = count
            remaining -= count
    if remaining > 0 and lang_counts:
        lang_counts[list(lang_counts.keys())[0]] += remaining

    # Track stats
    total_files = 0
    indexable_files = 0
    bytes_total = 0
    file_paths: List[Path] = []

    # Generate Java files (controllers, services, repositories, tests)
    java_count = lang_counts.get("java", 0)
    java_pkgs = [
        "invoice/controller",
        "invoice/service",
        "invoice/repository",
        "payment/controller",
        "payment/service",
        "payment/repository",
        "ledger/service",
        "tax/service",
    ]

    for i in range(java_count):
        pkg = rng.choice(java_pkgs)
        pkg_dir = root / "src/main/java/com/example" / pkg.replace("_", "/")
        if profile.include_tests:
            pkg_dir_tests = root / "src/test/java/com/example" / pkg.replace("_", "/")

        is_test = rng.random() < 0.3 and profile.include_tests
        if is_test:
            pkg_dir_tests.mkdir(parents=True, exist_ok=True)
            file_path = pkg_dir_tests / f"{pkg.split('/')[-1]}{total_files:05d}Test.java"
            content = _generate_java_test(rng, total_files, pkg.replace("/", "_"))
        else:
            pkg_dir.mkdir(parents=True, exist_ok=True)
            file_path = pkg_dir / f"{pkg.split('/')[-1]}{total_files:05d}.java"
            content = _generate_java_content(rng, total_files, pkg.replace("/", "_"))

        file_path.write_text(content)
        file_paths.append(file_path)
        total_files += 1
        indexable_files += 1
        bytes_total += len(content.encode())

    # Generate Python files
    py_count = lang_counts.get("python", 0)
    py_modules = ["invoice", "payment", "ledger", "tax", "common"]

    for i in range(py_count):
        module = rng.choice(py_modules)
        pkg_dir = root / module
        pkg_dir.mkdir(parents=True, exist_ok=True)
        file_path = pkg_dir / f"service_{total_files:05d}.py"
        content = _generate_python_content(rng, total_files, module)
        file_path.write_text(content)
        file_paths.append(file_path)
        total_files += 1
        indexable_files += 1
        bytes_total += len(content.encode())

    # Generate Markdown files
    md_count = lang_counts.get("markdown", 0)

    for i in range(md_count):
        docs_dir = root / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        file_path = docs_dir / f"guide-{total_files:05d}.md"
        content = _generate_markdown_content(rng, total_files)
        file_path.write_text(content)
        file_paths.append(file_path)
        total_files += 1
        indexable_files += 1
        bytes_total += len(content.encode())

    # Generate YAML files
    yaml_count = lang_counts.get("yaml", 0)

    for i in range(yaml_count):
        config_dir = root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        file_path = config_dir / f"application-{total_files:05d}.yml"
        content = _generate_yaml_content(rng, total_files)
        file_path.write_text(content)
        file_paths.append(file_path)
        total_files += 1
        indexable_files += 1
        bytes_total += len(content.encode())

    # Generate JSON files
    json_count = lang_counts.get("json", 0)

    for i in range(json_count):
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / f"records-{total_files:05d}.json"
        content = _generate_json_content(rng, total_files)
        file_path.write_text(content)
        file_paths.append(file_path)
        total_files += 1
        indexable_files += 1
        bytes_total += len(content.encode())

    # Generate other files
    other_count = lang_counts.get("other", 0)

    for i in range(other_count):
        misc_dir = root / "misc"
        misc_dir.mkdir(parents=True, exist_ok=True)
        file_path = misc_dir / f"file-{total_files:05d}.txt"
        content = f"Generated content for file {total_files}\n" * 10
        file_path.write_text(content)
        file_paths.append(file_path)
        total_files += 1
        indexable_files += 1
        bytes_total += len(content.encode())

    # Create ignored directories (not counted as indexable)
    if profile.include_ignored_dirs:
        ignored = [".git", "node_modules", "target", ".venv", "build", "dist"]
        for ign in ignored:
            ign_dir = root / ign / "subdir"
            ign_dir.mkdir(parents=True, exist_ok=True)
            (ign_dir / f".keep-{total_files}.txt").write_text("ignored")
            total_files += 1
            bytes_total += 50

    # Create sensitive-looking files (should be skipped)
    sensitive_dir = root / "secrets"
    sensitive_dir.mkdir(parents=True, exist_ok=True)
    (sensitive_dir / ".env").write_text("SECRET_KEY=dummy\n")
    (sensitive_dir / "id_rsa").write_text("-----BEGIN RSA PRIVATE KEY-----\nMII...\n")
    total_files += 2
    bytes_total += 100

    # Add some filler to reach target bytes
    target_bytes = profile.file_count * profile.avg_bytes_per_file
    if bytes_total < target_bytes:
        filler_dir = root / "filler"
        filler_dir.mkdir(parents=True, exist_ok=True)
        filler_needed = target_bytes - bytes_total
        filler_file = filler_dir / f"filler-{profile.name}.txt"
        filler_file.write_text(("\n" + "x" * 78 + "\n") * (filler_needed // 80 + 1))
        bytes_total = target_bytes

    return CorpusManifest(
        profile=profile.name,
        root=root,
        files_total=total_files,
        indexable_files=indexable_files,
        bytes_total=bytes_total,
        query_terms=QUERY_TERMS,
    )


def apply_mutation(root: Path, manifest: CorpusManifest, mutation: MutationPlan) -> CorpusManifest:
    """
    Apply a mutation plan to an existing corpus.

    Modifies files deterministically and returns an updated manifest.
    """
    root = Path(root)
    rng = _deterministic_random(manifest.indexable_files + 1)

    files = list(root.rglob("*.java")) + list(root.rglob("*.py")) + list(root.rglob("*.md"))
    indexable_files = [f for f in files if not any(part in f.parts for part in [".git", "node_modules", "target", ".venv", "secrets", "filler"])]

    if mutation.change_files > 0 and indexable_files:
        for _ in range(min(mutation.change_files, len(indexable_files))):
            f = rng.choice(indexable_files)
            content = f.read_text()
            # Add a deterministic change marker
            marker = f"\n# Modified by benchmark: {rng.randint(1000, 9999)}\n"
            lines = content.split("\n")
            mid = len(lines) // 2
            lines.insert(mid, marker)
            f.write_text("\n".join(lines))

            # Force mtime update if requested
            if mutation.force_mtime_tick:
                import time

                time.sleep(0.01)  # Small delay to ensure mtime changes
                os.utime(f, None)

    if mutation.delete_files > 0 and indexable_files:
        for _ in range(min(mutation.delete_files, len(indexable_files))):
            f = rng.choice(indexable_files)
            if f.exists():
                f.unlink()

    if mutation.add_files > 0:
        add_dir = root / "added"
        add_dir.mkdir(parents=True, exist_ok=True)
        for i in range(mutation.add_files):
            f = add_dir / f"added-{i:05d}.java"
            content = _generate_java_content(rng, manifest.indexable_files + i, "added")
            f.write_text(content)

    # Recalculate manifest
    new_files = list(root.rglob("*.java")) + list(root.rglob("*.py")) + list(root.rglob("*.md"))
    new_indexable = [f for f in new_files if not any(part in f.parts for part in [".git", "node_modules", "target", ".venv", "secrets", "filler"])]
    bytes_total = sum(f.stat().st_size for f in new_indexable)

    return CorpusManifest(
        profile=manifest.profile,
        root=manifest.root,
        files_total=len(list(root.rglob("*"))) - len(list(root.rglob(".git/*"))),
        indexable_files=len(new_indexable),
        bytes_total=bytes_total,
        query_terms=manifest.query_terms,
    )
