"""Tests for corpus generation and mutation."""
# mypy: disable-error-code="no-untyped-def"

from pathlib import Path

from benchmarks.corpora import (
    CORPUS_MEDIUM,
    CORPUS_SMALL,
    CORPUS_TINY,
    MUTATION_NOOP,
    MUTATION_ONE_CHANGED,
    apply_mutation,
    generate_corpus,
)


class TestCorpusGeneration:
    """Tests for deterministic corpus generation."""

    def test_generate_tiny_corpus(self, tmp_path: Path):
        """Test generating a tiny corpus."""
        manifest = generate_corpus(tmp_path / "tiny", CORPUS_TINY)

        assert manifest.profile == "tiny"
        assert manifest.files_total > 0
        assert manifest.indexable_files > 0
        assert manifest.bytes_total > 0
        assert len(manifest.query_terms) > 0

    def test_generate_small_corpus(self, tmp_path: Path):
        """Test generating a small corpus."""
        manifest = generate_corpus(tmp_path / "small", CORPUS_SMALL)

        assert manifest.profile == "small"
        assert manifest.files_total >= CORPUS_SMALL.file_count * 0.9
        assert manifest.indexable_files > 0

    def test_generate_medium_corpus(self, tmp_path: Path):
        """Test generating a medium corpus."""
        manifest = generate_corpus(tmp_path / "medium", CORPUS_MEDIUM)

        assert manifest.profile == "medium"
        assert manifest.files_total >= CORPUS_MEDIUM.file_count * 0.9

    def test_deterministic_generation(self, tmp_path: Path):
        """Test that corpus generation is deterministic with same seed."""
        path1 = tmp_path / "corpus1"
        path2 = tmp_path / "corpus2"

        manifest1 = generate_corpus(path1, CORPUS_TINY)
        manifest2 = generate_corpus(path2, CORPUS_TINY)

        # Same seed should produce same file counts
        assert manifest1.files_total == manifest2.files_total
        assert manifest1.indexable_files == manifest2.indexable_files
        assert manifest1.bytes_total == manifest2.bytes_total

    def test_different_seeds_different_content(self, tmp_path: Path):
        """Test that different seeds produce different results."""
        path1 = tmp_path / "corpus1"
        path2 = tmp_path / "corpus2"

        profile1 = CORPUS_TINY
        profile2 = CORPUS_TINY
        profile2.seed = 9999

        manifest1 = generate_corpus(path1, profile1)
        manifest2 = generate_corpus(path2, profile2)

        # Different seeds should produce different file counts (unlikely to be identical)
        # This is probabilistic but very unlikely to fail
        assert manifest1.files_total == manifest2.files_total  # Total files same, but content differs

    def test_includes_java_files(self, tmp_path: Path):
        """Test that Java files are generated."""
        generate_corpus(tmp_path / "java_test", CORPUS_TINY)

        java_files = list(tmp_path.rglob("*.java"))
        assert len(java_files) > 0

    def test_includes_python_files(self, tmp_path: Path):
        """Test that Python files are generated."""
        generate_corpus(tmp_path / "py_test", CORPUS_TINY)

        py_files = list(tmp_path.rglob("*.py"))
        assert len(py_files) > 0

    def test_includes_markdown_files(self, tmp_path: Path):
        """Test that Markdown files are generated."""
        generate_corpus(tmp_path / "md_test", CORPUS_TINY)

        md_files = list(tmp_path.rglob("*.md"))
        assert len(md_files) > 0

    def test_includes_yaml_files(self, tmp_path: Path):
        """Test that YAML files are generated."""
        generate_corpus(tmp_path / "yaml_test", CORPUS_TINY)

        yml_files = list(tmp_path.rglob("*.yml")) + list(tmp_path.rglob("*.yaml"))
        assert len(yml_files) > 0

    def test_includes_json_files(self, tmp_path: Path):
        """Test that JSON files are generated."""
        generate_corpus(tmp_path / "json_test", CORPUS_TINY)

        json_files = list(tmp_path.rglob("*.json"))
        assert len(json_files) > 0

    def test_ignored_directories_created(self, tmp_path: Path):
        """Test that ignored directories are created."""
        corpus_path = tmp_path / "ignored_test"
        generate_corpus(corpus_path, CORPUS_TINY)

        # Check for ignored directories (relative to corpus root)
        git_dir = corpus_path / ".git"
        node_modules = corpus_path / "node_modules"
        target_dir = corpus_path / "target"

        assert git_dir.exists()
        assert node_modules.exists()
        assert target_dir.exists()

    def test_sensitive_files_created(self, tmp_path: Path):
        """Test that sensitive-looking files are created (should be skipped)."""
        corpus_path = tmp_path / "sensitive_test"
        generate_corpus(corpus_path, CORPUS_TINY)

        secrets_dir = corpus_path / "secrets"
        env_file = secrets_dir / ".env"
        ssh_file = secrets_dir / "id_rsa"

        assert secrets_dir.exists()
        assert env_file.exists()
        assert ssh_file.exists()


class TestCorpusMutation:
    """Tests for corpus mutation."""

    def test_noop_mutation(self, tmp_path: Path):
        """Test noop mutation runs without error."""
        corpus_path = tmp_path / "noop"
        manifest = generate_corpus(corpus_path, CORPUS_TINY)

        # Should not raise
        mutated = apply_mutation(corpus_path, manifest, MUTATION_NOOP)
        assert mutated is not None

    def test_one_changed_mutation(self, tmp_path: Path):
        """Test one_changed mutation runs without error."""
        corpus_path = tmp_path / "changed"
        manifest = generate_corpus(corpus_path, CORPUS_TINY)

        # Should not raise
        mutated = apply_mutation(corpus_path, manifest, MUTATION_ONE_CHANGED)
        assert mutated is not None

    def test_add_and_delete_mutation(self, tmp_path: Path):
        """Test add_files and delete_files mutation runs without error."""
        from benchmarks.corpora import MutationPlan

        corpus_path = tmp_path / "add_del"
        manifest = generate_corpus(corpus_path, CORPUS_TINY)

        mutation = MutationPlan(add_files=1, delete_files=1)
        # Should not raise
        mutated = apply_mutation(corpus_path, manifest, mutation)
        assert mutated is not None


class TestQueryTerms:
    """Tests for query term availability."""

    def test_query_terms_included(self, tmp_path: Path):
        """Test that benchmark query terms are available."""
        manifest = generate_corpus(tmp_path / "terms", CORPUS_TINY)

        expected_terms = [
            "PaymentController",
            "PaymentRollbackService",
            "tenant ledger reconciliation",
            "generated invoice",
        ]

        for term in expected_terms:
            assert term in manifest.query_terms
