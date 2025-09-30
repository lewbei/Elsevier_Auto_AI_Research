"""Pydantic models defining structured code generation artifacts.

This enables LLM forced JSON mode (via output parser) to return a normalized
plan before any filesystem writes occur.

Schema summary:
- FileArtifact: a single file to create or patch.
- CodegenPlan: list of FileArtifact + high-level intent + constraints.
- CodegenResult: result after attempting to apply the plan (including test
  outcomes).

The LLM is instructed to output JSON matching CodegenPlan exactly; any other
output is rejected before proceeding.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
from pathlib import Path
import hashlib
import difflib


def stable_checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class FileArtifact(BaseModel):
    path: str = Field(..., description="Relative path within repo (posix style)")
    language: Literal["python", "markdown", "text", "yaml", "json"] = Field(
        "python", description="File language to assist formatting policies"
    )
    intent: str = Field(..., description="Short human readable purpose of the file")
    action: Literal["create", "update", "patch"] = Field(
        ..., description="Creation mode; patch means unified diff content"
    )
    content: str = Field(..., description="Full file body for create/update; for patch it's unified diff")
    tests: List[str] = Field(
        default_factory=list,
        description="Optional list of test function names expected inside generated tests for this artifact",
    )
    checksum: Optional[str] = Field(
        None, description="Checksum of content (autofilled if absent)"
    )

    @validator("path")
    def _no_traversal(cls, v: str) -> str:  # noqa: D401
        if ".." in Path(v).parts:
            raise ValueError("Path traversal not allowed")
        return v.replace("\\", "/")

    @validator("checksum", always=True)
    def _auto_checksum(cls, v: Optional[str], values):  # noqa: D401
        if not v and (content := values.get("content")):
            return stable_checksum(content)
        return v

    def as_path(self, root: Path) -> Path:
        return (root / self.path).resolve()

    def make_patch(self, existing_text: str) -> str:
        """Return a unified diff from existing_text -> self.content.
        Only used when action == 'patch' for transparency; does not apply it.
        """
        new_lines = self.content.splitlines(keepends=True)
        old_lines = existing_text.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines, fromfile=f"a/{self.path}", tofile=f"b/{self.path}", lineterm=""
        )
        return "\n".join(diff)


class CodegenPlan(BaseModel):
    objective: str = Field(..., description="High-level goal of this codegen session")
    summary: str = Field(..., description="Short narrative summary of planned changes")
    files: List[FileArtifact] = Field(..., description="List of file artifacts to create/update/patch")
    notes: Optional[str] = Field(None, description="Additional considerations or constraints")

    @validator("files")
    def _non_empty(cls, v: List[FileArtifact]):  # noqa: D401
        if not v:
            raise ValueError("At least one file artifact required")
        return v

    def file_index(self) -> dict[str, FileArtifact]:
        return {fa.path: fa for fa in self.files}

    def checksum_manifest(self) -> dict[str, str]:
        return {fa.path: fa.checksum or "" for fa in self.files}


class TestResult(BaseModel):
    passed: bool
    total: int
    failed: int
    errors: int
    duration_sec: float
    stdout: Optional[str] = None


class AppliedFile(BaseModel):
    path: str
    action: str
    status: Literal["skipped", "applied", "error"]
    reason: Optional[str] = None
    checksum: Optional[str] = None


class CodegenResult(BaseModel):
    plan: CodegenPlan
    applied: List[AppliedFile]
    test_result: Optional[TestResult] = None
    success: bool = False
    message: Optional[str] = None
    run_dir: Optional[str] = None

    def ok(self) -> bool:
        if not self.success:
            return False
        if self.test_result and not self.test_result.passed:
            return False
        return True
