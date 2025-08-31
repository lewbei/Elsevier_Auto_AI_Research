import os
from lab.experiment_runner import dataset_choice


def test_dataset_choice_default(monkeypatch):
    monkeypatch.delenv("DATASET", raising=False)
    assert dataset_choice() == "isic"


def test_dataset_choice_cifar(monkeypatch):
    monkeypatch.setenv("DATASET", "cifar10")
    assert dataset_choice() == "cifar10"

