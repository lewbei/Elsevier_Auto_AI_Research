from lab.codegen_utils import write_generated_aug, sanity_check_generated_aug


def test_generated_aug_guard_smoke(tmp_path):
    # Generate augmentation module in the default lab/ path
    p = write_generated_aug("jitter rotate erase blur")
    assert p.exists()
    # Guard should pass even without torchvision installed
    assert sanity_check_generated_aug() is True

