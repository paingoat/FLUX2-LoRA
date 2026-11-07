"""Install a guarded, opt-in region weighting patch in the pinned AI Toolkit."""
import argparse
import hashlib
from pathlib import Path

REVISION = 'beb94824c51480a04d3c86c718603676fdef6938'
ORIGINAL_SHA256 = 'be1f637042d2fb5693138770b732feda27c11f61d14e7acd6cc961c211fcf6a1'
MARKER = '# FLUX2-LoRA region weighting v1'
OLD = '''                    # make avg 1.0
                    mask_multiplier = mask_multiplier / mask_multiplier.mean()'''
NEW = '''                    # FLUX2-LoRA region weighting v1
                    if "FLUX_REGION_STRENGTH" in os.environ:
                        from region_loss import region_weights
                        mask_multiplier = region_weights(
                            mask_multiplier[:, :1], float(os.environ["FLUX_REGION_STRENGTH"])
                        )
                    else:
                        mask_multiplier = mask_multiplier / mask_multiplier.mean()'''
GUARD_OLD = '''            mask_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)'''
GUARD_NEW = GUARD_OLD + '''
            if "FLUX_REGION_STRENGTH" in os.environ:
                if batch.mask_tensor is None or not self.sd.do_masked_loss:
                    raise RuntimeError("Region loss requires a loaded mask and do_masked_loss")'''
FAIL_OLD = '''            print("Could not apply mask multiplier to loss")'''
FAIL_NEW = '''            if "FLUX_REGION_STRENGTH" in os.environ:
                raise
            print("Could not apply mask multiplier to loss")'''


def patched_source(source: str) -> str:
    if MARKER in source:
        if NEW not in source or GUARD_NEW not in source or FAIL_NEW not in source:
            raise ValueError('incomplete or modified region loss patch')
        original = source.replace(NEW, OLD).replace(GUARD_NEW, GUARD_OLD).replace(FAIL_NEW, FAIL_OLD)
        if hashlib.sha256(original.encode()).hexdigest() != ORIGINAL_SHA256:
            raise ValueError('patched trainer differs from the pinned source')
        return source
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError('trainer hash differs from the pinned source')
    if source.count(OLD) != 1 or source.count(GUARD_OLD) != 1 or source.count(FAIL_OLD) != 1:
        raise ValueError('unsupported SDTrainer source; use the pinned toolkit revision')
    result = source.replace(OLD, NEW).replace(GUARD_OLD, GUARD_NEW).replace(FAIL_OLD, FAIL_NEW)
    compile(result, 'SDTrainer.py', 'exec')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('toolkit', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = args.toolkit / 'extensions_built_in/sd_trainer/SDTrainer.py'
    source = path.read_text(encoding='utf-8')
    result = patched_source(source)
    if args.check:
        if source != result:
            raise SystemExit('region patch missing; run core/patch_region_trainer.py first')
    elif source != result:
        path.write_text(result, encoding='utf-8', newline='\n')
    print(f'region trainer ready: sha256={hashlib.sha256(result.encode()).hexdigest()}')


if __name__ == '__main__':
    main()
