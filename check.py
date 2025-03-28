import pkg_resources

dependencies = [
    "torch>=2.5.1",
    "setuptools",
    "packaging",
    "inflect>=7.5.0",
    "kanjize>=1.5.0",
    "numpy>=2.2.2",
    "phonemizer>=3.3.0",
    "sudachidict-full>=20241021",
    "sudachipy>=0.6.10",
    "torchaudio>=2.5.1",
    "transformers>=4.48.1",
    "soundfile>=0.13.1",
    "huggingface-hub>=0.28.1",
    "gradio>=5.15.0",
    "flash-attn>=2.7.3",
    "mamba-ssm>=2.2.4",
    "causal-conv1d>=1.5.0.post8",
]

def check_dependencies(dependencies):
    for dependency in dependencies:
        try:
            pkg_resources.require(dependency)
            print(f"{dependency} is installed.")
        except pkg_resources.DistributionNotFound:
            print(f"{dependency} is NOT installed.")
        except pkg_resources.VersionConflict as e:
            print(f"{dependency} version conflict: {e}")

check_dependencies(dependencies)