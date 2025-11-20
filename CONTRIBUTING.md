# Contributing to World Models (JAX)

Thank you for your interest in contributing to this World Models implementation! This project aims to provide an accessible, educational reproduction of the seminal [World Models paper](https://arxiv.org/abs/1803.10122) by Ha & Schmidhuber.

## Types of Contributions

We welcome contributions in several forms:

- **Bug fixes**: Report and fix issues in the implementation
- **Performance improvements**: Optimize training speed or model accuracy
- **Documentation**: Improve clarity of code, README, or add tutorials
- **Research extensions**: Novel improvements to the World Models architecture
- **Reproducibility**: Ensure results match reported scores

## Academic Citation

If you use this code in your research, please cite both the original World Models paper and this implementation:

```bibtex
@article{ha2018world,
  title={World models},
  author={Ha, David and Schmidhuber, J{\"u}rgen},
  journal={arXiv preprint arXiv:1803.10122},
  year={2018}
}

@software{tiraspolsky2025worldmodelsjax,
  author = {Tiraspolsky, Sergey},
  title = {World Models (JAX): A JAX/Equinox Implementation},
  url = {https://github.com/Sha01in/world-models-jax},
  version = {0.1.0},
  year = {2025}
}
```

## Development Setup

1. Fork and clone the repository
2. Install dependencies: `uv sync` or `pip install -r requirements.txt`
3. Run tests: `python run_test.py`
4. Make your changes

## Code Style

- Follow PEP 8 style guidelines
- Use descriptive variable names
- Add docstrings to functions and classes
- Include type hints where possible

## Testing

Before submitting a pull request:

1. Ensure all existing tests pass
2. Add tests for new functionality
3. Verify the agent can achieve the reported scores
4. Test on multiple environments if applicable

## Reporting Issues

When reporting bugs, please include:

- JAX/Equinox versions
- Python version
- GPU/CPU specifications
- Steps to reproduce the issue
- Expected vs. actual behavior

## Research Contributions

For significant research contributions (e.g., novel loss functions, architectural changes):

- Clearly document the motivation and methodology
- Provide ablation studies or comparative results
- Ensure backward compatibility where possible
- Update relevant documentation

## License

By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project.
