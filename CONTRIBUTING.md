# Contributing to AI Feedback Pipeline

Thank you for considering contributing to this project! We welcome all contributions that help make this project more secure, robust, and useful for the community.

## How to Contribute

- Fork the repository and create your branch from `main` or `develop`.
- Write clear, concise commit messages.
- Ensure your code passes all tests and linting checks.
- Submit a pull request with a detailed description of your changes.

## Security Guidelines

- **Never commit secrets or API keys.** Use environment variables and `.env` files (excluded from version control).
- Use the provided `.gitleaks.toml` and pre-commit hooks to scan for secrets before pushing code.
- Run `bandit` and `safety` to check for Python security issues.
- Report any security vulnerabilities to [security@cyclecheap.org](mailto:security@cyclecheap.org) immediately.
- Follow the [OWASP Top 10](https://owasp.org/www-project-top-ten/) best practices for secure coding.

## Reporting Security Issues

If you discover a security vulnerability, please email [security@cyclecheap.org](mailto:security@cyclecheap.org) with details. Do **not** open a public issue for security problems.

## Code Style & Quality

- Use [Black](https://black.readthedocs.io/) and [Ruff](https://docs.astral.sh/ruff/) for formatting and linting.
- Write tests for new features and bug fixes.
- Document public functions and classes with docstrings.

## Pull Request Process

1. Ensure your branch is up to date with the base branch.
2. Run all tests and security checks locally.
3. Submit your pull request and describe your changes.
4. A maintainer will review your PR and may request changes.
5. Once approved, your PR will be merged.

## Community Standards

- Be respectful and inclusive. See our [Code of Conduct](CODE_OF_CONDUCT.md).
- Help us keep the project safe and welcoming for everyone.

Thank you for helping us build a secure and open feedback pipeline!
