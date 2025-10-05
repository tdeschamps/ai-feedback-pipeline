# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of this project seriously. If you have discovered a security vulnerability, we appreciate your help in disclosing it to us responsibly.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisories** (Preferred): https://github.com/yourusername/ai-feedback-pipeline/security/advisories/new
2. **Email**: security@yourproject.org

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours.
- **Updates**: We will send you regular updates about our progress, at least every 5 business days.
- **Timeline**: We aim to address critical vulnerabilities within 7 days and high-priority issues within 30 days.
- **Disclosure**: We will work with you to understand the issue and develop a fix before any public disclosure.

## Security Best Practices

When using this project, please follow these security best practices:

### API Key Management

- **Never commit API keys, secrets, or credentials to the repository.**
- Use environment variables and a `.env` file (excluded from version control) for all secrets.
- Example `.env` (do not commit real values):

	```env
	OPENAI_API_KEY=your_openai_key_here
	NOTION_API_KEY=your_notion_key_here
	PINECONE_API_KEY=your_pinecone_key_here
	```
- Rotate API keys regularly and immediately if you suspect exposure.
- Use the provided `.gitleaks.toml` and pre-commit hooks to scan for secrets before pushing code.
- Review CI/CD logs for accidental secret exposure.
- Use separate keys for development, staging, and production.
- Restrict API key permissions to minimum required scope.

### Environment Variables

- Always use `.env` file for local development (never commit this file)
- Use the provided `.env.example` as a template
- In production, use secure environment variable injection
- Validate all environment variables on startup

### Data Security

- Review all data before storing in vector databases
- Be cautious with PII (Personally Identifiable Information)
- Implement proper access controls for Notion databases
- Sanitize user inputs before processing
- Enable encryption at rest and in transit where applicable

### Dependency Security

- Regularly update dependencies using `uv sync`
- Review security advisories from Dependabot
- Run `safety check` to scan for known vulnerabilities
- Use `bandit` for Python security linting

### Network Security

- Use HTTPS in production
- Implement rate limiting on API endpoints
- Configure CORS properly for your use case
- Use API authentication/authorization when deploying

## Security Features

This project includes several security features:

- **Automated Security Scanning**: Bandit and Safety checks in CI/CD
- **Secret Detection**: Gitleaks integration to prevent secret commits
- **Dependency Monitoring**: Dependabot for automatic security updates
- **Type Safety**: MyPy type checking to prevent common bugs
- **Input Validation**: Pydantic models for request validation

## Known Security Considerations

### LLM-Specific Risks

- **Prompt Injection**: User-provided transcript content is sent to LLMs. Validate and sanitize inputs.
- **Data Leakage**: Be aware that data sent to cloud LLM providers may be stored temporarily.
- **Cost Control**: Implement rate limiting to prevent abuse and unexpected costs.

### Vector Database Security

- **Access Control**: ChromaDB and Pinecone collections should have proper access controls.
- **Data Sensitivity**: Embeddings may contain sensitive information from source data.

### Notion Integration

- **API Token Security**: Notion API tokens have broad permissions. Store securely.
- **Data Validation**: Validate data before writing to Notion to prevent injection.

## Responsible Disclosure

We follow a coordinated disclosure process:

1. Security researchers report vulnerabilities privately
2. We confirm and develop a fix
3. We release a patch and security advisory
4. Public disclosure after patch is available

We appreciate the security research community and will acknowledge contributors in our security advisories (unless you prefer to remain anonymous).

## Updates and Patches

Security updates will be released as:

- **Critical**: Immediate patch release
- **High**: Patch within 7 days
- **Medium**: Patch in next minor version
- **Low**: Patch in next regular release

Follow our [GitHub releases](https://github.com/yourusername/ai-feedback-pipeline/releases) for security update notifications.

## Contact

For security concerns, contact:
- Security team: security@yourproject.org
- Project maintainer: See CONTRIBUTING.md

---

Last updated: 2025-10-05
