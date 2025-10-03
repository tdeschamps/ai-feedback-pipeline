# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of AI Feedback Pipeline seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Where to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via:
- GitHub Security Advisories: [Report a vulnerability](https://github.com/tdeschamps/ai-feedback-pipeline/security/advisories/new)
- Or open a private issue and tag it with the security label

### What to Include

Please include the following information in your report:
- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- We will acknowledge your report within 72 hours
- We will provide a detailed response within 7 days indicating the next steps
- We will keep you informed about the progress towards a fix and announcement
- We may ask for additional information or guidance

## Security Best Practices

### API Key Management

1. **Never commit API keys to version control**
   - Use `.env` files (already in `.gitignore`)
   - Use environment variables in production
   - Rotate keys regularly

2. **Use environment-specific keys**
   - Development keys for local work
   - Separate keys for staging and production

3. **Limit API key permissions**
   - Use read-only keys where possible
   - Apply least privilege principle

### Data Security

1. **Customer Data**
   - Do not commit real customer data to the repository
   - Use synthetic/anonymized data for testing
   - Implement data retention policies

2. **PII Handling**
   - Avoid logging PII (names, emails, phone numbers)
   - Sanitize data before storage
   - Comply with GDPR/CCPA requirements

### Dependency Security

1. **Keep dependencies updated**
   - Dependabot automatically creates PRs for security updates
   - Review and merge security patches promptly

2. **Review new dependencies**
   - Check dependency health and maintenance status
   - Scan for known vulnerabilities before adding

### Deployment Security

1. **Environment Variables**
   - Never expose API keys in logs or error messages
   - Use secrets management in production (AWS Secrets Manager, Azure Key Vault, etc.)

2. **Network Security**
   - Use HTTPS for all external communications
   - Implement rate limiting on API endpoints
   - Use authentication for production endpoints

## Security Testing

We use the following tools to ensure code security:

- **Bandit**: Python security linting
- **Safety**: Dependency vulnerability scanning  
- **Ruff**: Code quality and security patterns
- **GitHub Secret Scanning**: Detects accidentally committed secrets
- **Dependabot**: Automated dependency updates

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find any similar problems
3. Prepare fixes for all supported versions
4. Release patches as soon as possible

## Comments on This Policy

If you have suggestions on how this process could be improved, please submit a pull request.
