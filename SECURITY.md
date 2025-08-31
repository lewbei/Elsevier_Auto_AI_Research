# Security Policy

## Supported Versions

We actively support the latest version of the Elsevier Auto AI Research pipeline. Security updates will be provided for:

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

### 🔒 Private Reporting
- **Email**: Create a GitHub issue with the label `security` and mark it as private
- **Response Time**: We aim to respond within 48 hours
- **Updates**: We'll provide updates on the investigation every 72 hours

### 🚨 What to Include
When reporting a security issue, please include:

1. **Description**: Detailed description of the vulnerability
2. **Impact**: Potential impact and affected components
3. **Reproduction**: Steps to reproduce the issue
4. **Environment**: OS, Python version, and configuration details
5. **Proof of Concept**: If applicable (please be responsible)

### 🔐 Security Considerations

This project handles sensitive information that requires careful attention:

#### API Keys and Credentials
- **Elsevier API Keys**: Required for paper access - store in `.env` only
- **DeepSeek API Keys**: Required for LLM functionality - store in `.env` only
- **Institutional Tokens**: Optional but sensitive - store in `.env` only

#### Data Handling
- **Downloaded Papers**: May contain copyrighted content
- **Generated Content**: May include sensitive research information
- **Cache Files**: May contain API responses with sensitive data
- **Log Files**: May inadvertently log sensitive information

#### Potential Security Risks
1. **API Key Exposure**: Keys accidentally committed to version control
2. **Path Traversal**: Unsafe file operations with user-controlled paths
3. **Code Injection**: Unsafe evaluation of generated code or configurations
4. **Data Leakage**: Sensitive information in logs or generated outputs
5. **MITM Attacks**: Insecure API communications

### 🛡️ Security Best Practices

#### For Users
- Keep API keys in `.env` files only (never commit to git)
- Regularly rotate API keys
- Use institutional tokens when available
- Review generated code before execution
- Monitor API usage for unusual activity
- Keep dependencies updated

#### For Contributors
- Never log API keys or sensitive data
- Validate all external inputs
- Use parameterized queries for any database operations
- Sanitize file paths and names
- Review generated code for safety
- Test with minimal privileges

### 🚫 Out of Scope

The following are generally not considered security vulnerabilities:
- Rate limiting or quota exhaustion
- API service availability issues
- Performance issues or resource consumption
- Issues requiring physical access to the machine
- Social engineering attacks

### 🔄 Response Process

1. **Initial Response**: Acknowledge receipt within 48 hours
2. **Investigation**: Assess the vulnerability and its impact
3. **Development**: Create and test a fix
4. **Disclosure**: Coordinate disclosure timeline
5. **Release**: Deploy fix and notify users
6. **Documentation**: Update security documentation as needed

### 🏆 Recognition

We appreciate security researchers who help keep our project safe:
- Responsible disclosure will be acknowledged in release notes
- Significant findings may be eligible for recognition in project documentation
- We'll work with you on appropriate attribution

---

Thank you for helping keep Elsevier Auto AI Research secure! 🔒