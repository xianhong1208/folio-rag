# Harborlight Technologies — Security Incident Response Runbook

Source: written for the Folio retrieval eval set (fictional company procedure). License: MIT, same as this repository.

Document owner: Information Security Office (ISO)
Document number: SEC-RB-002, version 2.1
Last reviewed: 15 August 2026

## 1. Scope

This runbook describes how Harborlight responds to information security incidents affecting company systems, the Harborlight Home cloud service and the smart devices connected to it (including the Aster K2 kettle). It applies to employees, contractors and on-call engineers.

## 2. Severity Levels

| Severity | Definition | Example | Response target |
|---|---|---|---|
| SEV-1 | Confirmed breach of customer personal data, or a full outage of the Harborlight Home service | Customer database exported by an attacker | Incident commander assigned within 15 minutes |
| SEV-2 | Suspected breach, or partial outage affecting more than 10% of devices | Unusual login pattern on an admin account | Incident commander assigned within 1 hour |
| SEV-3 | Security weakness with no evidence of exploitation | Outdated TLS configuration on a test server | Ticket triaged within 1 working day |
| SEV-4 | Minor policy violation | Shared password found in a team chat | Ticket triaged within 3 working days |

## 3. Roles

- Incident Commander (IC): owns the response, decides severity and approves external communication. For SEV-1 the IC is the Chief Information Security Officer or a deputy.
- Communications Lead: drafts customer, regulator and press messages. Works with Legal on every notification.
- Scribe: keeps the timeline in the incident channel. Every action is logged with a timestamp.
- Subject Matter Experts: engineers from the affected service.

## 4. Response Phases

1. Detect and report: anyone who suspects an incident reports it in the #sec-incident channel or calls the security hotline at extension 7110. Do not investigate on your own and do not delete evidence.
2. Triage: the on-call security engineer confirms the report and sets the initial severity within 30 minutes.
3. Contain: isolate affected hosts, revoke exposed credentials and rotate secrets. Disable affected accounts rather than deleting them.
4. Eradicate and recover: remove the root cause, restore from known-good backups and monitor for 72 hours before closing.
5. Notify: when personal data is involved, Legal decides on notification of the data subjects and the competent authority. Under the Personal Data Protection Act the data subjects must be notified after the facts are verified; Harborlight's internal target is to send the first notification within 72 hours of confirming a SEV-1 breach.
6. Post-incident review: a blameless review meeting is held within 10 working days for every SEV-1 and SEV-2 incident. Action items are tracked in the risk register with an owner and a due date.

## 5. Evidence Handling

- Preserve logs for at least 180 days after an incident is closed; for SEV-1 incidents preserve them for 3 years.
- Take disk images before rebuilding any compromised host.
- Store evidence in the restricted "forensics" storage bucket. Access is limited to the ISO and Legal.

## 6. Credential Exposure Checklist

When a password, API key or token is exposed:

- Rotate the credential immediately and invalidate active sessions.
- Search the code repositories and chat history for other copies.
- Check access logs for use of the credential since the time of exposure.
- If the credential was reused on other systems, rotate those as well and record the reuse as a SEV-4 policy violation.

## 7. Contacts

- Security hotline: extension 7110 (24/7)
- ISO mailbox: security@harborlight.example
- Legal on-call: legal-oncall@harborlight.example
