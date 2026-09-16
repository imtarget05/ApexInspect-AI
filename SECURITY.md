# SECURITY — Industrial Edge AI & Operational Safety Policy

**ApexInspect AI** is designed for mission-critical factory floor deployments where software faults or unauthorized actions can impact physical production machinery (SMT lines).

---

## 🛡️ 1. Operational Safety & Excessive Agency Mitigation

- **Deterministic Human-In-The-Loop (HITL)**: LangGraph `interrupt()` primitives prevent automated agents from directly halting physical production lines. Halting mutations strictly require explicit supervisor approval via the MES console.
- **Physical Line Fail-Safe**: In the event of system crash or loss of camera signal, line state defaults to safe alert status rather than silent failure.

---

## 🔒 2. Data Privacy & Edge Isolation

- **Zero Cloud Image Leaks**: Image inspection and ONNX Runtime inference execute locally on the edge device / private cloud container. No raw photographic PCB frames are sent to external third-party LLM APIs.
- **SOP Knowledge Retrieval Isolation**: SOP manuals and equipment specifications are indexed locally via vector embeddings without public cloud exposure.

---

## 🔑 3. Secrets & Credentials Governance

- **Zero-Secret Commitment**: API keys (`GROQ_API_KEY`, database connection strings) are strictly managed via environment variables and **Azure Key Vault**.
- Real `.env` files are gitignored (`.env`, `.env.*`).
- Container images contain no baked-in credentials.
