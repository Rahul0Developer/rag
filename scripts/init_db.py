"""
Initialise the DocuMind AI database schema (and optionally seed sample docs).

Usage:
    python scripts/init_db.py            # create tables
    python scripts/init_db.py --seed     # create tables + ingest sample documents
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/init_db.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

from app.database import engine, init_db, async_session_factory  # noqa: E402


SAMPLE_DOCS = {
    "acme_service_agreement.md": """# Master Services Agreement — Acme Corp & Globex Ltd

This Master Services Agreement ("Agreement") is entered into as of January 15, 2025,
between Acme Corporation ("Provider") and Globex Limited ("Client").

## Term
The initial term of this Agreement is twenty-four (24) months from the Effective Date.
Either party may terminate with sixty (60) days written notice.

## Fees and Payment
Client shall pay Provider a monthly retainer of $45,000 USD. Invoices are due within
thirty (30) days. Late payments accrue interest at 1.5% per month.

## Confidentiality
Each party agrees to protect the other's confidential information for a period of
five (5) years following termination.

## Governing Law
This Agreement is governed by the laws of the State of Delaware, USA.
""",
    "invoice_2025_0042.txt": """INVOICE
Invoice Number: INV-2025-0042
Date: March 3, 2025
Bill To: Globex Limited, 500 Market Street, San Francisco, CA

Line Items:
1. Enterprise AI Platform - Q1 Subscription .......... $135,000.00
2. Premium Support Package (24/7) .................... $12,500.00
3. Onboarding & Integration Services ................. $8,750.00

Subtotal: $156,250.00
Tax (8.25%): $12,890.63
Total Amount Due: $169,140.63

Payment Terms: Net 30. Wire transfer to Acme Corp, Account ****4821.
Late Fee: 1.5% per month on overdue balances.
""",
    "quarterly_report_q1.md": """# Q1 2025 Performance Report — Acme Corporation

## Executive Summary
Acme reported Q1 revenue of $42.3 million, a 28% increase year-over-year.
Net income reached $6.1 million. The AI division grew 64% and now represents
35% of total revenue.

## Key Metrics
- Monthly Active Users: 1.2 million (+19% QoQ)
- Customer Retention: 94%
- Average Contract Value: $86,000 (+11%)
- Churn: reduced from 3.1% to 2.4%

## Outlook
Management projects full-year revenue between $180M and $195M, driven by
enterprise adoption of the DocuMind product line in APAC and EMEA regions.
""",
}


async def seed() -> None:
    """Ingest small built-in sample documents so the demo works instantly."""
    from app.services.document_service import ingest_file

    async with async_session_factory() as db:
        for name, text in SAMPLE_DOCS.items():
            try:
                doc = await ingest_file(db, original_name=name, content=text.encode("utf-8"))
                logger.info(f"Seeded '{name}' → id={doc.id}, chunks={doc.chunk_count}, category={doc.category}")
            except Exception as exc:
                logger.error(f"Seed failed for {name}: {exc}")
        await db.commit()


async def main(with_seed: bool) -> None:
    await init_db()
    logger.success("✅ Database schema created (documents, chunks, conversations, messages, metric_events).")
    if with_seed:
        await seed()
        logger.success("✅ Sample documents ingested. Open the dashboard and start asking questions.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main(with_seed="--seed" in sys.argv))
