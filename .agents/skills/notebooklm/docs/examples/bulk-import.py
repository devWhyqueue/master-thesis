#!/usr/bin/env python3
"""Bulk import sources example.

This script demonstrates:
1. Create a notebook
2. Add multiple sources of different types
3. Handle errors gracefully
4. Report import status

Prerequisites:
    pip install "notebooklm-py[browser]"
    notebooklm login

Usage:
    python bulk-import.py
"""

import asyncio
import logging

from notebooklm import NotebookLMClient

logger = logging.getLogger(__name__)

# Example sources to import
SOURCES = {
    "urls": [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Deep_learning",
    ],
    "youtube": [
        "https://www.youtube.com/watch?v=aircAruvnKk",  # 3Blue1Brown neural networks
    ],
    "text": [
        {
            "title": "Project Notes",
            "content": """
            Key points for our ML research project:
            - Focus on transformer architectures
            - Compare with traditional RNN approaches
            - Benchmark on standard datasets
            """,
        },
    ],
}


async def main():
    logger.info("=== Bulk Import Example ===\n")

    async with await NotebookLMClient.from_storage() as client:
        # 1. Create a notebook
        logger.info("Creating notebook...")
        nb = await client.notebooks.create("Bulk Import Demo")
        logger.info(f"  Created: {nb.id}\n")

        results = {"success": [], "failed": []}

        # 2. Import URLs
        logger.info("Importing URLs...")
        for url in SOURCES["urls"]:
            try:
                source = await client.sources.add_url(nb.id, url)
                results["success"].append(f"URL: {source.title}")
                logger.info(f"  + {source.title}")
            except Exception as e:
                results["failed"].append(f"URL: {url} - {e}")
                logger.info(f"  - Failed: {url}")

        # 3. Import YouTube videos (add_url auto-detects YouTube)
        logger.info("\nImporting YouTube videos...")
        for url in SOURCES["youtube"]:
            try:
                source = await client.sources.add_url(nb.id, url)
                results["success"].append(f"YouTube: {source.title}")
                logger.info(f"  + {source.title}")
            except Exception as e:
                results["failed"].append(f"YouTube: {url} - {e}")
                logger.info(f"  - Failed: {url}")

        # 4. Import text content
        logger.info("\nImporting text content...")
        for item in SOURCES["text"]:
            try:
                source = await client.sources.add_text(nb.id, item["title"], item["content"])
                results["success"].append(f"Text: {source.title}")
                logger.info(f"  + {source.title}")
            except Exception as e:
                results["failed"].append(f"Text: {item['title']} - {e}")
                logger.info(f"  - Failed: {item['title']}")

        # 5. Report results
        logger.info("\n" + "=" * 40)
        logger.info("Import complete!")
        logger.info(f"  Successful: {len(results['success'])}")
        logger.info(f"  Failed: {len(results['failed'])}")

        if results["failed"]:
            logger.info("\nFailed imports:")
            for item in results["failed"]:
                logger.info(f"  - {item}")

        logger.info(f"\n  Notebook ID: {nb.id}")
        logger.info("  (Notebook kept for review - delete manually when done)")

    logger.info("\n=== Done! ===")


if __name__ == "__main__":
    asyncio.run(main())
