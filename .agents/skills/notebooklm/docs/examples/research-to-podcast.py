#!/usr/bin/env python3
"""Research to podcast workflow example.

This script demonstrates:
1. Create a notebook
2. Start deep research on a topic
3. Import discovered sources
4. Generate a podcast from the research

Prerequisites:
    pip install "notebooklm-py[browser]"
    notebooklm login

Usage:
    python research-to-podcast.py "Your research topic"
"""

import asyncio
import logging
import sys

from notebooklm import NotebookLMClient

logger = logging.getLogger(__name__)


async def main(topic: str):
    logger.info(f"=== Research to Podcast: {topic} ===\n")

    async with await NotebookLMClient.from_storage() as client:
        # 1. Create a notebook
        logger.info("Creating notebook...")
        nb = await client.notebooks.create(f"Research: {topic}")
        logger.info(f"  Created: {nb.id}\n")

        # 2. Start deep research
        logger.info("Starting deep research (this may take a while)...")
        research = await client.research.start(nb.id, topic, source="web", mode="deep")
        task_id = research.get("task_id") if research else None
        logger.info(f"  Task ID: {task_id}\n")

        # 3. Poll for completion
        logger.info("Waiting for research to complete...")
        max_polls = 30
        for i in range(max_polls):
            status = await client.research.poll(nb.id)
            state = status.get("status", "unknown")
            logger.info(f"  Poll {i + 1}/{max_polls}: {state}")

            if state == "completed":
                sources = status.get("sources", [])
                logger.info(f"  Found {len(sources)} sources!\n")
                break

            await asyncio.sleep(10)
        else:
            logger.info("  Research timed out\n")
            return

        # 4. Import discovered sources
        if sources:
            logger.info("Importing sources...")
            await client.research.import_sources(nb.id, task_id, sources[:10])  # Limit to 10
            logger.info(f"  Imported {min(len(sources), 10)} sources\n")

        # 5. Generate podcast
        logger.info("Generating podcast...")
        gen_status = await client.artifacts.generate_audio(
            nb.id, instructions=f"Create an engaging overview of {topic}"
        )

        logger.info("Waiting for audio generation...")
        final = await client.artifacts.wait_for_completion(nb.id, gen_status.task_id, timeout=600)

        if final.is_complete:
            logger.info(f"\n  Success! Audio URL: {final.url}")
            logger.info("\n  Use 'notebooklm download audio' to save the file")
        else:
            logger.info(f"\n  Generation ended with status: {final.status}")

        logger.info(f"\n  Notebook ID: {nb.id}")
        logger.info("  (Notebook kept for review - delete manually when done)")

    logger.info("\n=== Done! ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.info("Usage: python research-to-podcast.py 'Your research topic'")
        logger.info("Example: python research-to-podcast.py 'renewable energy trends 2024'")
        sys.exit(1)

    topic = " ".join(sys.argv[1:])
    asyncio.run(main(topic))
