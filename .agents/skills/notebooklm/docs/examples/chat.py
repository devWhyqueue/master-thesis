"""Example: Chat with a notebook and manage conversations.

This example demonstrates:
1. Asking questions about notebook content
2. Follow-up questions in a conversation
3. Retrieving conversation history
4. Configuring chat behavior (response length, custom personas)

Prerequisites:
    - Authentication configured via `notebooklm auth` CLI command
    - Valid Google account with NotebookLM access
"""

import asyncio
import logging

from notebooklm import ChatGoal, ChatMode, ChatResponseLength, NotebookLMClient

logger = logging.getLogger(__name__)


async def main():
    """Demonstrate chat and conversation features."""

    async with await NotebookLMClient.from_storage() as client:
        # Create a notebook with some content
        logger.info("Setting up notebook with sources...")
        notebook = await client.notebooks.create("Python Learning")

        # Add a source for context
        source = await client.sources.add_url(
            notebook.id,
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
        )
        logger.info(f"Added source: {source.title}")

        # Give NotebookLM a moment to process the source
        logger.info("Waiting for source processing...")
        await asyncio.sleep(3)

        # =====================================================================
        # Basic Question/Answer
        # =====================================================================

        logger.info("\n--- Basic Q&A ---")

        # Ask a question about the notebook's content
        result = await client.chat.ask(
            notebook.id,
            "What are the main features of Python?",
        )

        logger.info("Question: What are the main features of Python?")
        logger.info(f"Answer: {result.answer[:500]}...")
        logger.info(f"Conversation ID: {result.conversation_id}")
        logger.info(f"Turn number: {result.turn_number}")

        # =====================================================================
        # Follow-up Questions (Conversation Threading)
        # =====================================================================

        logger.info("\n--- Follow-up Questions ---")

        # Use the same conversation_id for follow-up questions
        # This maintains context from previous exchanges
        followup = await client.chat.ask(
            notebook.id,
            "How does it compare to other programming languages?",
            conversation_id=result.conversation_id,  # Continue the conversation
        )

        logger.info("Follow-up: How does it compare to other programming languages?")
        logger.info(f"Answer: {followup.answer[:500]}...")
        logger.info(f"Is follow-up: {followup.is_follow_up}")
        logger.info(f"Turn number: {followup.turn_number}")

        # Another follow-up
        followup2 = await client.chat.ask(
            notebook.id,
            "What about for data science specifically?",
            conversation_id=result.conversation_id,
        )

        logger.info("\nFollow-up 2: What about for data science specifically?")
        logger.info(f"Answer: {followup2.answer[:400]}...")

        # =====================================================================
        # Conversation History
        # =====================================================================

        logger.info("\n--- Conversation History ---")

        # Get locally cached conversation turns
        turns = client.chat.get_cached_turns(result.conversation_id)
        logger.info(f"Cached turns in this conversation: {len(turns)}")
        for turn in turns:
            logger.info(f"  Turn {turn.turn_number}:")
            logger.info(f"    Q: {turn.query[:50]}...")
            logger.info(f"    A: {turn.answer[:50]}...")

        # Get conversation history from the API (all conversations)
        try:
            history = await client.chat.get_history(notebook.id, limit=10)
            logger.info(f"\nAPI conversation history: {type(history)}")
        except Exception as e:
            logger.info(f"Note: History retrieval returned: {e}")

        # =====================================================================
        # Configuring Chat Behavior
        # =====================================================================

        logger.info("\n--- Chat Configuration ---")

        # Method 1: Use predefined chat modes
        # Available modes: DEFAULT, LEARNING_GUIDE, CONCISE, DETAILED
        logger.info("Setting chat mode to LEARNING_GUIDE...")
        await client.chat.set_mode(notebook.id, ChatMode.LEARNING_GUIDE)

        # Ask a question with the new mode
        learning_result = await client.chat.ask(
            notebook.id,
            "Explain decorators in Python",
        )
        logger.info(f"Learning mode answer: {learning_result.answer[:400]}...")

        # Method 2: Fine-grained configuration
        # ChatGoal: DEFAULT, CUSTOM, LEARNING_GUIDE
        # ChatResponseLength: SHORTER, DEFAULT, LONGER
        logger.info("\nSetting custom chat configuration...")
        await client.chat.configure(
            notebook.id,
            goal=ChatGoal.DEFAULT,
            response_length=ChatResponseLength.SHORTER,
        )

        concise_result = await client.chat.ask(
            notebook.id,
            "What is Python used for?",
        )
        logger.info(f"Concise answer: {concise_result.answer[:300]}...")

        # Method 3: Custom persona with specific instructions
        logger.info("\nSetting custom persona...")
        await client.chat.configure(
            notebook.id,
            goal=ChatGoal.CUSTOM,
            response_length=ChatResponseLength.DEFAULT,
            custom_prompt="You are an experienced Python developer. "
            "Explain concepts with practical code examples. "
            "Focus on best practices and real-world usage.",
        )

        custom_result = await client.chat.ask(
            notebook.id,
            "How should I handle errors in Python?",
        )
        logger.info(f"Custom persona answer: {custom_result.answer[:500]}...")

        # =====================================================================
        # Source-Specific Questions
        # =====================================================================

        logger.info("\n--- Source-Specific Questions ---")

        # Get source IDs to target specific sources
        sources = await client.sources.list(notebook.id)
        if sources:
            source_ids = [sources[0].id]

            # Ask about specific sources only
            targeted_result = await client.chat.ask(
                notebook.id,
                "Summarize the key points from this source",
                source_ids=source_ids,  # Only use these sources for context
            )
            logger.info(f"Targeted answer: {targeted_result.answer[:400]}...")

        # =====================================================================
        # Cleanup
        # =====================================================================

        # Clear conversation cache (optional)
        client.chat.clear_cache(result.conversation_id)
        logger.info("\nConversation cache cleared")


if __name__ == "__main__":
    asyncio.run(main())
