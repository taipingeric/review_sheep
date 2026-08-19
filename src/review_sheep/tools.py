"""A ready-made LangChain tool for embedding Review in another agent."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

from review_sheep.domain import Review
from review_sheep.manifest import SnapshotSource, create_manifest_review_agent
from review_sheep.report import render_report

logger = logging.getLogger(__name__)

__all__ = ["review_pull_request_tool"]


def review_pull_request_tool(
    model: str | BaseChatModel,
    source: SnapshotSource,
    *,
    instructions: str = "",
) -> BaseTool:
    """Build one LangChain tool that runs a full Manifest Review.

    Wraps fetching a pull-request snapshot, running every Lens, and
    rendering the Report behind a single tool call, so a caller can drop it
    into another agent's own ``tools=[...]`` list without wiring
    ``SnapshotSource``, ``create_manifest_review_agent``, and
    ``render_report`` together by hand.
    """
    agent = create_manifest_review_agent(
        source=source, model=model, instructions=instructions
    )

    @tool
    def review_pull_request(repo: str, number: int) -> str:
        """Review one GitHub pull request and return a Markdown Report.

        Args:
            repo: Repository in owner/name form, for example octocat/hello-world.
            number: Pull-request number.
        """
        logger.info("tools.review_pull_request.start repo=%s number=%d", repo, number)
        result = agent.review(repo=repo, number=number)
        if isinstance(result, Review):
            logger.info(
                "tools.review_pull_request.complete repo=%s number=%d findings=%d",
                repo,
                number,
                len(result.findings),
            )
            return render_report(result).text
        logger.info(
            "tools.review_pull_request.failed repo=%s number=%d operation=%s",
            repo,
            number,
            result.operation.value,
        )
        return (
            f"{result.operation.value} failed for "
            f"{result.repo}#{result.pull_request_number}: {result.message}"
        )

    return review_pull_request
