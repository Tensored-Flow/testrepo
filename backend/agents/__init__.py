# Module: agents
# Owner: ___
# Status: IN PROGRESS
# Depends on: core/, integrations/, models/
#
# The pipeline agents. Each is a stateless function.
# All agents accept optional ConversationContext for shared working memory.

from backend.agents.triage import triage_agent
from backend.agents.snapshot import snapshot_agent
from backend.agents.analyst import analyst_agent, diagnose_failure
from backend.agents.optimizer import optimizer_agent
from backend.agents.test_designer import test_designer_agent
from backend.agents.validator import validator_agent
from backend.agents.planner import planner_agent, reorder_targets, get_strategy_hint, get_plan_directives
from backend.agents.report_generator import report_generator_agent, format_report_as_pr_body
