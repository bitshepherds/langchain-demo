import pytest
from conftest import agent, config


@pytest.mark.langsmith
class TestGoldenAnswers:
    """Golden/ground truth tests - verify expected outputs."""

    def test_calculator_add(self, agent, config):
        """Test basic addition."""
        result = agent.invoke(
            {"messages": [("user", "What is 5 + 7?")]},
            config
        )
        response = result["messages"][-1].content
        assert "12" in response, f"Expected '12' in response, got: {response}"

    def test_calculator_multiply(self, agent, config):
        """Test multiplication."""
        result = agent.invoke(
            {"messages": [("user", "What is 15 * 4?")]},
            config
        )
        response = result["messages"][-1].content
        assert "60" in response, f"Expected '60' in response, got: {response}"

    def test_calculator_large(self, agent, config):
        """Test larger calculation."""
        result = agent.invoke(
            {"messages": [("user", "What is 100 * 100?")]},
            config
        )
        response = result["messages"][-1].content
        assert "10000" in response or "10,000" in response, f"Expected '10000' in response, got: {response}"

    def test_get_current_time(self, agent, config):
        """Test current time tool."""
        result = agent.invoke(
            {"messages": [("user", "What time is it right now?")]},
            config
        )
        response = result["messages"][-1].content
        assert "sorry" not in response.lower(), f"Agent failed to answer: {response}"


@pytest.mark.langsmith
class TestTrajectory:
    """Trajectory tests - verify correct tools are called."""

    def test_calculator_tool_called(self, agent, config):
        """Verify calculator tool is called for math queries."""
        result = agent.invoke(
            {"messages": [("user", "What is 25 + 17?")]},
            config
        )
        messages = result["messages"]
        tool_names = []
        for m in messages:
            if hasattr(m, "tool_calls") and m.tool_calls:
                for tc in m.tool_calls:
                    tool_names.append(tc["name"])
        assert "calculator" in tool_names, f"Expected calculator to be called, got: {tool_names}"

    def test_wikipedia_tool_called(self, agent, config):
        """Verify wikipedia_search tool is called for factual queries."""
        result = agent.invoke(
            {"messages": [("user", "Who was the first US president?")]},
            config
        )
        messages = result["messages"]
        tool_names = []
        for m in messages:
            if hasattr(m, "tool_calls") and m.tool_calls:
                for tc in m.tool_calls:
                    tool_names.append(tc["name"])
        assert "wikipedia_search" in tool_names, f"Expected wikipedia_search to be called, got: {tool_names}"

    def test_time_tool_called(self, agent, config):
        """Verify get_current_time tool is called for time queries."""
        result = agent.invoke(
            {"messages": [("user", "What is the current time?")]},
            config
        )
        messages = result["messages"]
        tool_names = []
        for m in messages:
            if hasattr(m, "tool_calls") and m.tool_calls:
                for tc in m.tool_calls:
                    tool_names.append(tc["name"])
        assert "get_current_time" in tool_names, f"Expected get_current_time to be called, got: {tool_names}"
