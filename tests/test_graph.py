from app.graph import app
from app.state import AppState
from pytest import fixture

@fixture
def sample_state():
    return AppState(topic="Test", depth=1, follow_up=False, user_id="test")

def test_graph(sample_state):
    result = app.invoke(sample_state)
    assert result["final_brief"].topic == "Test"