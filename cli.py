import argparse
from app.graph import app
from app.state import AppState

parser = argparse.ArgumentParser()
parser.add_argument("--topic", required=True)
parser.add_argument("--depth", type=int, default=3)
parser.add_argument("--follow_up", action="store_true")
parser.add_argument("--user_id", default="user1")
args = parser.parse_args()

try:
    inputs = AppState(topic=args.topic, depth=args.depth, follow_up=args.follow_up, user_id=args.user_id)
    result = app.invoke(inputs)
    print(result["final_brief"].model_dump_json())
except Exception as e:
    print(f"Error: {e}")