import json
from collections import defaultdict
from datetime import datetime, timedelta, UTC
import tiktoken

# Set parameters
def analyze_conversations(n_weeks, energy_per_query_wh):
    # Calculate cutoff date
    cutoff_date = datetime.now(UTC).date() - timedelta(weeks=n_weeks)

    # Initialize tokenizer for GPT-4
    tokenizer = tiktoken.encoding_for_model("gpt-4o")

    # Load conversations from exported JSON file
    with open('conversations.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Track messages and tokens
    messages_per_day = defaultdict(int)
    total_tokens = 0
    total_messages = 0

    # Process user messages
    for conversation in data:
        for message in conversation.get('mapping', {}).values():
            if message.get('message') and message['message'].get('author', {}).get('role') == 'user':
                timestamp = message['message'].get('create_time')
                content = message['message'].get('content', {}).get('parts', [])
                if timestamp and content:
                    date = datetime.fromtimestamp(timestamp, UTC).date()
                    if date >= cutoff_date:
                        messages_per_day[date] += 1
                        total_messages += 1
                        total_tokens += sum(
                            len(tokenizer.encode(part)) for part in content if isinstance(part, str)
                        )

    # Compute averages and energy
    total_days = len(messages_per_day)
    average_per_day = total_messages / total_days if total_days > 0 else 0
    average_tokens = total_tokens / total_messages if total_messages > 0 else 0
    energy_usage_wh = total_messages * energy_per_query_wh

    return {
        "n_weeks": n_weeks,
        "total_messages": total_messages,
        "total_days": total_days,
        "average_per_day": average_per_day,
        "average_tokens": average_tokens,
        "energy_per_query_wh": energy_per_query_wh,
        "energy_usage_wh": energy_usage_wh
    }
