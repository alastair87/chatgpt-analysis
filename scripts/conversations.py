import json
from collections import defaultdict
from datetime import datetime, timedelta, UTC
import tiktoken

# Set parameters
n_weeks = 4 # Time window in weeks
energy_per_query_wh = 0.3              # Energy per query in Wh (e.g., 0.02 Wh = 20 mWh)

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

# Output
print(f"Time window: last {n_weeks} week(s)")
print(f"Total messages: {total_messages}")
print(f"Total days active: {total_days}")
print(f"Average queries per active day: {average_per_day:.2f}")
print(f"Average query length (tokens): {average_tokens:.2f}")
print(f"Assumed energy usage per query: {energy_per_query_wh:.2f} Wh" )
print(f"Estimated energy usage: {energy_usage_wh:.3f} Wh")

