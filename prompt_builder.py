def Build(memory):
    system_prompt = f"""
Your Character setting:

## CORE IDENTITY (must strictly follow):
{chr(10).join(memory["agent_state"]["identity"])}

## VALUES (behavior guidance):
{chr(10).join(memory["agent_state"]["values"])}

## MOTIVATION (drives behavior):
{chr(10).join(memory["agent_state"]["motivation"])}

## cognative_style (how to think and respond):
{chr(10).join(memory["agent_state"]["cognative_style"])}


## BEHAVIOR RULES (highest priority):
1. you must maintain the internal state above
2. stay within knowledge level
3. respond naturally to user
4. NEVER asks what to do next! JUST chat with user!


## LONG-TERM USER MEMORY:
{chr(10).join(memory["user_state"]["facts"])}
"""
    return system_prompt