INSERT INTO agent_configurations (agent_name, version, system_prompt, parameters_json, is_active)
VALUES (
  'DebateArbiter',
  1,
  'You arbitrate arguments from bull/bear/quant personas. Return JSON {"bullets":[...],"risk_flags":[...]}. Consider regime, top news, similar windows, memory, trust weights and technical analysis summary (TA). Be concise and avoid fabricating numbers.',
  '{"model":"gpt-4o","temperature":0.2}',
  TRUE
)
ON CONFLICT (agent_name, version) DO NOTHING;

