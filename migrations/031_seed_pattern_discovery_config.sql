INSERT INTO agent_configurations (agent_name, version, system_prompt, parameters_json, is_active)
VALUES (
  'PatternDiscoveryAgent',
  1,
  'You help formalize selection rules for recurring pre-move contexts. Keep rules simple and robust.',
  '{"min_match_count":8, "min_success_rate":0.65, "max_p_value":0.05}',
  TRUE
)
ON CONFLICT (agent_name, version) DO NOTHING;

