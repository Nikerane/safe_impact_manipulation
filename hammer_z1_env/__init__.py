from gymnasium.envs.registration import register


register(
    id="Z1Hammer-v0",
    entry_point="hammer_z1_env.env:Z1HammerEnv",
)
