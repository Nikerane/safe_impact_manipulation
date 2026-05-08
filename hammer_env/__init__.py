from gymnasium.envs.registration import register


register(
    id="FrankaHammer-v0",
    entry_point="hammer_env.env:FrankaHammerEnv",
)

