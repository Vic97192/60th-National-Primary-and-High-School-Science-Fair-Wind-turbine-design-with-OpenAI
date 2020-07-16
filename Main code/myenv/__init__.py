from gym.envs.registration import register
register(
    id='MyHotterColder-v0',
    entry_point='myenv.my_hotter_colder:MyHotterColder',
)
