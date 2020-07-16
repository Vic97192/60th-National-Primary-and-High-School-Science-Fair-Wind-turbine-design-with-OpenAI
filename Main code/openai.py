import gym
from gym import envs
 
# 列出可用的遊戲
print(envs.registry.all())
 
# 定義訓練的遊戲
env = gym.make('Pong-v0')   
# unwrapped 可以得到更多的資訊，可用 dir 得知可用 method
# env = env.unwrapped
 
# 查看環境中可用的 action 有多少個
print(env.action_space) 
# 查看環境中可用的 action 意義，不見得每個遊戲都有
print(env.unwrapped.get_action_meanings())
# 查看環境中可用的 observation 有多少個
print(env.observation_space)    
# 查看 observation 最高值
print(env.observation_space.high)   
# 查看 observation 最低值
print(env.observation_space.low)    
 
for i_episode in range(20):
    # 初始化
    observation = env.reset()
    for t in range(1000):
        # 畫圖
        env.render()
        # 學習到的動作，目前的設定是隨機產生
        action = env.action_space.sample()
        # 輸入動作，並得到狀態結果
        observation, reward, done, info = env.step(action)
        '''
        observation：環境狀態，像是影像 pixel，角度，角速度等，參數意義必須參照原始碼
        reward：越好的結果，會回饋越大的值，若不適用，也可從 observation 自行產生
        done：判斷是否達到 game over 的條件，像是生命已結束，或是已經超出範圍
        info：debug 用的資訊，不允許使用在學習上
        '''
        print(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
