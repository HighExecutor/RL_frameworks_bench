def learn_problem(env, agent, episodes, max_steps, need_render):
    scores = []

    for e in range(episodes):
        state = env.reset()
        score = 0
        for i in range(max_steps):
            if need_render:
                env.render()
            action, action_values = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            score += reward
            agent.remember(state, action, reward, next_state, int(done))
            state = next_state
            agent.replay()
            if done:
                break
        agent.end_episode()
        print("episode: {}/{}, score: {}".format(e, episodes, score))
        scores.append(score)

    return scores

def learn_policy_problem(env, agent, episodes, max_steps, need_render):
    scores = []
    for e in range(episodes):
        state = env.reset()
        score = 0
        for i in range(max_steps):
            if need_render:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            agent.remember(state, action, reward)
            state = next_state
            if done:
                break
        agent.learn()
        agent.end_episode()
        print("episode: {}/{}, score: {}".format(e, episodes, score))
        scores.append(score)
    return scores

def result_learning(env, agent, max_steps):
    agent.epsilon = 0.0
    while True:
        state = env.reset()
        score = 0
        for i in range(max_steps):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            state = next_state
            if done:
                break
        print("score: {}".format(score))
