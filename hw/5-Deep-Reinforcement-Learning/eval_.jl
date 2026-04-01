using DMUStudent.HW5: HW5, mc
using CommonRLInterface
using Flux
using Plots
using CommonRLInterface.Wrappers: QuickWrapper
using JLD2


# The following are some basic components needed for DQN
# ai generated save and load funcs




function load_model(filename="best_q.jld2")
    Q = Chain(Dense(2, 128, relu),
            # Dense(128, 128, relu),
              Dense(128, 5))
    @load filename best_Q_params
    Flux.loadmodel!(Q, best_Q_params)
    return Q
end
# Overrid
Q = load_model("best_q.jld2")

HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))], "owen.kranz@colorado.edu") # you will need to remove the n_episodes=100 keyword argument and add your email as a positional argument to create a json file; evaluate needs to run 10_000 episodes to produce a json

#----------