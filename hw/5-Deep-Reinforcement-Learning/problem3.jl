############
# Question 3
############
using DMUStudent.HW5: HW5, mc
using CommonRLInterface
using Flux
using Plots
using CommonRLInterface.Wrappers: QuickWrapper
using JLD2


# The following are some basic components needed for DQN
# ai generated save and load funcs



function save_model(Q, filename="best_q.jld2")
    model_state = Flux.state(Q)
    @save filename model_state
end

function load_model(filename="best_q.jld2")
    Q = Chain(Dense(2, 128, relu),
            # Dense(128, 128, relu),
              Dense(128, 5))
    @load filename model_state
    Flux.loadmodel!(Q, model_state)
    return Q
end
# Override to a discrete action space, and position and velocity observations rather than the matrix.
env = QuickWrapper(HW5.mc,
                   actions=[-1.0, -0.5, 0.0, 0.5, 1.0],
                   observe=mc->observe(mc)[1:2]
                  )

# create your loss function for Q training here
function loss(Q, Q_target, s, a_ind, r, sp, done)
    if done
        target_Q = r
    else
        target_Q = r + 0.99f0 * maximum(Q_target(sp))
    end
    return (target_Q - Q(s)[a_ind])^2 # Q learning loss i think?
end

function dqn(env)
    # This network should work for the Q function - an input is a state; the output is a vector containing the Q-values for each action 
    Q = Chain(Dense(2, 128, relu),
            # Dense(128, 128, relu),
              Dense(128, length(actions(env))))

    opt = Flux.setup(Adam(0.0005), Q)


    # We can create 1 tuple of experience like this
    s = observe(env)
    a_ind = 1 # action index - the index, rather than the actual action itself, will be needed in the loss function
    r = act!(env, actions(env)[a_ind])
    sp = observe(env)
    done = terminated(env)

    experience_tuple = (s, a_ind, r, sp, done)

    # this container should work well for the experience buffer:
    buffer = [experience_tuple]
    # you will need to push more experience into it and randomly select data for training
    
    reset!(env) # NOTE: after each time the environment reaches a terminal state, you need to reset it

    # this is the fixed target Q network
    Q_target = deepcopy(Q)

    best_Q_params = nothing
    episodes = 30000
    copy_freq = 100
    num_samples_per_episode = 64 
    max_buffer = 30000
    max_return = -1000000
    
    # Track learning curve
    eval_episodes = Int[]
    eval_scores = Float64[]
    
    for episode in 1:episodes

        # sample the current pollicy
        for sample in 1:num_samples_per_episode
            s = observe(env)

            eps = max(0.05, 1.0 - episode / (episodes * 0.5))
            if rand() < eps
                a_ind = rand(1:length(actions(env)))
            else
                a_ind = argmax(Q(Float32.(s[1:2])))  # Remove device(), just use Float32
            end
            r = act!(env, actions(env)[a_ind])
            sp = observe(env)
            done = terminated(env)

            pushfirst!(buffer, (s, a_ind, r, sp, done))

            if length(buffer) > max_buffer
                pop!(buffer)
            end

            if done
                reset!(env)
            end
        end

        if episode % copy_freq == 0
            # # trying soft updatin of target.
            # tau = 0.01f0
            # for (p, p_target) in zip(Flux.trainables(Q), Flux.trainables(Q_target))
            #     p_target .= (1 - tau) .* p_target .+ tau .* p
            # end
            Q_target = deepcopy(Q)
        end

        # select some data from the buffer and train (you may have to adjust some things, and you will have to do this many times):
        for data in rand(buffer, 400)
            if length(buffer) < 1000 #dont wanna traing on crap
                continue
            end
            s = Float32.(data[1])
            a_ind = data[2]
            r = Float32(data[3])
            sp = data[5] ? zeros(Float32, 2) : Float32.(data[4])
            done = data[5]
            
            loss_value, grads = Flux.withgradient(loss, Q, Q_target, s, a_ind, r, sp, done)
            Flux.update!(opt, Q, grads[1])
        end

        if episode % 200 ==0
            Q_cpu = Q |> cpu
            ret = HW5.evaluate(s->actions(env)[argmax(Q_cpu(s[1:2]))], n_episodes=100)
            
            # Record for learning curve
            push!(eval_episodes, episode)
            push!(eval_scores, ret.score)
            
            # Save learning curve plot
            p = plot(eval_episodes, eval_scores, 
                     xlabel="Episode", 
                     ylabel="Score (100 episode avg)",
                     title="DQN Learning Curve",
                     legend=false,
                     linewidth=2,
                     marker=:circle,
                     markersize=3)
            savefig(p, "learning_curve.png")

            if ret.score > max_return
                #save Q somehow
                best_Q_params = Flux.state(Q)
                @save "best_q.jld2" best_Q_params
                println("NEw max!", ret.score)
                max_return = ret.score
            end
            
            println("Episode $episode: score = $(ret.score)")
        end

    end
    

    p = plot(eval_episodes, eval_scores, 
             xlabel="Episode", 
             ylabel="Score (100 episode avg)",
             title="DQN Learning Curve - Final",
             legend=false,
             linewidth=2,
             marker=:circle,
             markersize=3)
    savefig(p, "learning_curve_final.png")
    
    # Also save the data as JLD2 in case you want to replot later
    @save "learning_curve_data.jld2" eval_episodes eval_scores
    
    if best_Q_params !== nothing
        Flux.loadmodel!(Q, best_Q_params)
    end
    
    return Q |> cpu
end

Q = dqn(env)

HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))], n_episodes=100) # you will need to remove the n_episodes=100 keyword argument and add your email as a positional argument to create a json file; evaluate needs to run 10_000 episodes to produce a json

#----------
# Rendering
#----------

# You can show an image of the environment like this (use ElectronDisplay if running from REPL):
display(render(env))

# The following code allows you to render the value function
using Plots
xs = -3.0f0:0.1f0:3.0f0
vs = -0.3f0:0.01f0:0.3f0
heatmap(xs, vs, (x, v) -> maximum(Q([x, v])), xlabel="Position (x)", ylabel="Velocity (v)", title="Max Q Value")
