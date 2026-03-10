using Plots
using POMDPModels: SimpleGridWorld
using LinearAlgebra: I
using CommonRLInterface: render, actions, act!, observe, reset!, AbstractEnv, observations, terminated, clone
import POMDPTools
using SparseArrays
using Statistics: mean
using DMUStudent.HW4: HW4
ENV["GKSwstype"] = "100"

function sarsa_lambda_episode!(Q, env; ϵ=0.10, γ=0.99, α=0.05, λ=0.9)

    start = time()
    
    function policy(s)
        if rand() < ϵ
            return rand(actions(env))
        else
            return argmax(a->Q[(s, a)], actions(env))
        end
    end

    s = observe(env)
    a = policy(s)
    r = act!(env, a)
    sp = observe(env)
    hist = [s]
    N = Dict((s, a) => 0.0)

    while !terminated(env)
        ap = policy(sp)

        N[(s, a)] = get(N, (s, a), 0.0) + 1

        δ = r + γ*Q[(sp, ap)] - Q[(s, a)]

        for ((s, a), n) in N
            Q[(s, a)] += α*δ*n
            N[(s, a)] *= γ*λ
        end

        s = sp
        a = ap
        r = act!(env, a)
        sp = observe(env)
        push!(hist, sp)
    end

    N[(s, a)] = get(N, (s, a), 0.0) + 1
    δ = r - Q[(s, a)]

    for ((s, a), n) in N
        Q[(s, a)] += α*δ*n
        N[(s, a)] *= γ*λ
    end

    return (hist=hist, Q = copy(Q), time=time()-start)
end

function sarsa_lambda!(env; n_episodes=100, kwargs...)
    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
    episodes = []
    
    for i in 1:n_episodes
        reset!(env)
        push!(episodes, sarsa_lambda_episode!(Q, env;
                                              ϵ=max(0.01, 1-i/n_episodes),
                                              kwargs...))
    end
    
    return episodes
end

function q_lambda_episode!(Q, env; ϵ=0.10, γ=0.99, α=0.05, λ=0.9)

    start = time()
    
    #  need a functio for the greedy policy
    function greedy_policy(s)
        return argmax(a->Q[(s, a)], actions(env))
    end

    function rand_policy(s)
        # pull the if out later because need to zero N when rand policy is takes
        return rand(actions(env))
    end

    s = observe(env)
    a = rand()<ϵ ? rand_policy(s) : greedy_policy(s) #cool snytax
    r = act!(env, a)
    sp = observe(env)
    hist = [s]
    N = Dict((s, a) => 0.0)

    while !terminated(env)

        N[(s, a)] = get(N, (s, a), 0.0) + 1

        a_greedy = argmax(ap->Q[(sp, ap)], actions(env))

        δ = r + γ*Q[(sp, a_greedy)] - Q[(s, a)]

        for ((s_, a_), n) in N
            Q[(s_, a_)] += α*δ*n
            N[(s_, a_)] *= γ*λ
        end

        s = sp
        # choose the actualy action with eps greedy
        if rand() < ϵ
            a = rand_policy(s)
            # N gets emptied when random action taken
            empty!(N)
        else
            a = a_greedy # i dont need to re-call greedy policy?
        end

        r = act!(env, a)
        sp = observe(env)
        push!(hist, sp)
    end

    N[(s, a)] = get(N, (s, a), 0.0) + 1
    δ = r - Q[(s, a)]

    for ((s_, a_), n) in N
        Q[(s_, a_)] += α*δ*n
        N[(s_, a_)] *= γ*λ
    end

    return (hist=hist, Q = copy(Q), time=time()-start)
end

function q_lambda!(env; n_episodes=100, kwargs...)
    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
    episodes = []
    
    for i in 1:n_episodes
        reset!(env)
        push!(episodes, q_lambda_episode!(Q, env;
                                              ϵ=max(0.01, 1-i/n_episodes),
                                              kwargs...))
    end
    
    return episodes
end

m = HW4.gw
# m = SimpleGridWorld()
env = convert(AbstractEnv, m)
epsidoes = 300000
lambda_episodes = sarsa_lambda!(env, n_episodes=epsidoes, γ=0.99, α=0.01, λ=0.7);
q_lambda_episodes = q_lambda!(env, n_episodes=epsidoes, γ=0.99, α=0.01, λ=0.7);

# using Interact
# @manipulate for episode in 1:length(lambda_episodes), step in 1:maximum(ep->length(ep.hist), lambda_episodes)
#     ep = lambda_episodes[episode]
#     i = min(step, length(ep.hist))
#     POMDPTools.render(m, (s=ep.hist[i],), color=s->maximum(map(a->ep.Q[(s,a)], actions(env))))
# end

function evaluate(env, policy, n_episodes=1000, max_steps=1000, γ=1.0)
    returns = Float64[]
    for _ in 1:n_episodes
        t = 0
        r = 0.0
        reset!(env)
        s = observe(env)
        while !terminated(env)
            a = policy(s)
            r += γ^t*act!(env, a)
            s = observe(env)
            t += 1
        end
        push!(returns, r)
    end
    return returns
end

episodes = Dict("SARSA-λ"=>lambda_episodes, "Q-Learning-Lambda"=>q_lambda_episodes)


p1 = plot(xlabel="steps in environment", ylabel="avg return")
n = 10000
stop = epsidoes
for (name, eps) in episodes
    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
    xs = [0]
    ys = [mean(evaluate(env, s->argmax(a->Q[(s, a)], actions(env))))]
    for i in n:n:min(stop, length(eps))
        newsteps = sum(length(ep.hist) for ep in eps[i-n+1:i])
        push!(xs, last(xs) + newsteps)
        Q = eps[i].Q
        push!(ys, mean(evaluate(env, s->argmax(a->Q[(s, a)], actions(env)))))
    end    
    plot!(p1, xs, ys, label=name)
end
savefig("p1.png")

p2 = plot(xlabel="wall clock time", ylabel="avg return")
n = 10000
stop = epsidoes
for (name,eps) in episodes
    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
    xs = [0.0]
    ys = [mean(evaluate(env, s->argmax(a->Q[(s, a)], actions(env))))]
    for i in n:n:min(stop, length(eps))
        newtime = sum(ep.time for ep in eps[i-n+1:i])
        push!(xs, last(xs) + newtime)
        Q = eps[i].Q
        push!(ys, mean(evaluate(env, s->argmax(a->Q[(s, a)], actions(env)))))
    end    
    plot!(p2, xs, ys, label=name)
end

savefig("p2.png")