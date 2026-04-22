using POMDPs
using DMUStudent.HW6
using POMDPTools: transition_matrices, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater, has_consistent_distributions
using QuickPOMDPs: QuickPOMDP
using POMDPModels: TigerPOMDP, TIGER_LEFT, TIGER_RIGHT, TIGER_LISTEN, TIGER_OPEN_LEFT, TIGER_OPEN_RIGHT
using NativeSARSOP: SARSOPSolver 
using Statistics
ENV["GKSwstype"] = "100"
using Plots
using LinearAlgebra
using Random
using Printf
using BasicPOMCP
# using POMDPGifs
# import Cairo, Fontconfig # needed to display properly

struct HW6Updater{M<:POMDP} <: Updater
    m::M
end

function POMDPs.update(up::HW6Updater, b::DiscreteBelief, a, o)
    m = up.m
    bp_vec = zeros(length(states(m)))
    b_vec = beliefvec(b)

    for sp in ordered_states(m)
        spi = stateindex(m, sp)
        pred_sp = 0.0
        for s in ordered_states(m)
            si = stateindex(m, s)
            pred_sp += T(m, s, a, sp) * b_vec[si]
        end
        bp_vec[spi] = Z(m, a, sp, o) * pred_sp
    end

    z = sum(bp_vec)
    if z > 0.0
        bp_vec ./= z
    else # good practice so no div 0
        bp_vec .= 1.0 / length(bp_vec)
    end

    return DiscreteBelief(m, bp_vec)
end

Z(m::POMDP, a, sp, o) = pdf(observation(m, a, sp), o)
# T(s' | s, a) can be programmed with
T(m::POMDP, s, a, sp) = pdf(transition(m, s, a), sp)
# POMDPs.transtion and POMDPs.observation return distribution objects. See the POMDPs.jl documentation for more details.

function POMDPs.initialize_belief(up::HW6Updater, distribution::Any)
    b_vec = zeros(length(states(up.m)))
    for s in states(up.m)
        b_vec[stateindex(up.m, s)] = pdf(distribution, s)
    end
    return DiscreteBelief(up.m, b_vec)
end

struct HW6AlphaVectorPolicy{A} <: Policy
    alphas::Vector{Vector{Float64}}
    alpha_actions::Vector{A}
end

function POMDPs.action(p::HW6AlphaVectorPolicy, b::DiscreteBelief)
    bv = beliefvec(b)
    vals = [dot(α, bv) for α in p.alphas]
    return p.alpha_actions[argmax(vals)]
end

beliefvec(b::DiscreteBelief) = b.b # this function may be helpful to get the belief as a vector in stateindex order


function qmdp_solve(m, discount=discount(m); tol=1e-4, max_iters=100)
    
    gamma = 0.99
    eps = 1e-4
    S = ordered_states(m)
    A = ordered_actions(m)
    T = transition_matrices(m)
    R = reward_vectors(m)
    Q = zeros(length(S), length(A))
    V = ones(length(S))
    V_last = ones(length(S))
    itr = 0 
    while itr < max_iters
        
        for i in 1:length(A)
            a = A[i]
            Q[]

        if maximum(abs.(V - V_last)) < eps


            return HW6AlphaVectorPolicy(alphas, acts)
        else
            V_last = V
            itr += 1
        end

    end
end



function evaluate_policy(m, p, up; n_episodes=5000, max_steps=500)
    sim = RolloutSimulator(max_steps=max_steps)
    returns = [simulate(sim, m, p, up) for _ in 1:n_episodes]
    μ = mean(returns)
    sem = std(returns) / sqrt(n_episodes)
    return μ, sem
end

function extract_alphas(policy)
    if hasproperty(policy, :alphas)
        return [Vector{Float64}(a) for a in getproperty(policy, :alphas)]
    else
        error("Could not extract alpha vectors from policy type $(typeof(policy)).")
    end
end

function plot_tiger_alphas(m, qmdp_p, sarsop_p; filename="tiger_alphas.png")
    q_alphas = qmdp_p.alphas
    s_alphas = extract_alphas(sarsop_p)

    iL = stateindex(m, TIGER_LEFT)
    iR = stateindex(m, TIGER_RIGHT)
    bs = range(0.0, 1.0, length=201)

    plt = plot(
        title="QMDP vs SARSOP",
        xlabel="b(TL)",
        ylabel="alpha vector values",
        legend=:outerright,
    )

    for (i, α) in enumerate(q_alphas)
        ys = [b * α[iL] + (1 - b) * α[iR] for b in bs]
        plot!(plt, bs, ys, lw=2, label="QMDP α$i")
    end

    for (i, α) in enumerate(s_alphas)
        ys = [b * α[iL] + (1 - b) * α[iR] for b in bs]
        plot!(plt, bs, ys, lw=2, ls=:dash, label="SARSOP α$i")
    end

    savefig(plt, filename)
    println("Saved alpha-vector plot to: $filename")
end

m = TigerPOMDP()

qmdp_p = qmdp_solve(m)
# Note: you can use the QMDP.jl package to verify that your QMDP alpha vectors are correct.
sarsop_p = solve(SARSOPSolver(), m)
up = HW6Updater(m)

plot_tiger_alphas(m, qmdp_p, sarsop_p)

q_mean, q_sem = evaluate_policy(m, qmdp_p, up; n_episodes=500, max_steps=500)
s_mean, s_sem = evaluate_policy(m, sarsop_p, up; n_episodes=500, max_steps=500)

println("QMDP Monte Carlo return: mean = $(round(q_mean, digits=3)), SEM = $(round(q_sem, digits=3))")
println("SARSOP Monte Carlo return: mean = $(round(s_mean, digits=3)), SEM = $(round(s_sem, digits=3))")

###################
# Problem 2: Cancer
###################

cancer = QuickPOMDP(
    states = [:HEALTHY, :INSITU, :INVASIVE, :DEATH],
    actions = [:WAIT, :TEST, :TREAT],
    observations = [:POS, :NEG],

    transition = function (s, a)
        if s == :HEALTHY
            return SparseCat([s, :INSITU], [0.98, 0.02])
        elseif s == :INSITU
            if a == :TREAT
                return SparseCat([s, :HEALTHY], [0.4, 0.6])
            else
                return SparseCat([s, :INVASIVE], [0.9, 0.1])
            end
        elseif s == :INVASIVE
            if a == :TREAT
                return SparseCat([s, :HEALTHY, :DEATH], [0.6, 0.2, 0.2])
            else
                return SparseCat([s, :DEATH], [0.4, 0.6])
            end
        else # :DEATH
            return Deterministic(:DEATH)
        end
    end,

    # use (a, sp) form to match observation(m, a, sp) calls in your updater
    observation = function (a, sp)
        if a == :TEST
            if sp == :HEALTHY
                return SparseCat([:POS, :NEG], [0.05, 0.95])
            elseif sp == :INSITU
                return SparseCat([:POS, :NEG], [0.8, 0.2])
            elseif sp == :INVASIVE
                return Deterministic(:POS)
            else # :DEATH
                return Deterministic(:NEG)
            end
        elseif a == :TREAT
            if sp == :INSITU || sp == :INVASIVE
                return Deterministic(:POS)
            else
                return Deterministic(:NEG)
            end
        else # :WAIT
            return Deterministic(:NEG)
        end
    end,

    reward = function (s, a)
        if s == :DEATH
            return 0.0
        elseif a == :WAIT
            return 1.0
        elseif a == :TEST
            return 0.8
        else # :TREAT
            return 0.1
        end
    end,

    discount = 0.99,
    initialstate = Deterministic(:HEALTHY),
    isterminal = s -> s == :DEATH,
)

@assert has_consistent_distributions(cancer)

qmdp_p = qmdp_solve(cancer)
sarsop_p = solve(SARSOPSolver(), cancer)
up = HW6Updater(cancer)

heuristic = FunctionPolicy(function (b)
    p_invasive = pdf(b, :INVASIVE)
    p_insitu = pdf(b, :INSITU)
    p_perf = pdf(b, :HEALTHY)

    if p_invasive + p_insitu > 0.3 # combining works good
        return :TREAT
    elseif p_perf < 0.9 # not super healthy
        return :TEST
    else
        return :WAIT
    end
end)

@show mean(simulate(RolloutSimulator(), cancer, qmdp_p, up) for _ in 1:1000)
@show mean(simulate(RolloutSimulator(), cancer, heuristic, up) for _ in 1:1000)
@show mean(simulate(RolloutSimulator(), cancer, sarsop_p, up) for _ in 1:1000)

#####################
# Problem 4: LaserTag
#####################

m = LaserTagPOMDP()
println("Building updater...")
up = DiscreteUpdater(m)
println("Done.")

println("Running QMDP on LaserTag (this may take a moment)...")
@time laser_qmdp = qmdp_solve(m; tol=1e-3, max_iters=50)  # looser tolerance, fewer iters
println("QMDP done.")

function pomcp_solve(m, qmdp_rollout; tree_queries=10, c=)
    A = collect(actions(m))
    rollout_pol = FunctionPolicy(s -> begin
        si = stateindex(m, s)
        i_best = argmax([α[si] for α in qmdp_rollout.alphas])
        qmdp_rollout.alpha_actions[i_best]
    end)
    solver = POMCPSolver(
        tree_queries=tree_queries,
        c=c,
        default_action=A[1],
        estimate_value=FORollout(rollout_pol),
    )
    return solve(solver, m)
end

println("Solving POMCP (fast)...")
@time pomcp_fast = pomcp_solve(m, laser_qmdp; tree_queries=1, c=1)
println("Evaluating (10 episodes)...")
@time @show HW6.evaluate((pomcp_fast, up), n_episodes=10)

#----------------
# Visualization
# (all code below is optional)
#----------------

# You can make a gif showing what's going on like this:
# using POMDPGifs
# import Cairo, Fontconfig # needed to display properly

# makegif(m, qmdp_p, up, max_steps=30, filename="lasertag.gif")

# # You can render a single frame like this
# using POMDPTools: stepthrough, render
# using Compose: draw, PNG

# history = []
# for step in stepthrough(m, qmdp_p, up, max_steps=10)
#     push!(history, step)
# end
# displayable_object = render(m, last(history))
# # display(displayable_object) # <-this will work in a jupyter notebook or if you have vs code or ElectronDisplay
# draw(PNG("lasertag.png"), displayable_object)
