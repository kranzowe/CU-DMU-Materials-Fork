using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate, stateindex
using POMDPTools: render
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean, std
using BenchmarkTools: @btime
using LinearAlgebra
##############
# Instructions
##############
#=

This starter code is here to show examples of how to use the HW3 code that you
can copy and paste into your homework code if you wish. It is not meant to be a
fill-in-the blank skeleton code, so the structure of your final submission may
differ from this considerably.

Please make sure to update DMUStudent to gain access to the HW3 module.

=#

############
# Question 3
############

m = HW3.DenseGridWorld(seed=3)

function rollout(mdp, policy_function, s0, max_steps=100, eps=0)
    r_total = 0.0
    t=0
    s = s0
    while !isterminal(mdp, s) && t < max_steps
        a = policy_function(mdp, s, eps)
        s, r = @gen(:sp, :r)(mdp, s, a)
        r_total += discount(mdp)^t * r
        t += 1
    end
    return r_total # replace this with the reward
end

function rand_policy(m, s, eps)
    # put a smarter heuristic policy here
    return rand(actions(m))
end

function heuristic_policy(m, s, eps)
    if rand() < eps
        return rand(actions(m))
    end
    
    # modulo to nearest 20, clamp
    target_x = clamp(round(Int, s[1] / 20) * 20, 20, 80)
    target_y = clamp(round(Int, s[2] / 20) * 20, 20, 80)
    
    dx = target_x - s[1]
    dy = target_y - s[2]
    
    # ai cleanup
    if abs(dx) >= abs(dy)
        return dx > 0 ? :right : :left
    else
        return dy > 0 ? :up : :down
    end
end


# This code runs monte carlo simulations: you can calculate the mean and standard error from the results
num_runs = 400
results = [rollout(m, rand_policy, rand(initialstate(m))) for _ in 1:num_runs]

@show mean_results = mean(results)
@show std_results = std(results)
println("Computed SEM is: ", 1/sqrt(num_runs) * std_results)


results = [rollout(m, heuristic_policy, rand(initialstate(m))) for _ in 1:num_runs]
println("Min reward: ", minimum(results))
println("Max reward: ", maximum(results))
@show mean_results = mean(results)
@show std_results = std(results)
println("Computed SEM is: ", 1/sqrt(num_runs) * std_results)
############
# Question 4
############

m = DenseGridWorld(seed=4)

# S = statetype(m)
# A = actiontype(m)


mutable struct Policy{S,A}
    # implied from the pseudo i think i got everything lol
    N::Dict{Tuple{S, A}, Int}
    Q::Dict{Tuple{S, A}, Float64}
    T::Dict{Tuple{S, A, S}, Int}
    policy_function::Function # for da rollout
    max_rollout_steps::Int64
    env::DenseGridWorld
    max_itr::Int64
    depth::Int64
    c::Float64
    beta::Float64
end

mutable struct FasterPolicy
    # implied from the pseudo i think i got everything lol
    N::Array{Int, 2}
    Q::Array{Float64, 2}
    # remove T cuz that only for viz i think
    policy_function::Function # for da rollout
    max_rollout_steps::Int64
    env::DenseGridWorld
    max_itr::Int64
    depth::Int64
    c::Float64
    beta::Float64
    expanded::BitVector
    eps::Float64
end

# SOLVER ATTEMPT
struct MySolverThingy <: POMDPs.Solver
    depth::Int
    max_rollout_steps::Int
    c::Float64
    beta::Float64
    eps::Float64
end

struct SolverPolicy <: POMDPs.Policy
    # persistent state that carries between steps
    N::Matrix{Int}
    Q::Matrix{Float64}
    policy_function::Function # for da rollout

    expanded::BitVector
    env::DenseGridWorld
    solver::MySolverThingy

end

function POMDPs.solve(solver::MySolverThingy, m::DenseGridWorld)
    num_states = length(states(m))
    N = zeros(Int, num_states, 4)
    Q = zeros(Float64, num_states, 4)
    expanded = falses(num_states)
    
    # idk what other precomputation to do...
    # okay gonna try just random rollouts till we run outta time

    start = time_ns()

    while time_ns() < start + 30_000_000
        # pick random state and action
        si = rand(1:num_states)
        s = states(m)[si]
        ai = rand(1:4)
        N[si,ai] += 1
        r = rollout(m, heuristic_policy, s, solver.max_rollout_steps, solver.eps)
        Q[si, ai] += (r - Q[si, ai]) / N[si, ai]

    end

    
    return SolverPolicy(N, Q, heuristic_policy, expanded, m, solver)
end

function POMDPs.action(policy::SolverPolicy, s)
    si = stateindex(policy.env, s)
    
    start = time_ns()
    while time_ns() < start + 40_000_000
        solver_simulate!(policy, s)
    end
    
    ai = argmax(a -> policy.Q[si, a], 1:4)
    return actions(policy.env)[ai]
end


function monte_carlo_tree_search(policy::Policy, s::S, visualize::Bool) where {S}

    start = time_ns()
    # while time_ns() < start + 40_000_000 # you can replace the above line with this if you want to limit this loop to run within 40ms
    # 
    for k in 1:policy.max_itr
        simulate!(policy, s)
    end

    if visualize
        inchrome(visualize_tree(policy.Q, policy.N, policy.T, s))
    end

    # okay julia is pretty fresh this is cool syntax
    return argmax(a -> policy.Q[s,a], actions(policy.env))
end

function fast_monte_carlo_tree_search(policy::FasterPolicy, s::S) where {S}

    start = time_ns()
    
    #while time_ns() < start + 40_000_000 # you can replace the above line with this if you want to limit this loop to run within 40ms
    # 
    for k in 1:policy.max_itr
        fast_simulate!(policy, s)
    end

    # okay julia is pretty fresh this is cool syntax
    si = stateindex(policy.env, s)
    ai = argmax(a -> policy.Q[si,a], 1:4)
    return actions(policy.env)[ai]

end


function explore(p::Policy, s::S) where {S}
    Ns = sum(a -> get(p.N, (s,a), 0), actions(p.env))
    return argmax(a -> get(p.Q, (s,a), 0.0) + (p.c * (Ns^p.beta)/(1e-6 + sqrt(get(p.N, (s,a), 0)))), actions(p.env))
end

function fast_explore(p::FasterPolicy, si::Int)
    Ns = p.N[si,1] + p.N[si,2] + p.N[si,3] + p.N[si,4]
    best_a = 1
    best_val = -Inf
    @inbounds for ai in 1:4
        val = p.Q[si, ai] + p.c * (Ns^p.beta) / (1e-6 + sqrt(p.N[si, ai]))
        if val > best_val
            best_val = val
            best_a = ai
        end
    end
    return best_a
end

function solver_explore(p::SolverPolicy, si::Int)
    Ns = p.N[si,1] + p.N[si,2] + p.N[si,3] + p.N[si,4]
    best_a = 1
    best_val = -Inf
    @inbounds for ai in 1:4
        val = p.Q[si, ai] + p.solver.c * (Ns^p.solver.beta) / (1e-6 + sqrt(p.N[si, ai]))
        if val > best_val
            best_val = val
            best_a = ai
        end
    end
    return best_a
end

function simulate!(policy::Policy, s::S, d::Int64 = policy.depth) where {S}
    if d <= 0
        return rollout(policy.env, policy.policy_function, s, policy.max_rollout_steps)
    end
    #tryna keep it like the pseudocode as much as possible
    env, N, Q, T, c = policy.env, policy.N, policy.Q, policy.T, policy.c

    A, gamma = actions(env), discount(env)

    if !haskey(N, (s, first(A)))
        for a in A
            N[(s, a)] = 0
            Q[(s,a)] = 0.0
        end
        return rollout(env, policy.policy_function, s, policy.max_rollout_steps)
    end

    a = explore(policy, s)

    s_prime, r = @gen(:sp, :r)(env, s, a)

    q = r + gamma * simulate!(policy, s_prime, d-1)

    N[(s, a)] += 1
    Q[(s,a)] += (q - Q[(s,a)])/N[(s,a)] # backs up Q apparent i dont get it
    T[(s,a,s_prime)] = get(T, (s,a,s_prime), 0) + 1
    
    return q
end

function fast_simulate!(policy::FasterPolicy, s::S, d::Int64 = policy.depth) where {S}
    if d <= 0
        return rollout(policy.env, policy.policy_function, s, policy.max_rollout_steps, policy.eps)
    end
    #tryna keep it like the pseudocode as much as possible
    env, N, Q, c = policy.env, policy.N, policy.Q, policy.c

    A, gamma = actions(env), discount(env)

    si = stateindex(env, s)
    if !policy.expanded[si] # tracks expansion
        policy.expanded[si] = true
        return rollout(env, policy.policy_function, s, policy.max_rollout_steps, policy.eps)
    end

    ai = fast_explore(policy, si)

    s_prime, r = @gen(:sp, :r)(env, s, actions(env)[ai])

    q = r + gamma * fast_simulate!(policy, s_prime, d-1)

    N[si, ai] += 1
    Q[si,ai] += (q - Q[si,ai])/N[si,ai] # backs up Q apparent i dont get it

    return q
end

function solver_simulate!(policy::SolverPolicy, s::S, d::Int64 = policy.solver.depth) where {S}
    if d <= 0
        return rollout(policy.env, policy.policy_function, s, policy.solver.max_rollout_steps, policy.solver.eps)
    end
    env, N, Q, c = policy.env, policy.N, policy.Q, policy.solver.c

    A, gamma = actions(env), discount(env)

    si = stateindex(env, s)
    if !policy.expanded[si]
        policy.expanded[si] = true
        return rollout(env, policy.policy_function, s, policy.solver.max_rollout_steps, policy.solver.eps)
    end

    ai = solver_explore(policy, si)

    s_prime, r = @gen(:sp, :r)(env, s, actions(env)[ai])

    q = r + gamma * solver_simulate!(policy, s_prime, d-1)

    N[si, ai] += 1
    Q[si,ai] += (q - Q[si,ai])/N[si,ai]

    return q
end


function select_action(m, s, visualize=false)

    n = Dict{Tuple{statetype(m), actiontype(m)}, Int}()
    q = Dict{Tuple{statetype(m), actiontype(m)}, Float64}()
    t = Dict{Tuple{statetype(m), actiontype(m), statetype(m)}, Int}()
    max_rollout_steps = 20
    max_itr = 7
    search_depth = 2
    c = 100.0 # from the rollout of # 1
    beta = 0.25
    policy = Policy(n, q, t, rand_policy, max_rollout_steps, m, max_itr, search_depth, c, beta)

 
    return monte_carlo_tree_search(policy, s, visualize)
end
function fast_select_action(m, s, _)

    num_states = length(states(m))
    n = zeros(Int, num_states, 4)
    q = zeros(Float64, num_states, 4)
    max_rollout_steps = 20
    max_itr = 1000 # NOT USED anymore 
    search_depth = 30
    c = 1.0*(100 + 250) # from the rollout of # 1
    beta = 0.25
    expanded = falses(num_states)
    eps = 0.1
    policy = FasterPolicy(n, q, rand_policy, max_rollout_steps, m, max_itr, search_depth, c, beta, expanded, eps)

 
    return fast_monte_carlo_tree_search(policy, s)
end


fast_select_action(m, SA[35,35])


#### answer prob 4:
#select_action(m, SA[19,19], true)
##### 

###### Q5 answer
# use the code below to evaluate the MCTS policy
println("RESULTS FOR SLOW AND FAST")

results = [rollout(m, fast_select_action, rand(initialstate(m)), 100) for _ in 1:100]
@show mean_results = mean(results)
@show std_results = std(results)
println("Computed SEM is: ", 1/sqrt(num_runs) * std_results)
