using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate
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

function rollout(mdp, policy_function, s0, max_steps=100)
    r_total = 0.0
    t=0
    s = s0
    while !isterminal(mdp, s) && t < max_steps
        a = policy_function(mdp, s)
        s, r = @gen(:sp, :r)(mdp, s, a)
        r_total += discount(m)^t * r
        t += 1
    end
    return r_total # replace this with the reward
end

function rand_policy(m, s)
    # put a smarter heuristic policy here
    return rand(actions(m))
end

function heuristic_policy(m, s)
    # just go to 20, 20
    # use the module to go to the nearest multiple of 2020
    possible_terminal_states = [[20, 20], [20, 40], [40, 20], [40, 40]]
    diffs_terminal = [[0, 0], [0, 0], [0, 0], [0, 0]]
    norm_diffs = [0.0, 0.0, 0.0, 0.0]
    for (i, term_s) in enumerate(possible_terminal_states)
        diffs_terminal[i] = term_s - s
        norm_diffs[i] = norm(diffs_terminal[i])
    end
    #@show diffs_terminal
    #@show norm_diffs
    min_diff_index = argmin(norm_diffs)

    diff_s = possible_terminal_states[min_diff_index] - s

    abs_diff = diff_s .* diff_s
    max_element = argmax(abs_diff)

    if max_element == 1
        #move left or right
        if diff_s[1] > 0
            # state is right
            return :right
        else
            return :left
        end
    else
        #move up or down
        if diff_s[2] > 0
            # state is right
            return :up
        else
            return :down
        end
    end
    
    return rand(actions(m))
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

m = DenseGridWorld()

S = statetype(m)
A = actiontype(m)

# These would be appropriate containers for your Q, N, and t dictionaries:
n = Dict{Tuple{S, A}, Int}()
q = Dict{Tuple{S, A}, Float64}()
t = Dict{Tuple{S, A, S}, Int}()

# This is an example state - it is a StaticArrays.SVector{2, Int}
s = SA[19,19]
@show typeof(s)
@assert s isa statetype(m)

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


function explore(p::Policy, s::S) where {S}
    Ns = sum(a -> get(p.N, (s,a), 0), actions(p.env))
    return argmax(a -> get(p.Q, (s,a), 0.0) + (p.c * (Ns^p.beta)/(1e-6 + sqrt(get(p.N, (s,a), 0)))), actions(p.env))
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
# here is an example of how to visualize a dummy tree (q, n, and t should actually be filled in your mcts code, but for this we fill it manually)
# q[(SA[1,1], :right)] = 0.0
# q[(SA[2,1], :right)] = 0.0
# n[(SA[1,1], :right)] = 1
# n[(SA[2,1], :right)] = 0
# t[(SA[1,1], :right, SA[2,1])] = 1


# inchrome(visualize_tree(q, n, t, SA[1,1])) # use inbrowser(visualize_tree(q, n, t, SA[1,1]), "firefox") etc. if you want to use a different browser

# A starting point for the MCTS select_action function (a policy) which can be used for Questions 4 and 5
function select_action(m, s, visualize=false)

    n = Dict{Tuple{statetype(m), actiontype(m)}, Int}()
    q = Dict{Tuple{statetype(m), actiontype(m)}, Float64}()
    t = Dict{Tuple{statetype(m), actiontype(m), statetype(m)}, Int}()
    max_rollout_steps = 10 
    max_itr = 7
    search_depth = 7
    c = 2.0*(100 + 250) # from the rollout of # 1
    beta = 0.25
    policy = Policy(n, q, t, rand_policy, max_rollout_steps, m, max_itr, search_depth, c, beta)

 
    return monte_carlo_tree_search(policy, s, visualize)
end
# call a few to precompile
select_action(m, SA[35,35])
select_action(m, SA[35,35])
@btime select_action(m, SA[35,35]) # you can use this to see how much time your function takes to run. A good time is 10-20ms.


# answer prob 4:
select_action(m, SA[19,19], true)
# use the code below to evaluate the MCTS policy
@show results = [rollout(m, select_action, rand(initialstate(m)), 100) for _ in 1:100]

############
# Question 6
############

HW3.evaluate(select_action, "your.gradescope.email@colorado.edu")

# If you want to see roughly what's in the evaluate function (with the timing code removed), check sanitized_evaluate.jl

########
# Extras
########

# With a typical consumer operating system like Windows, OSX, or Linux, it is nearly impossible to ensure that your function *always* returns within 50ms. Do not worry if you get a few warnings about time exceeded.

# You may wish to call select_action once or twice before submitting it to evaluate to make sure that all parts of the function are precompiled.

# Instead of submitting a select_action function, you can alternatively submit a POMDPs.Solver object that will get 50ms of time to run solve(solver, m) to produce a POMDPs.Policy object that will be used for planning for each grid world.