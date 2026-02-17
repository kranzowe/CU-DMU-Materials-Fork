using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate, stateindex
using POMDPTools: render
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean, std
using BenchmarkTools: @btime
using LinearAlgebra

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



struct ThreadedMySolverThingy <: POMDPs.Solver
    depth::Int
    max_rollout_steps::Int
    c::Float64
    beta::Float64
    eps::Float64
end

struct ThreadedSolverPolicy <: POMDPs.Policy
    # persistent state that carries between steps
    N::Vector{Matrix{Int}}
    Q::Vector{Matrix{Float64}}
    policy_function::Function # for da rollout

    expanded::Vector{BitVector}
    env::DenseGridWorld
    solver::ThreadedMySolverThingy
    num_threads::Int

end

function POMDPs.solve(solver::ThreadedMySolverThingy, m::DenseGridWorld)
    num_threads = Threads.nthreads()
    
    num_states = length(states(m))
    N = [zeros(Int, num_states, 4) for _ in 1:num_threads]
    Q = [zeros(Float64, num_states, 4)  for _ in 1:num_threads]
    expanded = [falses(num_states)  for _ in 1:num_threads]
    
    # idk what other precomputation to do...
    # okay gonna try just random rollouts till we run outta time

    # start = time_ns()

    # while time_ns() < start + 30_000_000
    #     # pick random state and action
    #     si = rand(1:num_states)
    #     s = states(m)[si]
    #     ai = rand(1:4)
    #     N[si,ai] += 1
    #     r = rollout(m, heuristic_policy, s, solver.max_rollout_steps, solver.eps, num_threads)
    #     Q[si, ai] += (r - Q[si, ai]) / N[si, ai]

    # end

    
    return ThreadedSolverPolicy(N, Q, heuristic_policy, expanded, m, solver, num_threads)
end

function POMDPs.action(policy::ThreadedSolverPolicy, s)
    si = stateindex(policy.env, s)
    
    start = time_ns()

    Threads.@threads for tid in 1:policy.num_threads
        while time_ns() < start + 35_000_000
            threaded_solver_simulate!(policy, s, tid)
        end
    end

    # gotta merge... this is tricky. needed some AI help for sure here
    # only mergin for this state
    Qm = zeros(Float64, 4)
    Nm = zeros(Int64, 4)

    for tid in 1:policy.num_threads
        for ai in 1:4
            n = policy.N[tid][si, ai]
            Nm[ai] += n
            Qm[ai] += n * policy.Q[tid][si, ai]
        end
    end
        # gotta weighted average it so it dont get weird says ai
    for ai in 1:4
        if Nm[ai] > 0
            Qm[ai] /= Nm[ai]
        end
    end

    # okay julia is pretty fresh this is cool syntax
    ai = argmax(Qm)
    return actions(policy.env)[ai]
    
end

function threaded_solver_explore(p::ThreadedSolverPolicy, si::Int, tid::Int)
    Ns = p.N[tid][si,1] + p.N[tid][si,2] + p.N[tid][si,3] + p.N[tid][si,4]
    best_a = 1
    best_val = -Inf
    @inbounds for ai in 1:4
        val = p.Q[tid][si, ai] + p.solver.c * (Ns^p.solver.beta) / (1e-6 + sqrt(p.N[tid][si, ai]))
        if val > best_val
            best_val = val
            best_a = ai
        end
    end
    return best_a
end


function threaded_solver_simulate!(policy::ThreadedSolverPolicy, s::S, tid::Int ,d::Int64 = policy.solver.depth) where {S}
    if d <= 0
        return rollout(policy.env, policy.policy_function, s, policy.solver.max_rollout_steps, policy.solver.eps)
    end
    env, N, Q, c = policy.env, policy.N, policy.Q, policy.solver.c

    A, gamma = actions(env), discount(env)

    si = stateindex(env, s)
    if !policy.expanded[tid][si]
        policy.expanded[tid][si] = true
        return rollout(env, policy.policy_function, s, policy.solver.max_rollout_steps, policy.solver.eps)
    end

    ai = threaded_solver_explore(policy, si, tid)

    s_prime, r = @gen(:sp, :r)(env, s, actions(env)[ai])

    q = r + gamma * threaded_solver_simulate!(policy, s_prime, tid, d-1)

    N[tid][si, ai] += 1
    Q[tid][si,ai] += (q - Q[tid][si,ai])/N[tid][si,ai]

    return q
end


println("DETECTING THREADS: ", Threads.nthreads())

configs = [
    # Baseline
    (name="baseline",          depth=30, c=50.0, beta=0.18, steps=15, eps=0.2),

    # Tweak depth
    (name="depth_28",          depth=28, c=50.0, beta=0.18, steps=15, eps=0.2),
    (name="depth_33",          depth=33, c=50.0, beta=0.18, steps=15, eps=0.2),
    (name="depth_37",          depth=37, c=50.0, beta=0.18, steps=15, eps=0.2),

    # Tweak c
    (name="c_25",              depth=30, c=25.0,  beta=0.18, steps=15, eps=0.2),
    (name="c_75",              depth=30, c=75.0,  beta=0.18, steps=15, eps=0.2),
    (name="c_100",             depth=30, c=100.0, beta=0.18, steps=15, eps=0.2),
    (name="c_125",             depth=30, c=125.0, beta=0.18, steps=15, eps=0.2),

    # Tweak beta
    (name="beta_0.10",         depth=30, c=50.0, beta=0.10, steps=15, eps=0.2),
    (name="beta_0.14",         depth=30, c=50.0, beta=0.14, steps=15, eps=0.2),
    (name="beta_0.22",         depth=30, c=50.0, beta=0.22, steps=15, eps=0.2),
    (name="beta_0.26",         depth=30, c=50.0, beta=0.26, steps=15, eps=0.2),

    # Tweak rollout steps
    (name="steps_5",           depth=30, c=50.0, beta=0.18, steps=5,  eps=0.2),
    (name="steps_10",          depth=30, c=50.0, beta=0.18, steps=10, eps=0.2),
    (name="steps_20",          depth=30, c=50.0, beta=0.18, steps=20, eps=0.2),
    (name="steps_25",          depth=30, c=50.0, beta=0.18, steps=25, eps=0.2),

    # Tweak epsilon
    (name="eps_0.0",           depth=30, c=50.0, beta=0.18, steps=15, eps=0.0),
    (name="eps_0.05",          depth=30, c=50.0, beta=0.18, steps=15, eps=0.05),
    (name="eps_0.10",          depth=30, c=50.0, beta=0.18, steps=15, eps=0.10),
    (name="eps_0.3",           depth=30, c=50.0, beta=0.18, steps=15, eps=0.3),
    (name="eps_0.6",           depth=30, c=50.0, beta=0.18, steps=15, eps=0.6),
    (name="eps_0.8",           depth=30, c=50.0, beta=0.18, steps=15, eps=0.8),

    # Combined tweaks in promising directions
    (name="deeper_less_c",     depth=30, c=25.0,  beta=0.18, steps=15, eps=0.2),
    (name="deeper_more_steps", depth=30, c=50.0,  beta=0.18, steps=20, eps=0.2),
    (name="low_c_low_beta",    depth=30, c=25.0,  beta=0.14, steps=15, eps=0.2),
    (name="high_c_high_beta",  depth=30, c=75.0,  beta=0.22, steps=15, eps=0.2),
    (name="aggressive",        depth=30, c=25.0,  beta=0.14, steps=20, eps=0.2),
    (name="conservative",      depth=20, c=75.0,  beta=0.22, steps=10, eps=0.2),
]
# Warmup once with any config
warmup_solver = ThreadedMySolverThingy(5, 10, 200.0, 0.25, 0.3)
warmup_m = DenseGridWorld(seed=1)
warmup_policy = POMDPs.solve(warmup_solver, warmup_m)
POMDPs.action(warmup_policy, SA[35,35])

for cfg in configs
    println("Running: $(cfg.name)")
    solver = ThreadedMySolverThingy(cfg.depth, cfg.steps, cfg.c, cfg.beta, cfg.eps)
    result = HW3.evaluate(solver, "owen.kranz@colorado.edu", time = true, fname="threaded_solver_results_$(cfg.name).json")
    println("  Score: $(result.score)")
end


