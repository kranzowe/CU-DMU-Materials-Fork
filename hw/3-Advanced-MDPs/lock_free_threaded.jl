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

struct SharedMySolverThingy <: POMDPs.Solver
    depth::Int
    max_rollout_steps::Int
    c::Float64
    beta::Float64
    eps::Float64
end

struct SharedSolverPolicy <: POMDPs.Policy
    N::Matrix{Threads.Atomic{Int}}      # single shared tree
    Q::Matrix{Float64}                   # non-atomic, accept tiny races
    policy_function::Function
    expanded::Vector{Threads.Atomic{Int}} # atomic bool (0/1)
    env::DenseGridWorld
    solver::SharedMySolverThingy
end

function POMDPs.solve(solver::SharedMySolverThingy, m::DenseGridWorld)
    num_states = length(states(m))
    N = [Threads.Atomic{Int}(0) for _ in 1:num_states, _ in 1:4]
    Q = zeros(Float64, num_states, 4)
    expanded = [Threads.Atomic{Int}(0) for _ in 1:num_states]

    return SharedSolverPolicy(N, Q, heuristic_policy, expanded, m, solver)
end

function POMDPs.action(policy::SharedSolverPolicy, s)
    si = stateindex(policy.env, s)
    num_threads = Threads.nthreads()

    start = time_ns()
    Threads.@threads for _ in 1:num_threads
        while time_ns() < start + 36_000_000
            shared_simulate!(policy, s)
        end
    end

    # no merge needed — just read directly
    ai = argmax([policy.Q[si, a] for a in 1:4])
    return actions(policy.env)[ai]
end

function shared_explore(p::SharedSolverPolicy, si::Int)
    Ns = p.N[si,1][] + p.N[si,2][] + p.N[si,3][] + p.N[si,4][]
    best_a = 1
    best_val = -Inf
    for ai in 1:4
        n = p.N[si, ai][]
        val = p.Q[si, ai] + p.solver.c * (Ns^p.solver.beta) / (1e-6 + sqrt(n))
        if val > best_val
            best_val = val
            best_a = ai
        end
    end
    return best_a
end

function shared_simulate!(policy::SharedSolverPolicy, s::S, d::Int64=policy.solver.depth) where {S}
    if d <= 0
        return rollout(policy.env, policy.policy_function, s, policy.solver.max_rollout_steps, policy.solver.eps)
    end

    env = policy.env
    gamma = discount(env)
    si = stateindex(env, s)

    # atomic CAS for expansion — only one thread expands a node
    if Threads.atomic_cas!(policy.expanded[si], 0, 1) == 0
        return rollout(env, policy.policy_function, s, policy.solver.max_rollout_steps, policy.solver.eps)
    end

    ai = shared_explore(policy, si)
    s_prime, r = @gen(:sp, :r)(env, s, actions(env)[ai])
    q = r + gamma * shared_simulate!(policy, s_prime, d - 1)

    # lock-free update: tiny race on Q is acceptable
    n_new = Threads.atomic_add!(policy.N[si, ai], 1) + 1
    policy.Q[si, ai] += (q - policy.Q[si, ai]) / n_new

    return q
end



function next_filename(prefix="threaded_solver_results", dir=".")
    existing = filter(f -> startswith(f, prefix) && endswith(f, ".json"), readdir(dir))
    indices = Int[]
    for f in existing
        m = match(Regex("$(prefix)_(\\d+)\\.json"), f)
        if m !== nothing
            push!(indices, parse(Int, m.captures[1]))
        end
    end
    next_idx = isempty(indices) ? 1 : maximum(indices) + 1
    return "$(prefix)_$(lpad(next_idx, 4, '0')).json"
end

function load_best_score(best_file="best_score.txt")
    isfile(best_file) || return -Inf
    return parse(Float64, strip(read(best_file, String)))
end

function save_best_score(score, best_file="best_score.txt")
    open(best_file, "w") do f
        println(f, score)
    end
end

# Warmup
warmup_solver = ThreadedMySolverThingy(5, 10, 200.0, 0.25, 0.3)
warmup_m = DenseGridWorld(seed=1)
warmup_policy = POMDPs.solve(warmup_solver, warmup_m)
POMDPs.action(warmup_policy, SA[35,35])

function run_search(n_trials=200)
    best_score = load_best_score()
    println("Starting random search. Current best score: $best_score")

    for i in 1:n_trials
        cfg = random_config()
        fname = next_filename()

        println("\nTrial $i/$n_trials")
        println("  depth=$(cfg.depth), c=$(cfg.c), beta=$(cfg.beta), steps=$(cfg.steps), eps=$(cfg.eps)")
        println("  Saving to: $fname")

        solver = ThreadedMySolverThingy(cfg.depth, cfg.steps, cfg.c, cfg.beta, cfg.eps)
        result = HW3.evaluate(solver, "owen.kranz@colorado.edu", time=true, fname=fname)

        println("  Score: $(result.score)  (best so far: $best_score)")

        if result.score > best_score
            best_score = result.score
            save_best_score(best_score)
            println("  *** New best! Saved to best_score.txt ***")
        end
    end

    println("\nSearch complete. Best score achieved: $best_score")
end

run_search(200)

